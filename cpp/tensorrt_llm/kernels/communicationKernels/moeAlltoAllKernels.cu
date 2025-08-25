/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tensorrt_llm/common/vec_dtypes.cuh"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cooperative_groups.h>

namespace tensorrt_llm::kernels::moe_a2a
{

// ============================================================================
// Helper Functions for Vectorized Memory Operations
// ============================================================================

// Helper to determine the largest vector size that can be used based on element byte size
__device__ __forceinline__ int get_max_vec_size(size_t element_byte_size)
{
    // Find the largest power of 2 that divides element_byte_size
    if (element_byte_size % 16 == 0)
        return 16;
    if (element_byte_size % 8 == 0)
        return 8;
    if (element_byte_size % 4 == 0)
        return 4;
    if (element_byte_size % 2 == 0)
        return 2;
    return 1; // Default to byte-by-byte copy for any other size
}

template <int VEC_SIZE>
__device__ __forceinline__ void warp_vectorized_copy_impl(void* dst, void const* src, size_t size, int lane_id)
{
    using flashinfer::vec_t;
    constexpr int WARP_SIZE = 32;

    char* dst_ptr = static_cast<char*>(dst);
    char const* src_ptr = static_cast<char const*>(src);

    size_t offset = lane_id * VEC_SIZE;

    // Main vectorized copy loop
    while (offset + VEC_SIZE <= size)
    {
        if constexpr (VEC_SIZE == 16)
        {
            vec_t<uint8_t, 16> vec_data{};
            vec_data.load(reinterpret_cast<uint8_t const*>(src_ptr + offset));
            vec_data.store(reinterpret_cast<uint8_t*>(dst_ptr + offset));
        }
        else if constexpr (VEC_SIZE == 8)
        {
            vec_t<uint8_t, 8> vec_data{};
            vec_data.load(reinterpret_cast<uint8_t const*>(src_ptr + offset));
            vec_data.store(reinterpret_cast<uint8_t*>(dst_ptr + offset));
        }
        else if constexpr (VEC_SIZE == 4)
        {
            vec_t<uint8_t, 4> vec_data{};
            vec_data.load(reinterpret_cast<uint8_t const*>(src_ptr + offset));
            vec_data.store(reinterpret_cast<uint8_t*>(dst_ptr + offset));
        }
        else if constexpr (VEC_SIZE == 2)
        {
            vec_t<uint8_t, 2> vec_data{};
            vec_data.load(reinterpret_cast<uint8_t const*>(src_ptr + offset));
            vec_data.store(reinterpret_cast<uint8_t*>(dst_ptr + offset));
        }
        else
        {
            dst_ptr[offset] = src_ptr[offset];
        }
        offset += WARP_SIZE * VEC_SIZE;
    }

    // Handle remainder byte by byte
    while (offset < size)
    {
        if (lane_id == 0)
        {
            dst_ptr[offset] = src_ptr[offset];
        }
        offset += 1;
    }
}

__device__ __forceinline__ void warp_vectorized_copy(
    void* dst, void const* src, size_t size, int lane_id, size_t element_byte_size)
{
    int vec_size = get_max_vec_size(element_byte_size);

    switch (vec_size)
    {
    case 16: warp_vectorized_copy_impl<16>(dst, src, size, lane_id); break;
    case 8: warp_vectorized_copy_impl<8>(dst, src, size, lane_id); break;
    case 4: warp_vectorized_copy_impl<4>(dst, src, size, lane_id); break;
    case 2: warp_vectorized_copy_impl<2>(dst, src, size, lane_id); break;
    default: warp_vectorized_copy_impl<1>(dst, src, size, lane_id); break;
    }
}

// ============================================================================
// Generic Dispatch Kernel Implementation
// One warp per token design:
// - Each CTA has 256 threads = 8 warps
// - Each warp independently processes one token and all its payloads
// - Better GPU utilization and reduced synchronization overhead
// ============================================================================

template <int TopK>
__global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [local_num_tokens, top_k]
    void const** src_data_ptrs,                                             // Array of source data pointers
    void** recv_buffer_ptrs,                                                // Array of receive buffer pointers
    size_t const* element_sizes,                                            // Array of element sizes
    size_t const* elements_per_token,                                       // Array of elements per token
    int num_payloads,                                                       // Number of payloads
    int max_tokens_per_rank,                                                // Maximum tokens per rank
    int* recv_counters, // [ep_size] atomic counters - each rank tracks its own
    int local_num_tokens, int ep_size)
{
    // Constants
    constexpr int WARPS_PER_BLOCK = 8; // 256 threads / 32 threads per warp
    constexpr int WARP_SIZE = 32;

    // Determine which warp this thread belongs to and which token it handles
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    int local_token_idx = global_warp_id;

    if (local_token_idx >= local_num_tokens)
    {
        return;
    }

    // Shared memory organized per warp
    __shared__ uint32_t already_allocated[WARPS_PER_BLOCK];
    __shared__ int dst_positions[WARPS_PER_BLOCK][kMaxTopK];

    // Initialize shared memory for this warp
    if (lane_id == 0)
    {
        already_allocated[warp_id_in_block] = 0;
    }
    __syncwarp();

    // Step 1: Determine unique destination ranks and allocate positions
    // Each lane in the warp handles different k values
    for (int k = lane_id; k < TopK; k += WARP_SIZE)
    {
        int expert_idx = token_selected_experts[local_token_idx * TopK + k];
        int target_rank = expert_idx % ep_size;

        // Check if we've already allocated space for this rank
        uint32_t mask = 1U << target_rank;
        uint32_t old_mask = atomicOr(&already_allocated[warp_id_in_block], mask);

        if (!(old_mask & mask))
        {
            // First time sending to this rank - allocate position
            dst_positions[warp_id_in_block][k] = atomicAdd(&recv_counters[target_rank], 1);
        }
        else
        {
            dst_positions[warp_id_in_block][k] = -1; // Mark as duplicate rank
        }
    }
    __syncwarp();

    // Step 2: Process each payload
    for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
    {
        void const* src_data = src_data_ptrs[payload_idx];
        void* recv_buffer = recv_buffer_ptrs[payload_idx];
        size_t element_size = element_sizes[payload_idx];
        size_t elements_per_tok = elements_per_token[payload_idx];

        // Only send to ranks that have selected experts for this token
        for (int k = 0; k < TopK; k++)
        {
            if (dst_positions[warp_id_in_block][k] >= 0)
            {
                int expert_idx = token_selected_experts[local_token_idx * TopK + k];
                int target_rank = expert_idx % ep_size;
                int dst_token_position = dst_positions[warp_id_in_block][k];

                // Calculate destination address in this payload's receive buffer
                char* dest_ptr = static_cast<char*>(get_payload_ptr_in_recv_buffer(
                    recv_buffer, target_rank, dst_token_position, max_tokens_per_rank, elements_per_tok, element_size));

                // Source data
                char const* src_ptr
                    = static_cast<char const*>(src_data) + local_token_idx * elements_per_tok * element_size;

                // Warp-level vectorized copy
                size_t copy_size = elements_per_tok * element_size;
                warp_vectorized_copy(dest_ptr, src_ptr, copy_size, lane_id, copy_size);
            }
        }
    }
}

// ============================================================================
// Launch Functions
// ============================================================================

void moe_a2a_dispatch_op(MoeA2ADispatchParams const& params)
{
    // Validate parameters
    TLLM_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK);
    TLLM_CHECK(params.ep_size > 0);
    TLLM_CHECK(params.local_num_tokens > 0);
    TLLM_CHECK(params.num_payloads > 0 && params.num_payloads <= kMaxPayloads);

    // Configure kernel launch
    constexpr int block_size = 256;
    constexpr int warp_size = 32;
    constexpr int warps_per_block = block_size / warp_size; // 8 warps per block
    int grid_size = (params.local_num_tokens + warps_per_block - 1) / warps_per_block;

    // Prepare arrays from payload descriptors
    void const* h_src_data_ptrs[kMaxPayloads];
    void* h_recv_buffer_ptrs[kMaxPayloads];
    size_t h_element_sizes[kMaxPayloads];
    size_t h_elements_per_token[kMaxPayloads];

    for (int i = 0; i < params.num_payloads; i++)
    {
        h_src_data_ptrs[i] = params.payloads[i].src_data;
        h_recv_buffer_ptrs[i] = params.payloads[i].recv_buffer;
        h_element_sizes[i] = params.payloads[i].element_size;
        h_elements_per_token[i] = params.payloads[i].elements_per_token;
    }

    // Launch generic kernel based on top_k
    switch (params.top_k)
    {
    case 2:
        moeA2ADispatchKernel<2><<<grid_size, block_size, 0, params.stream>>>(params.token_selected_experts,
            h_src_data_ptrs, h_recv_buffer_ptrs, h_element_sizes, h_elements_per_token, params.num_payloads,
            params.max_tokens_per_rank, params.recv_counters, params.local_num_tokens, params.ep_size);
        break;
    case 4:
        moeA2ADispatchKernel<4><<<grid_size, block_size, 0, params.stream>>>(params.token_selected_experts,
            h_src_data_ptrs, h_recv_buffer_ptrs, h_element_sizes, h_elements_per_token, params.num_payloads,
            params.max_tokens_per_rank, params.recv_counters, params.local_num_tokens, params.ep_size);
        break;
    case 8:
        moeA2ADispatchKernel<8><<<grid_size, block_size, 0, params.stream>>>(params.token_selected_experts,
            h_src_data_ptrs, h_recv_buffer_ptrs, h_element_sizes, h_elements_per_token, params.num_payloads,
            params.max_tokens_per_rank, params.recv_counters, params.local_num_tokens, params.ep_size);
        break;
    default: TLLM_CHECK_WITH_INFO(false, "Unsupported top_k value");
    }
}

} // namespace tensorrt_llm::kernels::moe_a2a
