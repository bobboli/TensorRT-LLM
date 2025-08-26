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
#include <cstdint>

namespace tensorrt_llm::kernels::moe_a2a
{

// ============================================================================
// Helper Functions for Vectorized Memory Operations
// ============================================================================

template <int VEC_SIZE>
__device__ void warp_vectorized_copy_impl(void* dst, void const* src, int size, int lane_id)
{
    using flashinfer::vec_t;

    uint8_t* dst_ptr = static_cast<uint8_t*>(dst);
    uint8_t const* src_ptr = static_cast<uint8_t const*>(src);

    int offset = lane_id * VEC_SIZE;

    // Vectorized copy loop - size must be divisible by VEC_SIZE
    int n_repeats = size / VEC_SIZE;
    
    for (int i = 0; i < n_repeats; ++i)
    {
        vec_t<uint8_t, VEC_SIZE> vec_data;
        vec_data.load(src_ptr + offset);
        vec_data.store(dst_ptr + offset);
        offset += warpSize * VEC_SIZE;
    }
}

__device__ void warp_vectorized_copy(void* dst, void const* src, int size, int lane_id)
{
    if (size % 16 == 0)
    {
        warp_vectorized_copy_impl<16>(dst, src, size, lane_id);
    } else if (size % 8 == 0) {
        warp_vectorized_copy_impl<8>(dst, src, size, lane_id);
    } else if (size % 4 == 0) {
        warp_vectorized_copy_impl<4>(dst, src, size, lane_id);
    } else if (size % 2 == 0) {
        warp_vectorized_copy_impl<2>(dst, src, size, lane_id);
    } else {
        warp_vectorized_copy_impl<1>(dst, src, size, lane_id);
    }
}

// ============================================================================
// Generic Dispatch Kernel Implementation
// One warp per token design:
// - Each CTA has 256 threads = 8 warps
// - Each warp independently processes one token and all its payloads
// - Better GPU utilization and reduced synchronization overhead
// ============================================================================

__global__ void moeA2ADispatchKernel(int32_t const* token_selected_experts, // [local_num_tokens, top_k]
    void const* src_data_ptrs[kMaxPayloads],                                // Array of source data pointers
    void* (*recv_buffers)[kMaxPayloads],                                    // [ep_size][kMaxPayloads] - pointer to array of payload buffers per rank
    int const payload_bytes_per_token[kMaxPayloads],                                // Array of bytes per token
    int num_payloads,                                                       // Number of payloads
    int max_tokens_per_rank,                                                // Maximum tokens per rank
    int* recv_counters, // [ep_size] atomic counters - each rank tracks its own
    int local_num_tokens, int rank_id, int ep_size, int top_k)
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

    uint64_t already_copied = 0;
    for (int k = 0; k < top_k; k++)
    {
        int expert_idx = token_selected_experts[local_token_idx * top_k + k];
        int target_rank = expert_idx % ep_size;

        if (already_copied & (1ULL << target_rank)) continue;

        int dst_token_idx = atomicAdd(&recv_counters[target_rank], 1);
        void* (&recv_buffer_ptrs)[kMaxPayloads] = recv_buffers[target_rank];


        for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
        {
            uint8_t const* src_data = static_cast<uint8_t const*>(src_data_ptrs[payload_idx]);
            uint8_t* dst_data = static_cast<uint8_t*>(recv_buffer_ptrs[payload_idx]);
            
            int bytes_per_token = payload_bytes_per_token[payload_idx];

            uint8_t* dst_ptr = dst_data + dst_token_idx * bytes_per_token;
            uint8_t const* src_ptr = src_data + local_token_idx * bytes_per_token;

            warp_vectorized_copy(dst_ptr, src_ptr, bytes_per_token, lane_id);
        }

        already_copied |= 1ULL << target_rank;
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
    int h_payload_bytes_per_token[kMaxPayloads];

    for (int i = 0; i < params.num_payloads; i++)
    {
        h_src_data_ptrs[i] = params.payloads[i].src_data;
        h_payload_bytes_per_token[i] = params.payloads[i].element_size * params.payloads[i].elements_per_token;
    }
    
    // Need to prepare recv_buffers - this should be passed in params
    // For now, assume params has recv_buffers field
    void* (*recv_buffers)[kMaxPayloads] = params.recv_buffers;


    moeA2ADispatchKernel<<<grid_size, block_size, 0, params.stream>>>(params.token_selected_experts,
        h_src_data_ptrs, recv_buffers, h_payload_bytes_per_token, params.num_payloads,
        params.max_tokens_per_rank, params.recv_counters, params.local_num_tokens, params.ep_rank, params.ep_size, params.top_k);
}

} // namespace tensorrt_llm::kernels::moe_a2a
