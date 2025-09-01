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
    
    while(offset < size)
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
    const KernelPointers ptrs,                                                     // Struct containing all kernel pointers
    int num_payloads,                                                       // Number of payloads
    int max_tokens_per_rank,                                                // Maximum tokens per rank
    int* send_counters, // [ep_size] atomic counters - each rank tracks its own
    int* send_indices, // [local_num_tokens, ep_size] send index tensor
    int* local_token_counter, // Atomic counter for completed tokens on this rank
    int local_num_tokens, int rank_id, int ep_size, int top_k)
{
    // Constants
    constexpr int WARPS_PER_BLOCK = 8; // 256 threads / 32 threads per warp
    constexpr int WARP_SIZE = 32;

    // Initialize completion_flags and local_token_counter at kernel start
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // printf("Rank %d: Initializing with %d local tokens\n", rank_id, local_num_tokens);
        // Thread 0 of block 0 initializes everything
        for (int i = 0; i < ep_size; i++) {
            // Use uncached store for initialization
            int* flag_addr = &ptrs.completion_flags[rank_id][i];
            asm volatile("st.global.wt.u32 [%0], %1;" :: "l"(flag_addr), "r"(0));
        }
    }
    // Ensure initialization is complete before proceeding
    __threadfence();
    
    // Debug: confirm kernel is running
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("Rank %d: Kernel launched with %d local tokens, ep_size=%d\n", 
               rank_id, local_num_tokens, ep_size);
    }

    // Determine which warp this thread belongs to and which token it handles
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int global_warp_id = blockIdx.x * WARPS_PER_BLOCK + warp_id_in_block;
    int local_token_idx = global_warp_id;

    if (local_token_idx >= local_num_tokens)
    {
         return;
    }
    
    // Debug: Track token processing
    // if (lane_id == 0) {
    //     printf("###Rank %d: Processing token %d\n", rank_id, local_token_idx);
    // }

    uint64_t already_copied = 0;
    for (int k = 0; k < top_k; k++)
    {
        int expert_idx = token_selected_experts[local_token_idx * top_k + k];
        int target_rank = expert_idx % ep_size;

        if (already_copied & (1ULL << target_rank)) continue;

        // Only one thread per warp should increment the counter
        int dst_token_idx;
        if (lane_id == 0) {
            dst_token_idx = atomicAdd(&send_counters[target_rank], 1);

            send_indices[local_token_idx * ep_size + target_rank] = dst_token_idx;
        }
        // Broadcast the index to all threads in the warp
        dst_token_idx = __shfl_sync(0xffffffff, dst_token_idx, 0);

        for (int payload_idx = 0; payload_idx < num_payloads; payload_idx++)
        {
            uint8_t const* src_data = static_cast<uint8_t const*>(ptrs.src_data_ptrs[payload_idx]);
            uint8_t* dst_data = static_cast<uint8_t*>(ptrs.recv_buffers[target_rank][payload_idx]);
            
            int bytes_per_token = ptrs.payload_bytes_per_token[payload_idx];

            // Account for source rank offset in the 3D recv buffer layout [source_ranks, tokens, elements]
            uint8_t* dst_ptr = dst_data + (rank_id * max_tokens_per_rank + dst_token_idx) * bytes_per_token;
            uint8_t const* src_ptr = src_data + local_token_idx * bytes_per_token;

            // Debug: print write info for first few tokens
            if (lane_id == 0 && local_token_idx < 3 && payload_idx == 3) { // payload 3 is token_final_scales
                float* src_float = (float*)src_ptr;
                printf("Rank %d token %d -> Rank %d pos %d: writing [%.0f, %.0f] to addr %p (base %p + offset %ld)\n", 
                       rank_id, local_token_idx, target_rank, dst_token_idx,
                       src_float[0], src_float[1], dst_ptr, dst_data, 
                       (rank_id * max_tokens_per_rank + dst_token_idx) * bytes_per_token);
            }

            warp_vectorized_copy(dst_ptr, src_ptr, bytes_per_token, lane_id);
        }

        already_copied |= 1ULL << target_rank;
    }

   // __threadfence_system();

    // Finished sending this token. Check if we're the last token to complete.
    if (lane_id == 0) {
        int completed_tokens = atomicAdd(local_token_counter, 1) + 1;
        // printf("+++Rank %d: Token %d completed, total completed: %d/%d\n", 
        //        rank_id, local_token_idx, completed_tokens, local_num_tokens);
        
        if (completed_tokens == local_num_tokens) {
            printf("Rank %d: Last token completed! Signaling to other ranks\n", rank_id);
            // This warp processed the last token! Ensure all writes are visible
            // __threadfence_system();
            
            // Signal completion to all ranks with proper release semantics
            for (int target_rank = 0; target_rank < ep_size; target_rank++) {
                // Set flag in target rank's completion flags array at position rank_id
                int* flag_addr = &ptrs.completion_flags[target_rank][rank_id];
                printf("***Rank %d: Setting completion flag for target rank %d at address %p\n",
                       rank_id, target_rank, flag_addr);
                // Use release store for cross-GPU visibility
                // According to PTX docs, we need memory synchronization for cross-scope ops
                // __threadfence_system();
                asm volatile("st.release.sys.u32 [%0], %1;" :: "l"(flag_addr), "r"(1));
            }
            
            // Now wait for all ranks to complete their dispatch
            for (int source_rank = 0; source_rank < ep_size; source_rank++) {
                // Busy wait until source_rank signals completion
                int* flag_ptr = &ptrs.completion_flags[rank_id][source_rank];
                int flag_value = 0;
                do {
                    // Use acquire load for cross-GPU visibility
                    asm volatile("ld.acquire.sys.u32 %0, [%1];" : "=r"(flag_value) : "l"(flag_ptr));
                    // if (flag_value == 0) {
                    //     printf("rank %d spin wait for source rank %d at address %p. flag_value %d\n", 
                    //            rank_id, source_rank, flag_ptr, flag_value);
                    // }
                } while (flag_value == 0);
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

    // Prepare kernel pointers struct
    KernelPointers kernel_ptrs = {}; // Zero-initialize
    
    // Fill source data pointers and payload sizes
    for (int i = 0; i < params.num_payloads; i++)
    {
        kernel_ptrs.src_data_ptrs[i] = params.payloads[i].src_data;
        kernel_ptrs.payload_bytes_per_token[i] = params.payloads[i].element_size * params.payloads[i].elements_per_token;
    }
    
    // Fill receive buffer pointers and completion flags
    for (int rank = 0; rank < params.ep_size; rank++)
    {
        for (int payload = 0; payload < params.num_payloads; payload++)
        {
            kernel_ptrs.recv_buffers[rank][payload] = params.recv_buffers[rank][payload];
        }
        kernel_ptrs.completion_flags[rank] = params.completion_flags[rank];
    }

    moeA2ADispatchKernel<<<grid_size, block_size, 0, params.stream>>>(params.token_selected_experts,
        kernel_ptrs, params.num_payloads,
        params.max_tokens_per_rank, params.send_counters, params.send_indices,
        params.local_token_counter, 
        params.local_num_tokens, params.ep_rank, params.ep_size, params.top_k);
}

} // namespace tensorrt_llm::kernels::moe_a2a
