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

#pragma once
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/kernels/quantization.h"

namespace tensorrt_llm::kernels::moe_a2a
{

// Configuration constants
static constexpr int kMaxExperts = 256; // Maximum number of experts per rank
static constexpr int kMaxTopK = 8;      // Maximum top-k experts per token
static constexpr int kWarpSize = 32;
static constexpr int kMaxPayloads = 8;  // Maximum number of different payload types

// Describes a single payload type to be communicated
struct PayloadDescriptor
{
    void const* src_data;      // Source data pointer [local_num_tokens, elements_per_token]
    void* recv_buffer;         // Contiguous receive buffer for this payload [ep_size * max_tokens * elements_per_token]
    size_t element_size;       // Size of each element in bytes
    size_t elements_per_token; // Number of elements per token (e.g., hidden_size, top_k)
};

// Dispatch phase parameters
struct MoeA2ADispatchParams
{
    // EP configuration
    int ep_size;           // Number of EP ranks
    int ep_rank;           // Current EP rank
    int local_num_experts; // Number of experts on this rank

                           // Token configuration
    int local_num_tokens;    // Number of tokens on this rank
    int max_tokens_per_rank; // Maximum tokens per rank for pre-allocation
    int hidden_size;         // Hidden dimension
    int top_k;               // Number of experts per token

    // Data type
    nvinfer1::DataType dtype;

    // Expert routing information (always needed)
    int32_t const* token_selected_experts; // [local_num_tokens, top_k]

    // Generic payloads
    int num_payloads;                         // Number of different payload types
    PayloadDescriptor payloads[kMaxPayloads]; // Array of payload descriptors

    // Communication workspace
    void* workspace;    // IPC workspace for communication
    int* recv_counters; // [ep_size] atomic counters for each recv rank - tracks tokens received from each rank

    cudaStream_t stream;
};

// Combine phase parameters
struct MoeA2ACombineParams
{
    // EP configuration
    int ep_size;           // Number of EP ranks
    int ep_rank;           // Current EP rank
    int local_num_experts; // Number of experts on this rank

                           // Token configuration
    int local_num_tokens;    // Number of tokens that selected this rank
    int max_tokens_per_rank; // Maximum tokens per rank for pre-allocation
    int hidden_size;         // Hidden dimension
    int top_k;               // Number of experts per token

    // Data type
    nvinfer1::DataType dtype;

    // Input tensors (processed expert outputs)
    void* expert_outputs; // [max_batch_size, hidden_size] with holes

                          // Generic payloads for combine phase
    int num_result_payloads;                         // Number of result payloads to send back
    PayloadDescriptor result_payloads[kMaxPayloads]; // Array of result payload descriptors

    // Communication workspace
    void* workspace;    // IPC workspace for communication
    int* send_counters; // [ep_size] atomic counters - tracks tokens sent to each rank

    // Reduction parameters
    bool enable_reduction; // Whether to perform reduction during combine

    cudaStream_t stream;
};

// Dispatch kernels
void moe_a2a_dispatch_op(MoeA2ADispatchParams const& params);

// Combine kernels
void moe_a2a_combine_op(MoeA2ACombineParams const& params);

// Buffer layout helper functions
inline __host__ __device__ size_t get_rank_buffer_offset(int rank_id, size_t buffer_size_per_rank)
{
    return rank_id * buffer_size_per_rank;
}

inline __host__ __device__ void* get_payload_ptr_in_recv_buffer(void* recv_buffer, int rank_id, int token_position,
    size_t max_tokens_per_rank, size_t elements_per_token, size_t element_size)
{
    char* base = static_cast<char*>(recv_buffer);
    // Buffer layout: [rank0_tokens][rank1_tokens]...[rankN_tokens]
    size_t rank_offset = rank_id * max_tokens_per_rank * elements_per_token * element_size;
    size_t token_offset = token_position * elements_per_token * element_size;
    return base + rank_offset + token_offset;
}

} // namespace tensorrt_llm::kernels::moe_a2a
