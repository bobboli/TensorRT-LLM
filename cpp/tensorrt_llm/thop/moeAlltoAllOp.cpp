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

#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

namespace torch_ext
{

namespace
{

// MoE All-to-All Dispatch Operation
// This operation dispatches tokens and their associated payloads to different expert ranks.
//
// Inputs:
//   - tokenSelectedExperts: [local_num_tokens, top_k] tensor of expert indices
//   - inputPayloads: List of tensors with shape [local_num_tokens, ...] containing data to dispatch
//   - workspace: [ep_size, size_per_rank] unified virtual memory workspace where
//                size_per_rank = sum(ep_size * max_tokens_per_rank * elements_per_token * element_size) for all payloads
//   - maxTokensPerRank: Maximum number of tokens that can be received per rank
//   - epRank: Current expert parallel rank
//   - epSize: Total expert parallel size
//   - topK: Number of experts selected per token
//
// Returns:
//   - recvBuffers: List of receive buffers, one for each payload
//   - recvCounters: [ep_size] tensor tracking tokens received from each rank
//
// Note: token_selected_experts is used for routing but is NOT automatically included as a payload.
//       If you want to dispatch token_selected_experts, include it explicitly in inputPayloads.
std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor> moeA2ADispatchOp(torch::Tensor const& tokenSelectedExperts,
    std::vector<torch::Tensor> const& inputPayloads, torch::Tensor const& workspace, int64_t maxTokensPerRank, int64_t epRank, int64_t epSize,
    int64_t topK)
{
    using tensorrt_llm::kernels::moe_a2a::PayloadDescriptor;
    using tensorrt_llm::kernels::moe_a2a::MoeA2ADispatchParams;
    using tensorrt_llm::kernels::moe_a2a::moe_a2a_dispatch_launch;
    using tensorrt_llm::kernels::moe_a2a::kMaxTopK;
    using tensorrt_llm::kernels::moe_a2a::kMaxPayloads;

    // Validate inputs
    CHECK_INPUT(tokenSelectedExperts, torch::kInt32);
    TORCH_CHECK(tokenSelectedExperts.dim() == 2, "tokenSelectedExperts must be a 2D tensor");
    TORCH_CHECK(tokenSelectedExperts.size(1) == topK, "tokenSelectedExperts must have topK columns");

    int64_t localNumTokens = tokenSelectedExperts.size(0);
    TORCH_CHECK(localNumTokens > 0, "localNumTokens must be positive");
    TORCH_CHECK(maxTokensPerRank > 0, "maxTokensPerRank must be positive");
    TORCH_CHECK(epRank >= 0 && epRank < epSize, "epRank must be in the range [0, epSize)");
    TORCH_CHECK(topK > 0 && topK <= kMaxTopK, "topK must be in the range (0, kMaxTopK]");
    TORCH_CHECK(!inputPayloads.empty(), "inputPayloads must not be empty");
    TORCH_CHECK(inputPayloads.size() <= kMaxPayloads, "Too many input payloads");

    // All input payloads must have the same first dimension (localNumTokens)
    for (auto const& payload : inputPayloads)
    {
        TORCH_CHECK(payload.dim() >= 1, "All payloads must have at least 1 dimension");
        TORCH_CHECK(payload.size(0) == localNumTokens,
            "All payloads must have the same first dimension as tokenSelectedExperts");
        TORCH_CHECK(payload.is_contiguous(), "All payloads must be contiguous");
    }

    // Validate workspace - unified virtual memory [epSize, sizePerRank]
    CHECK_TH_CUDA(workspace);
    CHECK_TYPE(workspace, torch::kUInt8);
    // Don't check contiguous - MnnvlMemory creates strided tensors for multi-GPU
    TORCH_CHECK(workspace.dim() == 2, "workspace must be a 2D tensor of shape [epSize, sizePerRank]");
    TORCH_CHECK(workspace.size(0) == epSize, "workspace first dimension must equal epSize");
    
    // Calculate buffer sizes for all payloads
    // Each payload buffer needs space for data from ALL ranks: epSize * maxTokensPerRank * elementsPerToken
    int64_t totalBytesNeeded = 0;
    std::vector<int64_t> payloadByteSizes;
    std::vector<int> payloadElementSizes;
    std::vector<int> payloadElementsPerToken;
    
    for (auto const& payload : inputPayloads)
    {
        CHECK_CONTIGUOUS(payload);
        CHECK_TH_CUDA(payload);
        TORCH_CHECK(payload.dim() == 2, "payload must be a 2D tensor");
        TORCH_CHECK(payload.size(0) == localNumTokens, "payload must have the same first dimension as tokenSelectedExperts");

        int elementsPerToken = static_cast<int>(payload.size(1));
        int elementSize = static_cast<int>(payload.dtype().itemsize());
        // Each payload buffer stores data from ALL ranks
        int64_t bytesPerPayload = epSize * maxTokensPerRank * elementsPerToken * elementSize;
        
        payloadByteSizes.push_back(bytesPerPayload);
        payloadElementSizes.push_back(elementSize);
        payloadElementsPerToken.push_back(elementsPerToken);
        totalBytesNeeded += bytesPerPayload;
    }
    
    // Validate workspace size - must include space for completion_flags
    int64_t completionFlagsBytes = epSize * sizeof(int);  // Each rank has epSize flags
    int64_t totalBytesWithFlags = totalBytesNeeded + completionFlagsBytes;
    int64_t sizePerRank = workspace.size(1);  // size in bytes since dtype is uint8
    TORCH_CHECK(sizePerRank >= totalBytesWithFlags, 
        "Workspace size per rank insufficient for receive buffers and completion_flags. "
        "Need ", totalBytesWithFlags, " bytes (", totalBytesNeeded, " for payloads + ", 
        completionFlagsBytes, " for completion_flags), but got ", sizePerRank);
    
    // Setup receive buffer pointers from unified workspace
    std::vector<PayloadDescriptor> payloadDescriptors;
    
    // Get base workspace pointer
    auto* workspace_ptr = workspace.data_ptr<uint8_t>();
    
    // Setup payload descriptors for source data
    for (int i = 0; i < static_cast<int>(inputPayloads.size()); i++)
    {
        PayloadDescriptor desc{};
        desc.src_data = inputPayloads[i].data_ptr();
        desc.element_size = payloadElementSizes[i];
        desc.elements_per_token = payloadElementsPerToken[i];
        payloadDescriptors.push_back(desc);
    }

    // TODO: Use torch empty to replace initilization
    // Create send_counters tensor - tracks number of tokens sent to each target rank
    torch::Tensor sendCounters = torch::zeros({epSize}, tokenSelectedExperts.options().dtype(torch::kInt32));
    // Create local_token_counter - tracks completed tokens on this rank
    torch::Tensor localTokenCounter = torch::zeros({1}, tokenSelectedExperts.options().dtype(torch::kInt32));
    // Create local indices tensor for content verification - tracks destination positions for each source token
    torch::Tensor sendIndices = torch::full({localNumTokens, epSize}, -1, 
        tokenSelectedExperts.options().dtype(torch::kInt32));

    // Setup dispatch parameters
    MoeA2ADispatchParams params{};
    params.token_selected_experts = tokenSelectedExperts.data_ptr<int32_t>();
    params.num_payloads = static_cast<int32_t>(payloadDescriptors.size());
    std::copy(payloadDescriptors.begin(), payloadDescriptors.end(), &params.payloads[0]);
    // Calculate and store recv buffer pointers directly in params
    for (int target_rank = 0; target_rank < epSize; target_rank++)
    {
        // Each rank gets workspace[target_rank] - calculate base pointer
        auto* target_workspace = workspace_ptr + (target_rank * workspace.stride(0));
        int64_t offset = 0;
        
        // Debug: print workspace addresses
        if (epRank == 0) {
            printf("Rank %d: target_rank %d workspace at %p (base %p + stride %ld * %d)\n",
                   static_cast<int>(epRank), target_rank, target_workspace, workspace_ptr, 
                   workspace.stride(0), target_rank);
        }
        
        for (int payload_idx = 0; payload_idx < static_cast<int>(inputPayloads.size()); payload_idx++)
        {
            // Store buffer pointer for kernel
            params.recv_buffers[target_rank][payload_idx] = target_workspace + offset;
            
            // Update offset for next payload
            offset += payloadByteSizes[payload_idx];
        }
        
        // Set completion_flags pointer for this rank (after all payloads)
        params.completion_flags[target_rank] = reinterpret_cast<int*>(target_workspace + offset);
    }
    params.max_tokens_per_rank = static_cast<int>(maxTokensPerRank);
    params.send_counters = sendCounters.data_ptr<int>();
    params.local_token_counter = localTokenCounter.data_ptr<int>();
    params.send_indices = sendIndices.data_ptr<int>();
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.ep_size = static_cast<int>(epSize);
    params.ep_rank = static_cast<int>(epRank);
    params.top_k = static_cast<int>(topK);
    params.stream = at::cuda::getCurrentCUDAStream();

    // Launch the dispatch kernel
    moe_a2a_dispatch_launch(params);
    cudaError_t result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess, "moe_a2a_dispatch kernel launch failed: ", cudaGetErrorString(result));

    // Create tensor views for the current rank's receive buffers only
    std::vector<torch::Tensor> recvBuffers;
    auto* current_rank_workspace = workspace_ptr + (epRank * workspace.stride(0));
    int64_t offset = 0;
    
    for (int payload_idx = 0; payload_idx < static_cast<int>(inputPayloads.size()); payload_idx++)
    {
        auto const& payload = inputPayloads[payload_idx];
        
        // Create tensor view for this payload - contains data from ALL ranks
        auto recvBuffer = torch::from_blob(
            current_rank_workspace + offset,
            {epSize, maxTokensPerRank, payloadElementsPerToken[payload_idx]},
            payload.options()
        );
        recvBuffers.push_back(recvBuffer);
        
        // Update offset for next payload
        offset += payloadByteSizes[payload_idx];
    }

    return std::make_tuple(std::move(recvBuffers), sendCounters, sendIndices);
}

// TODO(trtllm-team): Implement moeA2ACombineOp for the combine phase (pull back payload with reduction)

} // anonymous namespace

} // namespace torch_ext

// PyTorch bindings
TORCH_LIBRARY_FRAGMENT(trtllm, module)
{
    module.def(
        "moe_a2a_dispatch(Tensor token_selected_experts, Tensor[] input_payloads, Tensor workspace, int max_tokens_per_rank, "
        "int ep_rank, int ep_size, int top_k) -> (Tensor[], Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, module)
{
    module.impl("moe_a2a_dispatch", &torch_ext::moeA2ADispatchOp);
}
