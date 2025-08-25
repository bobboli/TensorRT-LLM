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
std::tuple<std::vector<torch::Tensor>, torch::Tensor> moeA2ADispatchOp(torch::Tensor const& tokenSelectedExperts,
    std::vector<torch::Tensor> const& inputPayloads, int64_t maxTokensPerRank, int64_t epRank, int64_t epSize,
    int64_t topK)
{
    using tensorrt_llm::kernels::moe_a2a::PayloadDescriptor;
    using tensorrt_llm::kernels::moe_a2a::MoeA2ADispatchParams;
    using tensorrt_llm::kernels::moe_a2a::moe_a2a_dispatch_op;
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

    // Create receive buffers for each payload
    std::vector<torch::Tensor> recvBuffers;
    std::vector<PayloadDescriptor> payloadDescriptors;

    // Process input payloads
    for (auto const& payload : inputPayloads)
    {
        // Calculate elements per token based on tensor shape
        int64_t numel = payload.numel();
        TORCH_CHECK(numel % localNumTokens == 0, "Payload tensor size must be divisible by localNumTokens");
        size_t elementsPerToken = numel / localNumTokens;
        size_t elementSize = payload.dtype().itemsize();

        // Calculate number of elements for receive buffer
        auto numElements = static_cast<int64_t>(epSize * maxTokensPerRank * elementsPerToken);
        torch::Tensor recvBuffer = torch::empty({numElements}, payload.options());
        recvBuffers.push_back(recvBuffer);

        PayloadDescriptor desc{};
        desc.src_data = payload.data_ptr();
        desc.recv_buffer = recvBuffer.data_ptr();
        desc.element_size = elementSize;
        desc.elements_per_token = elementsPerToken;
        payloadDescriptors.push_back(desc);
    }

    // Create recv_counters tensor
    torch::Tensor recvCounters = torch::zeros({epSize}, tokenSelectedExperts.options().dtype(torch::kInt32));

    // Setup dispatch parameters
    MoeA2ADispatchParams params{};
    params.token_selected_experts = tokenSelectedExperts.data_ptr<int32_t>();
    params.num_payloads = static_cast<int>(payloadDescriptors.size());
    std::copy(payloadDescriptors.begin(), payloadDescriptors.end(), &params.payloads[0]);
    params.max_tokens_per_rank = static_cast<int>(maxTokensPerRank);
    params.recv_counters = recvCounters.data_ptr<int>();
    params.local_num_tokens = static_cast<int>(localNumTokens);
    params.ep_size = static_cast<int>(epSize);
    params.ep_rank = static_cast<int>(epRank);
    params.top_k = static_cast<int>(topK);
    params.stream = at::cuda::getCurrentCUDAStream();

    // Launch the dispatch kernel
    moe_a2a_dispatch_op(params);

    return std::make_tuple(recvBuffers, recvCounters);
}

// TODO(trtllm-team): Implement moeA2ACombineOp for the combine phase (pull back payload with reduction)

} // anonymous namespace

} // namespace torch_ext

// PyTorch bindings
TORCH_LIBRARY_FRAGMENT(trtllm, module)
{
    module.def(
        "moe_a2a_dispatch(Tensor token_selected_experts, Tensor[] input_payloads, int max_tokens_per_rank, "
        "int ep_rank, int ep_size, int top_k) -> (Tensor[], Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, module)
{
    module.impl("moe_a2a_dispatch", &torch_ext::moeA2ADispatchOp);
}
