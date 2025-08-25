# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

import tensorrt_llm as tllm


class TestMoEAlltoAll(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0x1234)
        tllm.logger.set_level('error')

    def test_dispatch_single_gpu(self):
        """Test MoE A2A dispatch on single GPU simulating multiple ranks"""
        torch.cuda.set_device(0)

        # Test configuration
        ep_size = 4
        local_num_tokens = 32
        hidden_size = 256
        top_k = 2
        max_tokens_per_rank = 64
        dtype = torch.float16
        num_experts_per_rank = 8

        # Storage for all ranks' data
        all_recv_buffers = []
        all_recv_counters = []
        all_token_selected_experts = []
        all_payloads = []

        # Generate data for each simulated rank
        for rank in range(ep_size):
            # Create token_selected_experts - random expert assignments
            token_selected_experts = torch.randint(
                0,
                ep_size * num_experts_per_rank, (local_num_tokens, top_k),
                dtype=torch.int32,
                device='cuda')
            all_token_selected_experts.append(token_selected_experts)

            # Create multiple payloads
            payloads = []

            # Payload 1: tokens
            tokens = torch.randn(local_num_tokens,
                                 hidden_size,
                                 dtype=dtype,
                                 device='cuda')
            payloads.append(tokens)

            # Payload 2: weights/scales
            weights = torch.rand(local_num_tokens,
                                 top_k,
                                 dtype=dtype,
                                 device='cuda')
            payloads.append(weights)

            # Payload 3: auxiliary data (e.g., scaling factors)
            aux_data = torch.randn(local_num_tokens,
                                   64,
                                   dtype=dtype,
                                   device='cuda')
            payloads.append(aux_data)

            all_payloads.append(payloads)

        # Execute dispatch for each rank
        for rank in range(ep_size):
            recv_buffers, recv_counters = torch.ops.trtllm.moe_a2a_dispatch(
                all_token_selected_experts[rank],
                all_payloads[rank],
                max_tokens_per_rank,
                rank,  # ep_rank
                ep_size,
                top_k)
            all_recv_buffers.append(recv_buffers)
            all_recv_counters.append(recv_counters)

        # Verify dispatch results
        self._verify_single_gpu_dispatch(all_token_selected_experts,
                                         all_payloads, all_recv_buffers,
                                         all_recv_counters, ep_size,
                                         local_num_tokens, top_k,
                                         max_tokens_per_rank)

    def test_dispatch_multi_gpu(self):
        """Test MoE A2A dispatch across multiple GPUs (placeholder for distributed test)"""
        # TODO: Implement multi-GPU test when distributed testing infrastructure is available
        # This will require MPI/NCCL setup and multiple physical GPUs
        self.skipTest("Multi-GPU dispatch test not yet implemented")

    def test_combine_single_gpu(self):
        """Test MoE A2A combine on single GPU simulating multiple ranks"""
        # TODO: Implement combine test once combine kernel is implemented
        self.skipTest("Combine functionality not yet implemented")

    def test_combine_multi_gpu(self):
        """Test MoE A2A combine across multiple GPUs (placeholder for distributed test)"""
        # TODO: Implement multi-GPU combine test when combine kernel and distributed testing are available
        self.skipTest("Multi-GPU combine test not yet implemented")

    def _verify_single_gpu_dispatch(self, all_token_selected_experts,
                                    all_payloads, all_recv_buffers,
                                    all_recv_counters, ep_size,
                                    local_num_tokens, top_k,
                                    max_tokens_per_rank):
        """Verify dispatch results for single GPU test"""

        # For each receiving rank, verify counters
        for recv_rank in range(ep_size):
            recv_counters = all_recv_counters[recv_rank]

            # Count how many unique tokens this rank should receive
            expected_token_count = 0
            token_sources = {}  # Track which tokens come from which ranks

            for send_rank in range(ep_size):
                token_experts = all_token_selected_experts[send_rank]

                # Check each token to see if it should go to recv_rank
                for token_idx in range(local_num_tokens):
                    experts = token_experts[token_idx]
                    target_ranks = experts % ep_size

                    # If any expert for this token maps to recv_rank
                    if recv_rank in target_ranks.tolist():
                        expected_token_count += 1
                        if send_rank not in token_sources:
                            token_sources[send_rank] = 0
                        token_sources[send_rank] += 1

            # Verify total received matches expected
            total_received = recv_counters.sum().item()
            self.assertEqual(
                total_received, expected_token_count,
                f"Rank {recv_rank} received {total_received} tokens, "
                f"expected {expected_token_count}")

            # Verify per-rank counters
            for send_rank in range(ep_size):
                expected_from_rank = token_sources.get(send_rank, 0)
                actual_from_rank = recv_counters[send_rank].item()
                self.assertEqual(
                    actual_from_rank, expected_from_rank,
                    f"Rank {recv_rank} received {actual_from_rank} tokens "
                    f"from rank {send_rank}, expected {expected_from_rank}")

            # Verify buffer sizes for each payload
            for payload_idx, recv_buffer in enumerate(
                    all_recv_buffers[recv_rank]):
                payload_shape = all_payloads[0][payload_idx].shape
                elements_per_token = payload_shape[1] if len(
                    payload_shape) > 1 else 1
                expected_buffer_size = ep_size * max_tokens_per_rank * elements_per_token

                self.assertEqual(
                    recv_buffer.numel(), expected_buffer_size,
                    f"Rank {recv_rank} payload {payload_idx} buffer size mismatch"
                )


if __name__ == "__main__":
    unittest.main()
