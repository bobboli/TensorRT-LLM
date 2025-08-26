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
from parameterized import parameterized

import tensorrt_llm as tllm


class TestMoEAlltoAll(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0x1234)
        tllm.logger.set_level('error')

    @parameterized.expand([
        # (ep_size, local_num_tokens_list, top_k)
        # Basic configurations
        (4, [32, 32, 32, 32], 2),  # Uniform distribution
        (4, [16, 32, 64, 48], 2),  # Non-uniform distribution
        (2, [100, 50], 2),         # Two ranks with different loads
        (8, [10, 20, 30, 40, 50, 60, 70, 80], 2),  # Eight ranks with increasing load
        
        # Different top_k values
        (4, [32, 32, 32, 32], 4),  # top_k = 4
        (4, [32, 32, 32, 32], 8),  # top_k = 8
        
        # Edge cases
        (1, [64], 2),              # Single rank
        (4, [1, 1, 1, 1], 2),      # Single token per rank
        (3, [100, 200, 150], 2),   # Non-power-of-2 ep_size
    ])
    def test_dispatch_single_gpu(self, ep_size, local_num_tokens_list, top_k):
        """Test MoE A2A dispatch on single GPU simulating multiple ranks"""
        torch.cuda.set_device(0)

        # Fixed configuration for NV FP4
        hidden_size = 1024  # Results in 512 packed uint8 elements
        num_experts_per_rank = 8
        
        # For FP4, we pack 2 values per uint8
        packed_hidden_size = hidden_size // 2  # 512 uint8 elements
        
        # NV FP4 uses 16 elements per scaling factor by default
        num_elts_per_sf = 16
        num_scaling_factors = hidden_size // num_elts_per_sf  # 64 scaling factors
        
        # Calculate max_tokens_per_rank as the maximum across all ranks
        max_tokens_per_rank = max(local_num_tokens_list)

        # Calculate workspace size for all payloads
        # Each buffer needs space for ALL ranks
        # Payload 1: packed FP4 tokens (uint8)
        packed_tokens_size = ep_size * max_tokens_per_rank * packed_hidden_size * 1  # uint8
        # Payload 2: scaling factors (float32)
        sf_size = ep_size * max_tokens_per_rank * num_scaling_factors * 4  # float32
        # Payload 3: token_selected_experts (int32)
        experts_size = ep_size * max_tokens_per_rank * top_k * 4  # int32
        # Payload 4: token_final_scales (float32)
        final_scales_size = ep_size * max_tokens_per_rank * top_k * 4  # float32
        
        workspace_size_per_rank = packed_tokens_size + sf_size + experts_size + final_scales_size
        
        # Create unified virtual memory workspace (simulated for single GPU)
        # In real MNNVL, this would be allocated using MnnvlMemory
        # Workspace shape: [ep_size, size_per_rank] where size_per_rank contains all payload buffers
        workspace = torch.zeros((ep_size, workspace_size_per_rank), 
                               dtype=torch.uint8, 
                               device='cuda')
        
        # Storage for all ranks' data
        all_recv_buffers = []
        all_recv_counters = []
        all_token_selected_experts = []
        all_payloads = []

        # Generate data for each simulated rank
        for rank in range(ep_size):
            # Get the number of tokens for this specific rank
            rank_local_tokens = local_num_tokens_list[rank]
            
            # Create token_selected_experts - random expert assignments
            token_selected_experts = torch.randint(
                0,
                ep_size * num_experts_per_rank, (rank_local_tokens, top_k),
                dtype=torch.int32,
                device='cuda')
            all_token_selected_experts.append(token_selected_experts)

            # Create 4 payloads for NV FP4
            payloads = []
            
            # Payload 1: Packed FP4 tokens (uint8)
            packed_tokens = torch.randint(0, 256, 
                                        (rank_local_tokens, packed_hidden_size),
                                        dtype=torch.uint8,
                                        device='cuda')
            payloads.append(packed_tokens)
            
            # Payload 2: Scaling factors for the tokens (float32)
            scaling_factors = torch.randn(rank_local_tokens, num_scaling_factors,
                                        dtype=torch.float32,
                                        device='cuda')
            # Add rank-specific pattern for verification
            scaling_factors += rank
            payloads.append(scaling_factors)
            
            # Payload 3: token_selected_experts (already created, add to payloads)
            payloads.append(token_selected_experts)
            
            # Payload 4: token_final_scales (float32)
            token_final_scales = torch.rand(rank_local_tokens, top_k,
                                          dtype=torch.float32,
                                          device='cuda')
            payloads.append(token_final_scales)

            all_payloads.append(payloads)

        # Execute dispatch for each rank
        for rank in range(ep_size):
            recv_buffers, recv_counters = torch.ops.trtllm.moe_a2a_dispatch(
                all_token_selected_experts[rank],
                all_payloads[rank],
                workspace,
                max_tokens_per_rank,
                rank,  # ep_rank
                ep_size,
                top_k)
            all_recv_buffers.append(recv_buffers)
            all_recv_counters.append(recv_counters)

        # Verify dispatch results
        self._verify_dispatch(all_token_selected_experts,
                             all_payloads, all_recv_buffers,
                             all_recv_counters, ep_size,
                             local_num_tokens_list, top_k,
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
    
    def _verify_dispatch(self, all_token_selected_experts,
                        all_payloads, all_recv_buffers,
                        all_recv_counters, ep_size,
                        local_num_tokens_list, top_k,
                        max_tokens_per_rank):
        """Verify dispatch results for any payload configuration"""
        
        # For each receiving rank, verify counters and buffer contents
        for recv_rank in range(ep_size):
            recv_counters = all_recv_counters[recv_rank]
            
            # Count expected tokens
            expected_token_count = 0
            token_sources = {}
            
            for send_rank in range(ep_size):
                token_experts = all_token_selected_experts[send_rank]
                for token_idx in range(token_experts.shape[0]):
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
            
            # Verify buffer sizes and types for each payload
            # Each rank only gets its own receive buffers now
            recv_buffers_for_rank = all_recv_buffers[recv_rank]
            num_payloads = len(all_payloads[0])  # Get number of payloads dynamically
            
            self.assertEqual(len(recv_buffers_for_rank), num_payloads,
                            f"Rank {recv_rank} should have {num_payloads} receive buffers")
            
            for payload_idx in range(num_payloads):
                recv_buffer = recv_buffers_for_rank[payload_idx]
                
                # Get the corresponding source payload to check dtype and shape
                source_payload = all_payloads[0][payload_idx]  # Use first rank's payload as reference
                payload_shape = source_payload.shape
                elements_per_token = payload_shape[1] if len(payload_shape) > 1 else 1
                
                # Each buffer contains space for ALL ranks
                expected_buffer_size = ep_size * max_tokens_per_rank * elements_per_token
                
                self.assertEqual(
                    recv_buffer.numel(), expected_buffer_size,
                    f"Rank {recv_rank} payload {payload_idx} buffer size mismatch")
                
                self.assertEqual(
                    recv_buffer.dtype, source_payload.dtype,
                    f"Rank {recv_rank} payload {payload_idx} dtype mismatch: "
                    f"expected {source_payload.dtype}, got {recv_buffer.dtype}")


if __name__ == "__main__":
    unittest.main()
