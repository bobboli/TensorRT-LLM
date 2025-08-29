# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pickle
import sys
import traceback

import cloudpickle
import torch
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import pytest

import tensorrt_llm as tllm

from tensorrt_llm._mnnvl_utils import MnnvlMemory

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


@pytest.fixture(autouse=True)
def setup_test():
    torch.manual_seed(0x1234)
    tllm.logger.set_level('error')


def compute_nvfp4_workspace_size(ep_size: int, max_tokens_per_rank: int,
                                 hidden_size: int, top_k: int) -> int:
    """Compute total workspace size per rank matching NV FP4 layout used in single-GPU test."""
    packed_hidden_size = hidden_size // 2  # 2 FP4 per uint8
    num_elts_per_sf = 16
    num_scaling_factors = hidden_size // num_elts_per_sf
    packed_tokens_size = ep_size * max_tokens_per_rank * packed_hidden_size * 1  # uint8
    sf_size = ep_size * max_tokens_per_rank * num_scaling_factors * 4  # float32
    experts_size = ep_size * max_tokens_per_rank * top_k * 4  # int32
    final_scales_size = ep_size * max_tokens_per_rank * top_k * 4  # float32
    completion_flags_size = ep_size * 4  # int32, one flag per rank
    return packed_tokens_size + sf_size + experts_size + final_scales_size + completion_flags_size


def generate_token_selected_experts(local_num_tokens: int, ep_size: int,
                                    num_experts_per_rank: int, top_k: int) -> torch.Tensor:
    """Generate global expert IDs tensor, aligned with single-GPU test semantics."""
    return torch.randint(
        0,
        ep_size * num_experts_per_rank,
        (local_num_tokens, top_k),
        dtype=torch.int32,
        device='cuda',
    )


def make_nvfp4_payloads(local_num_tokens: int, hidden_size: int, top_k: int,
                         rank: int, token_selected_experts: torch.Tensor) -> list:
    """Create the four NV FP4 payloads exactly as in single-GPU test."""
    payloads = []
    # Payload 1: Packed FP4 tokens (uint8)
    packed_hidden_size = hidden_size // 2
    packed_tokens = torch.randint(0, 256,
                                  (local_num_tokens, packed_hidden_size),
                                  dtype=torch.uint8,
                                  device='cuda')
    payloads.append(packed_tokens)

    # Payload 2: Scaling factors (float32), with rank-specific pattern
    num_elts_per_sf = 16
    num_scaling_factors = hidden_size // num_elts_per_sf
    scaling_factors = torch.randn(local_num_tokens, num_scaling_factors,
                                  dtype=torch.float32,
                                  device='cuda')
    scaling_factors += rank
    payloads.append(scaling_factors)

    # Payload 3: token_selected_experts
    payloads.append(token_selected_experts)

    # Payload 4: token_final_scales (float32)
    token_final_scales = torch.rand(local_num_tokens, top_k,
                                    dtype=torch.float32,
                                    device='cuda')
    payloads.append(token_final_scales)
    return payloads


def run_moe_a2a_dispatch_single_rank(ep_size, local_num_tokens_list, top_k, 
                                     workspace_size_per_rank, num_experts_per_rank, 
                                     hidden_size, max_tokens_per_rank):
    """Worker function for MPIPoolExecutor.
    Runs one-rank dispatch and returns tensors for verification.
    Mirrors the single-GPU test logic but runs on separate GPUs.
    """
    # Import everything inside the worker to avoid pickle issues
    import torch
    import traceback
    import tensorrt_llm as tllm
    from tensorrt_llm._mnnvl_utils import MnnvlMemory
    from tensorrt_llm.mapping import Mapping

    rank = tllm.mpi_rank()
    torch.cuda.set_device(rank)
    
    # Initialize MnnvlMemory after setting CUDA device
    MnnvlMemory.initialize()

    try:
        mapping = Mapping(
            rank=rank,
            tp_size=ep_size,
            pp_size=1,
            cp_size=1,
            world_size=ep_size,
        )
        
        # Create MNNVL workspace (multi-GPU equivalent of single-GPU workspace)
        mnnvl_mem = MnnvlMemory(mapping, workspace_size_per_rank)
        workspace = mnnvl_mem.as_torch_strided_tensor(torch.uint8)

        # Get the number of tokens for this specific rank (same as single-GPU)
        rank_local_tokens = local_num_tokens_list[rank]
        
        # Create token_selected_experts - random expert assignments (same as single-GPU)
        token_selected_experts = torch.randint(
            0,
            ep_size * num_experts_per_rank, 
            (rank_local_tokens, top_k),
            dtype=torch.int32,
            device='cuda')

        # Create 4 payloads for NV FP4 using helper (same as single-GPU)
        payloads = make_nvfp4_payloads(
            rank_local_tokens, hidden_size, top_k, rank, token_selected_experts)

        # Execute dispatch (same as single-GPU)
        # Access torch.ops inside worker to avoid pickle issues
        recv_buffers, send_counters = torch.ops.trtllm.moe_a2a_dispatch(
            token_selected_experts,
            payloads,
            workspace,
            max_tokens_per_rank,
            rank,  # ep_rank
            ep_size,
            top_k)
        
        # Synchronize CUDA before returning results
        torch.cuda.synchronize()

        # Return results to be collected (move to CPU for MPI transfer)
        return (token_selected_experts.cpu(),
                [p.cpu() for p in payloads],
                [rb.cpu() for rb in recv_buffers],
                send_counters.cpu())
    except Exception:
        traceback.print_exc()
        raise


class TestMoEAlltoAll:

    @pytest.mark.parametrize("ep_size,local_num_tokens_list,top_k", [
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
        # (1, [64], 2),              # Single rank  # TODO: fix
        (4, [1, 1, 1, 1], 2),      # Single token per rank
        (3, [100, 200, 150], 2),   # Non-power-of-2 ep_size
    ])
    def test_dispatch_single_gpu(self, ep_size, local_num_tokens_list, top_k):
        """Test MoE A2A dispatch on single GPU simulating multiple ranks"""
        torch.cuda.set_device(0)

        assert ep_size == len(local_num_tokens_list), "ep_size does not match local_num_tokens_list"

        hidden_size = 1024
        num_experts_per_rank = 8
        max_tokens_per_rank = max(local_num_tokens_list)

        # Calculate workspace size for all payloads
        workspace_size_per_rank = compute_nvfp4_workspace_size(
            ep_size, max_tokens_per_rank, hidden_size, top_k)
        
        # Create unified virtual memory workspace (simulated for single GPU)
        # In real MNNVL, this would be allocated using MnnvlMemory
        # Workspace shape: [ep_size, size_per_rank] where size_per_rank contains all payload buffers
        workspace = torch.zeros((ep_size, workspace_size_per_rank), 
                               dtype=torch.uint8, 
                               device='cuda')
        
        # Storage for all ranks' data
        all_recv_buffers = []
        all_send_counters = []
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

            # Create 4 payloads for NV FP4 using helper
            payloads = make_nvfp4_payloads(
                rank_local_tokens, hidden_size, top_k, rank, token_selected_experts)

            all_payloads.append(payloads)

        # Execute dispatch for each rank
        for rank in range(ep_size):
            recv_buffers, send_counters = torch.ops.trtllm.moe_a2a_dispatch(
                all_token_selected_experts[rank],
                all_payloads[rank],
                workspace,
                max_tokens_per_rank,
                rank,  # ep_rank
                ep_size,
                top_k)
            all_recv_buffers.append(recv_buffers)
            # Note: The second return value actually contains send counters, not recv counters
            all_send_counters.append(send_counters)

        # Synchronize CUDA to ensure all dispatch operations are complete
        torch.cuda.synchronize()

        # Verify dispatch results
        self._verify_dispatch(all_token_selected_experts,
                             all_payloads, all_recv_buffers,
                             all_send_counters, ep_size,
                             local_num_tokens_list, top_k,
                             max_tokens_per_rank)
    
    
    @pytest.mark.skipif(torch.cuda.device_count() < 2,
                        reason='needs at least 2 GPUs to run multi-GPU test')
    @pytest.mark.threadleak(enabled=False)  # MPI pool executors have known thread cleanup timing issues
    @pytest.mark.parametrize("mpi_pool_executor,local_num_tokens_list,top_k", [
        # (num_workers, local_num_tokens_list, top_k)
        # Basic configurations
        (4, [32, 32, 32, 32], 2),  # Four ranks with uniform distribution
        (4, [16, 32, 64, 48], 2),  # Four ranks with non-uniform distribution
        (2, [100, 50], 2),         # Two ranks with different loads
        (8, [10, 20, 30, 40, 50, 60, 70, 80], 2),  # Eight ranks with increasing load
        
        # Different top_k values
        (4, [32, 32, 32, 32], 4),  # Four ranks with top_k = 4
        (4, [32, 32, 32, 32], 8),  # Four ranks with top_k = 8
        
        # Edge cases
        # (1, [64], 2),            # Single rank  # TODO: fix (needs mpi_pool_executor to support 1 worker)
        (4, [1, 1, 1, 1], 2),      # Four ranks with single token per rank
        # (3, [100, 200, 150], 2), # Three ranks with non-power-of-2 ep_size (needs mpi_pool_executor to support 3 workers)
    ], indirect=["mpi_pool_executor"])
    def test_dispatch_multi_gpu(self, mpi_pool_executor, local_num_tokens_list, top_k):
        """Test MoE A2A dispatch with MNNVL across multiple GPUs"""

        try:
            MnnvlMemory.initialize()
            assert MnnvlMemory.supports_mnnvl()
        except Exception:
            pytest.skip("MNNVL not supported on this system")

        ep_size = mpi_pool_executor.num_workers
        assert ep_size == len(local_num_tokens_list), "ep_size does not match local_num_tokens_list"
        
        # Skip if not enough GPUs
        if torch.cuda.device_count() < ep_size:
            pytest.skip(f"Need at least {ep_size} GPUs, found {torch.cuda.device_count()}")
        
        hidden_size = 1024
        num_experts_per_rank = 8
        max_tokens_per_rank = max(local_num_tokens_list)
        
        # Calculate workspace size for all payloads (same as single-GPU)
        workspace_size_per_rank = compute_nvfp4_workspace_size(
            ep_size, max_tokens_per_rank, hidden_size, top_k)
        
        # Run dispatch on workers - each worker executes the same logic as single-GPU
        # but on separate GPUs with MNNVL memory instead of regular CUDA memory
        results = mpi_pool_executor.map(
            run_moe_a2a_dispatch_single_rank,
            *zip(*[(ep_size, local_num_tokens_list, top_k, workspace_size_per_rank,
                    num_experts_per_rank, hidden_size, max_tokens_per_rank)] * ep_size),
        )
        
        # Collect results from all ranks (same as single-GPU collecting from simulated ranks)
        all_results = list(results)
        
        # Extract results in same format as single-GPU test
        all_token_selected_experts = [r[0] for r in all_results]
        all_payloads = [r[1] for r in all_results]
        all_recv_buffers = [r[2] for r in all_results]
        all_send_counters = [r[3] for r in all_results]
        
        # Verify dispatch results using same verification logic as single-GPU
        self._verify_dispatch(all_token_selected_experts,
                             all_payloads, all_recv_buffers,
                             all_send_counters, ep_size,
                             local_num_tokens_list, top_k,
                             max_tokens_per_rank)
    


    def test_combine_single_gpu(self):
        """Test MoE A2A combine on single GPU simulating multiple ranks"""
        # TODO: Implement combine test once combine kernel is implemented
        pytest.skip("Combine functionality not yet implemented")

    def test_combine_multi_gpu(self):
        """Test MoE A2A combine across multiple GPUs (placeholder for distributed test)"""
        # TODO: Implement multi-GPU combine test when combine kernel and distributed testing are available
        pytest.skip("Multi-GPU combine test not yet implemented")
    
    def _verify_dispatch(self, all_token_selected_experts,
                        all_payloads, all_recv_buffers,
                        all_send_counters, ep_size,
                        local_num_tokens_list, top_k,
                        max_tokens_per_rank):
        """Verify dispatch results for any payload configuration"""
        
        # Synchronize CUDA to ensure all operations are complete
        torch.cuda.synchronize()
        
        # For each sending rank, verify its send counters
        for send_rank in range(ep_size):
            send_counters = all_send_counters[send_rank]  # Actually contains send counters
            
            # Count expected sends to each target
            expected_sends = {}
            
            token_experts = all_token_selected_experts[send_rank]
            sent_to_rank = set()  # Track which ranks we've sent to for each token
            
            for token_idx in range(token_experts.shape[0]):
                experts = token_experts[token_idx]
                target_ranks = experts % ep_size
                sent_to_rank.clear()
                
                # Due to deduplication, each token is sent to each unique target rank only once
                for target_rank in target_ranks.tolist():
                    if target_rank not in sent_to_rank:
                        if target_rank not in expected_sends:
                            expected_sends[target_rank] = 0
                        expected_sends[target_rank] += 1
                        sent_to_rank.add(target_rank)
            
            # Verify send counters for each target rank
            for target_rank in range(ep_size):
                expected_to_rank = expected_sends.get(target_rank, 0)
                actual_to_rank = send_counters[target_rank].item()
                assert actual_to_rank == expected_to_rank, \
                    f"Rank {send_rank} sent {actual_to_rank} tokens to rank {target_rank}, " \
                    f"expected {expected_to_rank}"
            
        
        # Verify buffer sizes and types for each rank
        for recv_rank in range(ep_size):
            recv_buffers_for_rank = all_recv_buffers[recv_rank]
            num_payloads = len(all_payloads[0])  # Get number of payloads dynamically
            
            assert len(recv_buffers_for_rank) == num_payloads, \
                f"Rank {recv_rank} should have {num_payloads} receive buffers"
            
            for payload_idx in range(num_payloads):
                recv_buffer = recv_buffers_for_rank[payload_idx]
                
                # Get the corresponding source payload to check dtype and shape
                source_payload = all_payloads[0][payload_idx]  # Use first rank's payload as reference
                payload_shape = source_payload.shape
                elements_per_token = payload_shape[1] if len(payload_shape) > 1 else 1
                
                # Each buffer contains space for ALL ranks
                expected_buffer_size = ep_size * max_tokens_per_rank * elements_per_token
                
                assert recv_buffer.numel() == expected_buffer_size, \
                    f"Rank {recv_rank} payload {payload_idx} buffer size mismatch"
                
                assert recv_buffer.dtype == source_payload.dtype, \
                    f"Rank {recv_rank} payload {payload_idx} dtype mismatch: " \
                    f"expected {source_payload.dtype}, got {recv_buffer.dtype}"
