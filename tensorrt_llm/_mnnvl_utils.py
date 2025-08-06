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
import array
import ctypes
import os
import platform
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import pynvml
import torch
from cuda import cuda

from ._dlpack_utils import pack_strided_memory
from ._utils import mpi_comm
from .logger import logger
from .mapping import Mapping


def _check_cu_result(cu_func_ret):
    if isinstance(cu_func_ret, tuple):
        cu_result, *others = cu_func_ret
        if cu_result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_result)
        if len(others) == 1:
            return others[0]
        elif len(others) > 1:
            return tuple(others)
        else:
            return None
    else:
        if cu_func_ret != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(cu_func_ret)
        return None


class FDExchangeServer(threading.Thread):
    """Unix Domain Socket server for file descriptor exchange."""

    def __init__(self, rank, socket_dir="/tmp"):
        super().__init__(daemon=True)
        self.rank = rank
        self.pid = os.getpid()
        self.socket_path = f"{socket_dir}/tensorrt-llm-{self.pid}.sock"
        self.server_socket = None
        self.running = False
        self.fd_map = {}  # Maps requester_rank -> fd to send

    def register_fd(self, requester_rank, fd):
        """Register a file descriptor to be sent to a specific rank."""
        self.fd_map[requester_rank] = fd

    def run(self):
        """Run the Unix Domain Socket server."""
        # Remove existing socket file if it exists
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        # Create Unix Domain Socket
        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(5)
        self.server_socket.settimeout(1.0)  # 1 second timeout for accept

        self.running = True

        while self.running:
            try:
                client_socket, _ = self.server_socket.accept()
                self._handle_client(client_socket)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.warning(f"FD exchange server error: {e}")

    def _handle_client(self, client_socket):
        """Handle a client connection requesting a file descriptor."""
        try:
            # Receive request (expecting rank number)
            data = client_socket.recv(4)
            if len(data) < 4:
                return

            requester_rank = struct.unpack("i", data)[0]

            # Get the file descriptor to send
            fd_to_send = self.fd_map.get(requester_rank)
            if fd_to_send is None:
                client_socket.send(b"NO_FD")
                return

            # Send the file descriptor using SCM_RIGHTS
            fds = array.array("i", [fd_to_send])
            ancdata = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)]
            client_socket.sendmsg([b"FD_OK"], ancdata)

        finally:
            client_socket.close()

    def stop(self):
        """Stop the server and cleanup."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        # Clean up socket file
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)


def exchange_fd_with_peer(peer_pid, local_fd, my_rank, max_retries=10):
    """
    Exchange file descriptors with a peer process using Unix Domain Sockets.

    Args:
        peer_pid: PID of the peer process
        local_fd: Local file descriptor to send
        my_rank: My rank in the communication group
        max_retries: Maximum connection attempts

    Returns:
        Remote file descriptor received from peer
    """
    socket_path = f"/tmp/tensorrt-llm-{peer_pid}.sock"

    # Retry connection with exponential backoff
    for attempt in range(max_retries):
        try:
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client_socket.connect(socket_path)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if attempt < max_retries - 1:
                time.sleep(0.1 * (2**attempt))  # Exponential backoff
            else:
                raise RuntimeError(
                    f"Failed to connect to peer {peer_pid} after {max_retries} attempts"
                )

    try:
        # Send our rank
        client_socket.send(struct.pack("i", my_rank))

        # Receive response with file descriptor
        data, ancdata, flags, addr = client_socket.recvmsg(1024, socket.CMSG_SPACE(4))

        if data == b"NO_FD":
            raise RuntimeError(f"Peer {peer_pid} has no FD for rank {my_rank}")
        elif data != b"FD_OK":
            raise RuntimeError(f"Unexpected response from peer {peer_pid}: {data}")

        # Extract file descriptor from ancillary data
        for cmsg_level, cmsg_type, cmsg_data in ancdata:
            if cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
                fd_array = array.array("i")
                fd_array.frombytes(cmsg_data)
                return fd_array[0]

        raise RuntimeError("No file descriptor received in ancillary data")

    finally:
        client_socket.close()


class MnnvlMemory:
    initialized: bool = False

    current_mem_offset: int = 0
    current_rank_stride: int = 0  # stride for ranks and also address space size.
    current_start_address: int = 0

    # allocation granularity
    allocation_granularity: int = 0

    # fabric address page size (512 MB)
    fabric_page_size: int = 1 << 29

    # MPI communicator
    comm = None

    dev_id: int = None

    allocated_map = {}
    address_refcnt = {}

    def __init__(self, mapping: Mapping, size: int):
        self.mapping = mapping
        self.segment_size = size
        self.ptr, self.rank_stride = MnnvlMemory.open_mnnvl_memory(self.mapping, size)

    def __del__(self):
        if not sys.is_finalizing():
            if hasattr(self, "ptr"):
                MnnvlMemory.close_mnnvl_memory(self.ptr)

    def as_torch_strided_tensor(self, dtype):
        num_segments = MnnvlMemory.comm.Get_size()
        return pack_strided_memory(
            self.ptr, self.segment_size, self.rank_stride, num_segments, dtype, MnnvlMemory.dev_id
        )

    @staticmethod
    def initialize():
        if not MnnvlMemory.initialized:
            # use a dummy torch CUDA tensor to trigger CUDA context initialization
            _ = torch.empty(1, device="cuda")
            # ensure nvml is initialized.
            try:
                pynvml.nvmlDeviceGetCount()
            except pynvml.NVMLError_Uninitialized:
                pynvml.nvmlInit()
            MnnvlMemory.initialized = True

    @staticmethod
    def get_comm(mapping: Mapping):
        if MnnvlMemory.comm is not None:
            return MnnvlMemory.comm
        comm = mpi_comm().Split(
            mapping.pp_rank * mapping.cp_size + mapping.cp_rank, mapping.tp_rank
        )
        MnnvlMemory.comm = comm
        return comm

    @staticmethod
    def get_allocation_prop(dev_id: int):
        location = cuda.CUmemLocation()
        location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        location.id = dev_id
        allocation_prop = cuda.CUmemAllocationProp()
        allocation_prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED

        # TODO: We differentiate FABRIC for GB200 (aarch64) and POSIX_FILE_DESCRIPTOR for BB200 (x86_64).
        # May need to find a better way to handle this.
        arch = platform.machine().lower()
        is_on_aarch64 = "aarch64" in arch
        if is_on_aarch64:
            allocation_prop.requestedHandleTypes = (
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            )
        else:
            allocation_prop.requestedHandleTypes = (
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            )
        allocation_prop.location = location
        return allocation_prop

    @staticmethod
    def get_allocation_granularity(dev_id: int):
        if MnnvlMemory.allocation_granularity != 0:
            return MnnvlMemory.allocation_granularity
        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        option = cuda.CUmemAllocationGranularity_flags(
            cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
        )
        granularity = _check_cu_result(
            cuda.cuMemGetAllocationGranularity(prop=allocation_prop, option=option)
        )
        MnnvlMemory.allocation_granularity = granularity
        return MnnvlMemory.allocation_granularity

    @staticmethod
    def new_mnnvl_memory_address(mapping: Mapping, size: int):
        page_count = (size + MnnvlMemory.fabric_page_size - 1) // MnnvlMemory.fabric_page_size
        current_rank_stride = page_count * MnnvlMemory.fabric_page_size
        logger.info(f"[MnnvlMemory] creating address with stride={current_rank_stride}")
        comm = MnnvlMemory.get_comm(mapping)
        comm_size = comm.Get_size()
        address_size = current_rank_stride * comm_size
        ptr = _check_cu_result(
            cuda.cuMemAddressReserve(address_size, MnnvlMemory.fabric_page_size, 0, 0)
        )
        MnnvlMemory.current_start_address = int(ptr)
        MnnvlMemory.current_rank_stride = current_rank_stride
        MnnvlMemory.current_mem_offset = 0

    @staticmethod
    def open_mnnvl_memory(mapping: Mapping, size: int):
        dev = _check_cu_result(cuda.cuCtxGetDevice())
        dev_id = int(dev)
        if MnnvlMemory.dev_id is None:
            MnnvlMemory.dev_id = dev_id
        assert dev_id == MnnvlMemory.dev_id, (
            f"Different dev_id found dev_id={dev_id} but MnnvlMemory.dev_id={MnnvlMemory.dev_id}"
        )
        comm = MnnvlMemory.get_comm(mapping)
        comm_rank = comm.Get_rank()
        comm_size = comm.Get_size()
        all_rank_allocate_sizes = comm.allgather(size)
        assert len(all_rank_allocate_sizes) == comm_size
        assert all(x == size for x in all_rank_allocate_sizes), "Not all rank allocating same size."
        granularity = MnnvlMemory.get_allocation_granularity(dev_id)
        aligned_size = (size + granularity - 1) // granularity * granularity

        if MnnvlMemory.current_mem_offset + aligned_size > MnnvlMemory.current_rank_stride:
            MnnvlMemory.new_mnnvl_memory_address(mapping, aligned_size)

        assert MnnvlMemory.current_mem_offset + aligned_size <= MnnvlMemory.current_rank_stride

        allocation_prop = MnnvlMemory.get_allocation_prop(dev_id)
        allocated_mem_handle = _check_cu_result(
            cuda.cuMemCreate(aligned_size, allocation_prop, flags=0)
        )
        exported_fabric_handle = _check_cu_result(
            cuda.cuMemExportToShareableHandle(
                allocated_mem_handle, allocation_prop.requestedHandleTypes, 0
            )
        )
        if (
            allocation_prop.requestedHandleTypes
            == cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        ):
            all_handles_data = comm.allgather(exported_fabric_handle.data)
        else:
            # Unix Domain Socket approach for better compatibility and safety
            all_handles_data = comm.allgather(exported_fabric_handle)
            all_pids = comm.allgather(os.getpid())

            # Check if we should use the legacy pidfd_open approach
            use_pidfd = os.environ.get("TENSORRT_LLM_USE_PIDFD", "0") == "1"

            if use_pidfd:
                # Legacy pidfd_open approach (requires Linux 5.3+)
                logger.warning(
                    "Using legacy pidfd_open approach."
                    "Consider switching to Unix Domain Sockets for better compatibility."
                )
                libc = ctypes.CDLL(None, use_errno=True)
                syscall = libc.syscall
                SYS_pidfd_open = 434
                SYS_pidfd_getfd = 438
                pidfds = []
                for i, pid in enumerate(all_pids):
                    pidfd = syscall(SYS_pidfd_open, pid, 0)
                    if pidfd < 0:
                        err = ctypes.get_errno()
                        raise RuntimeError(
                            f"pidfd_open({pid}) failed with errno {err}: {os.strerror(err)}"
                        )
                    pidfds.append(pidfd)

                remote_fds = []
                for i, (pidfd, fd) in enumerate(zip(pidfds, all_handles_data)):
                    remote_fd = syscall(SYS_pidfd_getfd, pidfd, fd, 0)
                    if remote_fd < 0:
                        err = ctypes.get_errno()
                        error_msg = f"pidfd_getfd(pidfd={pidfd}, fd={fd}) failed with errno {err}: {os.strerror(err)}."
                        if err == 1:  # EPERM
                            error_msg += (
                                " Permission denied. If running in a container, try adding --cap-add=SYS_PTRACE "
                                "to your docker run command."
                            )
                        else:
                            error_msg += " This may be due to kernel version (requires Linux 5.6+)."
                        raise RuntimeError(error_msg)
                    remote_fds.append(remote_fd)

                all_handles_data = remote_fds
            else:
                # Unix Domain Socket approach (recommended)
                # Start Unix Domain Socket server for this rank
                fd_server = FDExchangeServer(comm_rank)

                # Register file descriptors for all other ranks
                for rank in range(comm_size):
                    if rank != comm_rank:
                        fd_server.register_fd(rank, all_handles_data[comm_rank])

                fd_server.start()

                # Give servers time to start up
                comm.Barrier()  # Synchronize all ranks

                # Exchange file descriptors with all peers
                remote_fds = []
                for i, (peer_pid, peer_fd) in enumerate(zip(all_pids, all_handles_data)):
                    if i == comm_rank:
                        # For our own rank, just use the local FD
                        remote_fds.append(peer_fd)
                    else:
                        # Exchange FD with peer using Unix Domain Socket
                        try:
                            remote_fd = exchange_fd_with_peer(
                                peer_pid,
                                all_handles_data[comm_rank],  # Our FD to send
                                comm_rank,  # Our rank
                                max_retries=20,  # More retries for distributed setup
                            )
                            remote_fds.append(remote_fd)
                        except Exception as e:
                            # Enhanced error message
                            error_msg = (
                                f"Failed to exchange file descriptor with rank {i} (PID {peer_pid}): {str(e)}. "
                                "Unix Domain Socket approach requires all processes to be on the same node. "
                                "For multi-node setups, consider using fabric handles. "
                                "To fall back to pidfd_open, set TENSORRT_LLM_USE_PIDFD=1."
                            )
                            raise RuntimeError(error_msg)

                # Clean up the server
                fd_server.stop()

                # Synchronize before proceeding
                comm.Barrier()

                all_handles_data = remote_fds
        # all_handles_data like b'\x00\x00\x00 \x00\x00\x00\x00\x8f\xec\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\t\x00\x00\x00\x00\x00\x1d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'  # noqa: E501
        # can use buf = memoryview(data) to import if using plain buffer for data.

        madesc = cuda.CUmemAccessDesc()
        madesc.location = allocation_prop.location
        madesc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        mem_handles = [None] * comm_size

        for i, remote_handle_data in enumerate(all_handles_data):
            rank_ptr = (
                MnnvlMemory.current_start_address
                + MnnvlMemory.current_rank_stride * i
                + MnnvlMemory.current_mem_offset
            )
            if i == comm_rank:
                # Local memory mapping
                mem_handles[i] = allocated_mem_handle
                _check_cu_result(cuda.cuMemMap(rank_ptr, aligned_size, 0, allocated_mem_handle, 0))
            else:
                # Fabric memory mapping
                imported_mem_handle = _check_cu_result(
                    cuda.cuMemImportFromShareableHandle(
                        remote_handle_data, allocation_prop.requestedHandleTypes
                    )
                )
                mem_handles[i] = imported_mem_handle
                _check_cu_result(cuda.cuMemMap(rank_ptr, aligned_size, 0, imported_mem_handle, 0))

            _check_cu_result(cuda.cuMemSetAccess(rank_ptr, aligned_size, [madesc], 1))

        ptr = MnnvlMemory.current_start_address + MnnvlMemory.current_mem_offset
        stride = MnnvlMemory.current_rank_stride
        MnnvlMemory.allocated_map[ptr] = (
            mapping,
            aligned_size,
            mem_handles,
            MnnvlMemory.current_start_address,
            MnnvlMemory.current_rank_stride,
            MnnvlMemory.current_mem_offset,
        )
        MnnvlMemory.address_refcnt[MnnvlMemory.current_start_address] = (
            MnnvlMemory.address_refcnt.get(MnnvlMemory.current_start_address, 0) + 1
        )

        MnnvlMemory.current_mem_offset += aligned_size
        return ptr, stride

    @staticmethod
    def close_mnnvl_memory(ptr: int):
        mapping, aligned_size, mem_handles, start_address, rank_stride, address_offset = (
            MnnvlMemory.allocated_map.pop(ptr)
        )
        comm = MnnvlMemory.get_comm(mapping)
        comm_size = comm.Get_size()
        for i in range(comm_size):
            rank_ptr = start_address + i * rank_stride + address_offset
            _check_cu_result(cuda.cuMemUnmap(rank_ptr, aligned_size))
            _check_cu_result(cuda.cuMemRelease(mem_handles[i]))
        MnnvlMemory.address_refcnt[start_address] -= 1

        if MnnvlMemory.address_refcnt[start_address] == 0:
            MnnvlMemory.address_refcnt.pop(start_address)
            device_ptr = cuda.CUdeviceptr(start_address)
            _check_cu_result(cuda.cuMemAddressFree(device_ptr, comm_size * rank_stride))
            if start_address == MnnvlMemory.current_start_address:
                MnnvlMemory.current_start_address = 0
                MnnvlMemory.current_rank_stride = 0
                MnnvlMemory.current_mem_offset = 0

    @staticmethod
    def support_nvlink(need_all_up: bool = True):
        dev_id = torch.cuda.current_device()
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        link_count = pynvml.NVML_NVLINK_MAX_LINKS
        active_links = 0
        available_links = 0
        for link_idx in range(link_count):
            try:
                if pynvml.nvmlDeviceGetNvLinkCapability(
                    handle, link_idx, pynvml.NVML_NVLINK_CAP_P2P_SUPPORTED
                ):
                    available_links += 1
                    is_active = pynvml.nvmlDeviceGetNvLinkState(handle, link_idx)
                    if is_active:
                        active_links += 1
            except pynvml.NVMLError_NotSupported:
                continue
        return (
            active_links == available_links and available_links > 0
            if need_all_up
            else available_links > 0
        )

    @staticmethod
    def supports_mnnvl() -> bool:
        # TODO:
        # We check if it has all NVLink up now.
        # But it is not equivalent to MNNVL support.
        # May need better support check.
        support_nvlink_and_all_up = MnnvlMemory.support_nvlink(True)
        return support_nvlink_and_all_up


@dataclass
class MoEAlltoallInfo:
    local_gather_indices: torch.Tensor
    send_rank_count_cumsum: torch.Tensor
    send_rank_local_indices: torch.Tensor
    recv_rank_count_cumsum: torch.Tensor
    recv_rank_local_indices: torch.Tensor
    backward_recv_rank_local_indices: torch.Tensor
    local_token_allocation_count: int


class MnnvlMoe:
    moe_workspace: MnnvlMemory = None
    moe_prepare_workspace: MnnvlMemory = None
    moe_workspace_tensor: torch.Tensor = None
    moe_prepare_workspace_tensor: torch.Tensor = None
    moe_mapping: Mapping = None

    @staticmethod
    def get_moe_workspaces(mapping: Mapping):
        if MnnvlMoe.moe_workspace is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_workspace_tensor

        MnnvlMoe.moe_mapping = mapping
        workspace_size_per_rank = torch.ops.trtllm.get_moe_commworkspace_size_per_rank(
            mapping.tp_size
        )
        MnnvlMoe.moe_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_workspace_tensor = MnnvlMoe.moe_workspace.as_torch_strided_tensor(torch.uint64)
        return MnnvlMoe.moe_workspace_tensor

    @staticmethod
    def get_moe_prepare_workspace(mapping: Mapping):
        if MnnvlMoe.moe_prepare_workspace_tensor is not None:
            assert mapping == MnnvlMoe.moe_mapping, "only one moe mapping supported now"
            return MnnvlMoe.moe_prepare_workspace_tensor
        workspace_size_per_rank = torch.ops.trtllm.get_moe_prepare_workspace_size_per_rank(
            mapping.tp_size
        )
        MnnvlMoe.moe_prepare_workspace = MnnvlMemory(mapping, workspace_size_per_rank)
        MnnvlMoe.moe_prepare_workspace_tensor = (
            MnnvlMoe.moe_prepare_workspace.as_torch_strided_tensor(torch.uint64)
        )
        return MnnvlMoe.moe_prepare_workspace_tensor

    @staticmethod
    def compute_target_rank_id(
        token_selected_experts: torch.Tensor, expert_count: int, ep_size: int
    ):
        assert expert_count % ep_size == 0, "expert_count should be divisible by ep_size"
        expert_per_rank = expert_count // ep_size
        token_target_rank_ids = token_selected_experts // expert_per_rank
        return token_target_rank_ids

    @staticmethod
    def mnnvl_moe_alltoallv_prepare_without_allgather(
        expert_ids: torch.Tensor,
        scales: torch.Tensor,
        expert_statics: Optional[torch.Tensor],
        workspace: torch.Tensor,
        max_token_count_per_rank: int,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
        slot_count: int,
        top_k: int,
    ):
        (
            prepared_local_experts,
            prepared_local_scales,
            local_send_rank_count_cumsum,
            local_send_rank_indices,
            local_recv_rank_count_cumsum,
            local_recv_rank_indices,
            backward_local_recv_rank_indices,
            gathered_expert_statics,
        ) = torch.ops.trtllm.mnnvl_moe_alltoallv_prepare_without_allgather(
            expert_ids,
            scales,
            expert_statics,
            workspace,
            max_token_count_per_rank,
            ep_rank,
            ep_size,
            expert_count,
            slot_count,
            top_k,
        )

        local_token_allocation_count = max_token_count_per_rank * ep_size
        # Looks like we don't need this.
        local_gather_indices = None

        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            local_send_rank_count_cumsum,
            local_send_rank_indices,
            local_recv_rank_count_cumsum,
            local_recv_rank_indices,
            backward_local_recv_rank_indices,
            local_token_allocation_count,
        )

        return alltoall_info, prepared_local_experts, prepared_local_scales, gathered_expert_statics

    @staticmethod
    def mnnvl_moe_expert_static_allgather(
        expert_ids: torch.Tensor,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        expert_count: int,
    ):
        gathered_expert_ids = torch.ops.trtllm.mnnvl_moe_expert_static_allgather(
            expert_ids, workspace, ep_rank, ep_size, expert_count
        )
        return gathered_expert_ids

    @staticmethod
    def mnnvl_moe_alltoallv_prepare(
        gathered_target_rank_ids: torch.Tensor,
        real_rank_token_count_cumsum: Optional[torch.Tensor],
        gathered_expert_ids: torch.Tensor,
        gathered_scales: Optional[torch.Tensor],
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ):
        (
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
        ) = torch.ops.trtllm.moe_comm_prepare_indices(
            gathered_target_rank_ids,
            real_rank_token_count_cumsum,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

        local_token_allocation_count = max_token_count_per_rank * ep_size

        local_expert_ids = torch.empty(
            local_token_allocation_count, top_k, dtype=torch.int32, device=torch.device("cuda")
        )
        if gathered_scales is None:
            local_scales = None
        else:
            local_scales = torch.empty(
                local_token_allocation_count,
                top_k,
                dtype=torch.float32,
                device=torch.device("cuda"),
            )

        torch.ops.trtllm.moe_local_gather(
            recv_rank_count_cumsum,
            local_gather_indices,
            gathered_expert_ids,
            gathered_scales,
            local_expert_ids,
            local_scales,
            max_token_count_per_rank,
            expert_count,
            top_k,
            ep_rank,
            ep_size,
        )

        alltoall_info = MoEAlltoallInfo(
            local_gather_indices,
            send_rank_count_cumsum,
            send_rank_local_indices,
            recv_rank_count_cumsum,
            recv_rank_local_indices,
            backward_recv_rank_local_indices,
            local_token_allocation_count,
        )
        return alltoall_info, local_expert_ids, local_scales

    @staticmethod
    def mnnvl_moe_alltoallv(
        x: torch.Tensor,
        alltoall_info: MoEAlltoallInfo,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
    ):
        assert x.dim() == 2, "only 2D tensor supported, please reshape."
        output_tensor = torch.empty(
            alltoall_info.local_token_allocation_count,
            x.shape[1],
            dtype=x.dtype,
            device=torch.device("cuda"),
        )
        torch.ops.trtllm.moe_comm(
            x,
            alltoall_info.send_rank_count_cumsum,
            alltoall_info.send_rank_local_indices,
            output_tensor,
            alltoall_info.recv_rank_count_cumsum,
            alltoall_info.recv_rank_local_indices,
            workspace,
            ep_rank,
            ep_size,
        )
        return output_tensor

    @staticmethod
    def mnnvl_moe_alltoallv_combine(
        x: torch.Tensor,
        alltoall_info: MoEAlltoallInfo,
        workspace: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        top_k: int,
        token_count: int,
    ):
        assert x.dim() == 2, "2D tensor supported, please reshape."
        output_tensor = torch.zeros(
            token_count * top_k, x.shape[1], dtype=x.dtype, device=torch.device("cuda")
        )
        torch.ops.trtllm.moe_comm(
            x,
            alltoall_info.recv_rank_count_cumsum,
            alltoall_info.recv_rank_local_indices,
            output_tensor,
            alltoall_info.send_rank_count_cumsum,
            alltoall_info.backward_recv_rank_local_indices,
            workspace,
            ep_rank,
            ep_size,
        )
        return torch.sum(
            output_tensor.reshape(token_count, top_k, x.shape[1]), dim=1, keepdim=False
        )
