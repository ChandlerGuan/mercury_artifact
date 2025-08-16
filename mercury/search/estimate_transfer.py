# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

""" estimate the transfer time of one shard to another shard """

import os
from typing import List, Dict, Tuple, Optional
import torch
import torch.distributed as dist
import time
import numpy as np
from mercury.backend.pytorch.utils import n_dim_to_one_dim, one_dim_to_n_dim
from mercury.ir.elements import Buffer
from mercury.ir.distributed import ShardType
from mercury.ir.utils import get_element_size


def estimate_transfer_time(buffer_origin: List[Buffer], buffer_new: List[Buffer], rounds=50, debug: bool = False, debug_rank=None):
    """ 
    use random data to do the transfer
    to support arbitry shard pattern, we use P2POp to do the
    transfer, this will give a upper bound of the transfer time,
    as collectives are more efficient.

    For simplicity, we assume there is no replicated data in the origin buffer
    TODO: add more support to collective usage
    chance: e.g. (batch, seq/8) -> (batch/4, seq/2) with 8 devices
    the inital device mesh is (8,) and the final device mesh is (4, 2)
    note that seq is in dim 1, batch in dim 0, in this case,
    we can use all2all for each group in dim 0, i.e. 2 all2all for device 0-3 and 4-7
    
    Args:
        buffer_origin: the buffer to transfer from
        buffer_new: the buffer to transfer to
    Returns:
        the estimated transfer time
    """
    if not debug:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = debug_rank
        world_size = 8
        local_rank = rank
    device = torch.device(f"cuda:{local_rank}")
    
    # ensure the buffer list have the same length
    assert len(buffer_origin) == len(buffer_new), "Buffer lists must have the same length"
    
    p2p_amount = [0 for _ in range(world_size)]

    all_to_all_buffers: List[Tuple[torch.Tensor, torch.Tensor]] = []

    # determine the communication need for each buffer
    for i, (orig_buf, new_buf) in enumerate(zip(buffer_origin, buffer_new)):
        # mesh can be different, so we need to get the mesh shape
        origin_mesh = orig_buf.shard_spec.mesh.shape
        new_mesh = new_buf.shard_spec.mesh.shape
        indice = one_dim_to_n_dim(rank, new_mesh) # all indice will follow the new mesh

        if debug:
            print(f"Rank {rank} checking buffer {orig_buf.tensor}")
        # skip the buffer if it is not sharded
        if orig_buf.shard_spec is None or new_buf.shard_spec is None:
            raise RuntimeError("These buffer must have shard spec be initialized")

        all2all_buf = try_all2all(orig_buf, new_buf, debug, debug_rank)
        if all2all_buf is not None:
            all_to_all_buffers.append(all2all_buf)
            continue
            
        # get the global data range of the current rank in the new buffer
        # orig_ranges = get_shard_ranges(orig_buf, rank, origin_mesh)
        new_ranges = get_shard_ranges(new_buf, rank, new_mesh)

        if debug:
            print(f"Rank {rank} new_ranges: {new_ranges}")
        
        # get the receive data range from other ranks

        range_to_rank = {}
        
        for other_rank in range(world_size):

            # compute the global data range of the other rank in the original buffer
            other_orig_ranges = get_shard_ranges(orig_buf, other_rank, origin_mesh)

            if debug:
                print(f"Rank {rank} other_orig_ranges: {other_orig_ranges}")
            
            # if the new range is overlapped with the other rank's original range
            # we need to receive data from the other rank
            recv_range = ranges_overlap(new_ranges, other_orig_ranges)
            # we will assume that if the recv range is duplicated, it will be totally duplicated

            if debug:
                print(f"Rank {rank} recv_range: {recv_range}")

            if recv_range is None:
                continue

            recv_range = tuple(recv_range)

            if recv_range in range_to_rank:
                old_rank = range_to_rank[recv_range]
                new_rank = other_rank

                old_indice = one_dim_to_n_dim(old_rank, new_mesh)
                new_indice = one_dim_to_n_dim(new_rank, new_mesh)

                if more_near(indice, new_indice, old_indice):
                    # we will use the new rank
                    range_to_rank[recv_range] = new_rank
            else:
                range_to_rank[recv_range] = other_rank

        # check if the new_ranges is fully covered by the other ranks
        range_slices = [list(slice) for slice in range_to_rank.keys()]
        if not is_fully_covered(new_ranges, range_slices):
            print(f"Rank {rank} new_ranges: {new_ranges}, range_to_rank: {range_slices}")
            raise RuntimeError("The new ranges is not fully covered by the other ranks")

        for slice, src_rank in range_to_rank.items():
            if src_rank == rank:
                continue
            # sum the amount of data to receive
            p2p_amount[src_rank] += calculate_volume(slice) * get_element_size(orig_buf.dtype)

    if debug:
        return

    # gather the amount of data to be sent
    p2p_tensor = torch.tensor(p2p_amount, device=device, dtype=torch.int64)

    send_tensor = torch.empty_like(p2p_tensor)

    dist.all_to_all_single(send_tensor, p2p_tensor)

    send_tensor = send_tensor.tolist()
    send_buffers = []
    recv_buffers = []
    for i in range(world_size):
        
        send_size = send_tensor[i] // get_element_size(torch.int8)
        if send_size > 0:
            buffer = torch.empty(send_size, device=device, dtype=torch.int8)
            send_buffers.append(buffer)
        else:
            send_buffers.append(None)
            
        recv_size = p2p_amount[i] // get_element_size(torch.int8)
        if recv_size > 0:
            buffer = torch.empty(recv_size, device=device, dtype=torch.int8)
            recv_buffers.append(buffer)
        else:
            recv_buffers.append(None)
        
    # start to do the transfer
    dist.barrier(device_ids=[local_rank])
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(rounds * 2):
        if i == rounds: # fisrt half is for warmup
            start_event.record()
        ops = []

        # batch send and recv
        for i in range(world_size):
            if send_buffers[i] is not None:
                ops.append(dist.P2POp(dist.isend, send_buffers[i], i))

            if recv_buffers[i] is not None:
                ops.append(dist.P2POp(dist.irecv, recv_buffers[i], i))

        requests = []
        # combine the send and recv ops
        if len(ops) > 0:
            requests = dist.batch_isend_irecv(ops)

        # do the all2all
        for all2all in all_to_all_buffers:
            requests.append(dist.all_to_all_single(all2all[1], all2all[0], async_op = True))
        
        # wait for all requests to finish
        for req in requests:
            req.wait()

    end_event.record()
    torch.cuda.synchronize()
    
    # wait for all ranks to finish
    dist.barrier(device_ids=[local_rank])
    
    # calculate the transfer time
    transfer_time = start_event.elapsed_time(end_event) / rounds
    # get the max transfer time from all ranks
    max_transfer_time = torch.tensor([transfer_time], device=device)
    dist.all_reduce(max_transfer_time, op=dist.ReduceOp.MAX)
    
    return max_transfer_time.item()


def get_shard_coords(buffer: Buffer, rank: int, mesh_shape: Tuple[int, ...]) -> List[int]:
    """compute the shard coordinates of the given rank in the buffer"""
    coords = []

    for dim, spec in enumerate(buffer.shard_spec.specs):
        if isinstance(spec, tuple) and spec[0] == ShardType.SHARD:
            # we will change the way to determin the device coords by using the high dim mesh

            indices = one_dim_to_n_dim(rank, mesh_shape)

            # only use the sharded cords
            shard_coord = tuple([indices[i] for i in spec[1]])
            shard_mesh = tuple([mesh_shape[i] for i in spec[1]])

            coords.append(n_dim_to_one_dim(shard_coord, shard_mesh))

        else:
            # replicate dims
            coords.append(0)
    
    return coords


def get_shard_ranges(buffer: Buffer, rank: int, mesh_shape: Tuple[int, ...]) -> List[Tuple[int, int]]:
    """
    compute the global data range of the given rank in the buffer
    Returns a list of (start, end) tuples for each dimension
    """
    coords = get_shard_coords(buffer, rank, mesh_shape)
    ranges = []
    
    for dim, coord in enumerate(coords):
        dim_size = buffer.shape[dim]
        start = coord * dim_size
        end = start + dim_size
        ranges.append((start, end))
    
    return ranges


def ranges_overlap(ranges1: List[Tuple[int, int]], ranges2: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    """check if two ranges overlap"""
    if len(ranges1) != len(ranges2):
        return None
    
    overlapped_range = []
    # all dimensions must have overlap to be considered overlapping
    for (start1, end1), (start2, end2) in zip(ranges1, ranges2):
        if end1 <= start2 or end2 <= start1:
            return None
        overlapped_range.append((max(start1, start2), min(end1, end2)))
    
    return overlapped_range

def try_all2all(old_buf: Buffer, new_buf: Buffer, debug: bool = False, debug_rank: Optional[int] = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """if possible, we would like to use all2all collective to achive a 2x faster performance compared to p2p"""
    if not debug:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = debug_rank
    device = torch.device(f"cuda:{local_rank}")
    # if buf is not fully sharded, not supported for now
    if not old_buf.shard_spec.fully_sharded() or not new_buf.shard_spec.fully_sharded():
        return None
    common_dims = find_overlapped_dims(old_buf, new_buf)
    world_size = len(old_buf.shard_spec.mesh.devices)
    if len(common_dims) == 0:
        # case1: if no shard axis is shared by the old buffer and the new, it can use all2all
        if debug:
            print(f"can use all2all for{old_buf.tensor}")

        # only need the time, so only need the comm amout to be equal
        buffer_size = np.prod(old_buf.shape)
        shape = [world_size, buffer_size // world_size]
        if debug:
            print(f"shape: {shape}")
        send_tensor = torch.rand(
            shape,
            device=device,
            dtype=old_buf.dtype
        )
        recv_tensor = torch.empty_like(send_tensor)

        return send_tensor, recv_tensor
    else:
        return None

def find_overlapped_dims(old_buf, new_buf) -> List[int]:
    """find the dim that is both shared in old and the new"""

    share_dims = []
    for dim, (old_spec, new_spec) in enumerate(zip(old_buf.shard_spec.specs, new_buf.shard_spec.specs)):
        if isinstance(old_spec, tuple) and old_spec[0] == ShardType.SHARD \
        and isinstance(new_spec, tuple) and new_spec[0] == ShardType.SHARD:
            share_dims.append(dim)

    return share_dims

def more_near(indice, new_indice, old_indice):
    """check if the new indice is more near to the old indice"""
    
    new_diff = tuple([abs(i - j) for i, j in zip(indice, new_indice)])
    old_diff = tuple([abs(i - j) for i, j in zip(indice, old_indice)])

    for dif_new, dif_old in zip(reversed(new_diff), reversed(old_diff)):
        if dif_new < dif_old:
            return True
        elif dif_new > dif_old:
            return False
        
    return False


def is_fully_covered(r0: List[Tuple[int, int]], covering_ranges: List[List[Tuple[int, int]]]) -> bool:
    """
    Check if r0 is fully covered by covering_ranges without overlapping.
    
    Args:
        r0: The target range as a list of (start, end) tuples, one per dimension
        covering_ranges: List of ranges, each formatted like r0
        
    Returns:
        True if r0 is fully covered without overlaps, False otherwise
    """
    # 1. Check if all covering ranges are contained within r0
    for r in covering_ranges:
        overlap = ranges_overlap(r0, r)
        if overlap is None:
            return False
        if ranges_overlap(r0, r) != r:
            return False
    
    # 2. Check for overlaps between covering ranges
    for i in range(len(covering_ranges)):
        for j in range(i+1, len(covering_ranges)):
            if ranges_overlap(covering_ranges[i], covering_ranges[j]) is not None:
                return False
    
    # 3. Check for full coverage by comparing volumes
    r0_volume = calculate_volume(r0)
    total_volume = sum(calculate_volume(r) for r in covering_ranges)
    
    # Use a small epsilon for floating point comparison if needed
    return abs(total_volume - r0_volume) < 1e-9

def calculate_volume(range_: List[Tuple[int, int]]) -> int:
    """Calculate the volume of a multi-dimensional range."""
    volume = 1
    for start, end in range_:
        volume *= (end - start)
    return volume