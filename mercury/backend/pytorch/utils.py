# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

from functools import cache
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.distributed as dist

class SendRecv:
    def __init__(self):
        self._ops = []
        self._reqs = None

    def send_recv(
        self, to_rank: int, from_rank: int, send_tensor: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(send_tensor)
        else:
            res = recv_tensor
        send_tensor = send_tensor.contiguous()
        send_op = dist.P2POp(dist.isend, send_tensor, to_rank)
        recv_op = dist.P2POp(dist.irecv, res, from_rank)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

@cache
def one_dim_to_n_dim(one_dim_index: int, dimensions: Tuple[int]) -> Tuple[int]:
    """
    turn 1D index to n-D index.

    Args:
        one_dim_index: 1D index.
        dimensions: (D1, D2, ..., Dn) sizes of each dimension.

    Returns:
        n-D index [i1, i2, ..., in]
    """
    n_dim_index = []
    remainder = one_dim_index

    total_device = 1
    for dim_size in dimensions:
        total_device *= dim_size

    for dim_size in reversed(dimensions):
        total_device //= dim_size
        index = remainder // total_device
        remainder %= total_device
        n_dim_index.insert(0, index)

    return tuple(n_dim_index)

@cache
def n_dim_to_one_dim(n_dim_index: Tuple[int], dimensions: Tuple[int]) -> int:
    """
    turn n-D index to 1D index.
    Args:
        n_dim_index: n-D index [i1, i2, ..., in].
        dimensions: (D1, D2, ..., Dn) sizes of each dimension.
    Returns:
        1D index.
    """
    one_dim_index = 0
    for index, dim_size in zip(reversed(n_dim_index),reversed(dimensions)):
        one_dim_index = one_dim_index * dim_size + index

    return one_dim_index

@cache
def shift_tuple_element(original_tuple: tuple, index: int, value: int, modulo: int) -> tuple:
    """
    build a new tuple with the element at the specified index modified by the given value, with modulo.

    Args:
        original_tuple: The original tuple.
        index: The index of the element to be modified.
        value: The value to be added or subtracted, positive for addition and negative for subtraction
        modulo: The value to take modulo with.

    Returns:
        The new tuple.
    """
    list_version = list(original_tuple) 
    list_version[index] += value
    
    list_version[index] %= modulo
    if list_version[index] < 0:
        list_version[index] += modulo
    
    new_tuple = tuple(list_version)
    return new_tuple

def debug_print(*args):
    for rank in range(dist.get_world_size()):
        if rank == dist.get_rank():
            print(f"[Rank {rank}]", *args)
        dist.barrier()

@cache
def get_device_group(indices: Tuple[int], mesh_shape: Tuple[int], dims: Tuple[int], debug=False) -> dist.ProcessGroup | List:
    """
    get the communication group of the node in the specified dimension range
    Args:
        indices: The n-dimensional coordinates of the current node in the mesh.
        mesh_shape: The shape of the mesh as a tuple.
        dims: The dimensions to be used for the communication group.
        debug: Whether to use the debug mode.
    Returns:
        The communication group of the node.
    """
    for dim in dims:
        if dim < 0 or dim >= len(mesh_shape):
            raise ValueError(f"Invalid dimension index: {dim}, mesh shape: {mesh_shape}")

    dict_group = {}

    world_size = np.prod(mesh_shape)
    res_group = None
    
    # dist.new_group
    # This function requires that all processes in the main group 
    # (i.e. all processes that are part of the distributed job) enter this function, 
    # even if they are not going to be members of the group. Additionally, 
    # groups should be created in the same order in all processes.
    for rank in range(world_size):
        # compute all nodes in the group
        group_ranks = []
        # compute the rank of the current node in the mesh
        rank_indices = one_dim_to_n_dim(rank, mesh_shape)
        template_indices = list(rank_indices)

        # search all the coordinates in the specified dimension range
        def collect_ranks(dim_list_id, curr_indices):
            if dim_list_id >= len(dims):
                group_ranks.append(n_dim_to_one_dim(tuple(curr_indices), mesh_shape))
                return
            dim_idx = dims[dim_list_id]
            for i in range(mesh_shape[dim_idx]):
                curr_indices[dim_idx] = i
                collect_ranks(dim_list_id + 1, curr_indices)

        collect_ranks(0, template_indices)
        group_ranks = tuple(sorted(group_ranks))

        if dict_group.get(group_ranks) is None:
            if debug:
                dict_group[group_ranks] = group_ranks
            else:
                dict_group[group_ranks] = dist.new_group(ranks=group_ranks)
        
        if indices == rank_indices:
            res_group = dict_group[group_ranks]

    return res_group

def get_src_dst_ranks(indices: Tuple[int], mesh_shape: Tuple[int], ring_dims: List[int]) -> Tuple[int, int]:
    """ Used in reduce rings, where we need to put the output back to the right rank """

    # get the src of current rank
    src_indice = indices
    last_dim_len = 1
    for dim in reversed(ring_dims):
        src_indice = shift_tuple_element(src_indice, dim, -last_dim_len, mesh_shape[dim])
        # print(src_indice)
        last_dim_len = mesh_shape[dim]

    # get the dst of current rank
    dst_indice = indices
    last_dim_len = 1
    for dim in reversed(ring_dims):
        dst_indice = shift_tuple_element(dst_indice, dim, last_dim_len, mesh_shape[dim])
        # print(dst_indice)
        last_dim_len = mesh_shape[dim]

    src_rank = n_dim_to_one_dim(src_indice, mesh_shape)
    dst_rank = n_dim_to_one_dim(dst_indice, mesh_shape)
    return src_rank, dst_rank