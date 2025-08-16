# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

'''
parallize loop in spatial axis
'''

import copy
from typing import List, Optional
import numpy as np

from mercury.ir.distributed import DeviceMesh, ShardType
from mercury.ir.utils import collect_reduce, get_inner_buffer
from .nodes import GridLoop, BufferLoad, BufferStore, IRNode, Program, ReduceOp, RingComm
from .elements import Axis, Buffer

def identify_buffer_commands(ir: IRNode):
    """collect the commands related to the buffer in the axis"""
    def collect_buffers(node):
        if (isinstance(node, BufferStore) or isinstance(node, BufferLoad)):
            return node
    
    commands = ir.visit(collect_buffers)
    return commands

def check_split(axis: Axis, cards: int) -> bool:
    """
    check if the axis can be split into cards
    """
    if axis.size % cards != 0 or axis.size / cards < axis.min_block_size:
        return False
    return True

def shift(program: Program, axis: Axis, mesh: DeviceMesh, start_dim: int, end_dim: int, ring_divide: int, used_axes: Optional[set] = None):
    """
    ring divide is not used now because it is replaced by split axis
    ring_divide is the temp parameter for easy test, controlling the the rounds to do ring comm
    with a ring divide to more rounds, its memory usage will be less

    for 1 round, the memory usage is 3 * total_size / num_cards
    we don't do the last step of ring,
    so the original data must be stored, and the double buffer needs 2 more, resulting in 3

    for 2 rounds, the memory usage is (1 + 2 * 0.5) * (total_size / num_cards)
    for i rounds, the memory usage is (1 + 2 / i) * (total_size / num_cards)
    """
    if start_dim >= end_dim:
        return
    axes = get_loop_axis(program)
    buffer_commands = get_inner_buffer(program, axis)
    all_buffer_commands = identify_buffer_commands(program)
    
    buffer_active = set(command.buffer.tensor for command in buffer_commands) # if the axis is outer than the buffer def, the buffer is not active
    next_axis_active = copy.deepcopy(buffer_active)

    for inner_axis in reversed(axes): # start from the innermost axis
        if inner_axis == axis:
            continue
        if used_axes is not None and inner_axis.name in used_axes:
            continue
        if start_dim >= end_dim:
            break
        if inner_axis.ring_comm_cards != 1:
            continue

        axis_size = inner_axis.size
        if axis_size % mesh.shape[start_dim] != 0 or axis_size / mesh.shape[start_dim] < inner_axis.min_block_size:
            continue
        
        num_cards = mesh.shape[start_dim]

        divided = set()

        need_ring = False
        for command in buffer_commands:
            if command.buffer.tensor not in buffer_active:
                continue
            if command.buffer.def_axis is not None and command.buffer.def_axis == inner_axis and command.buffer.tensor in next_axis_active:
                next_axis_active.remove(command.buffer.tensor)
            if isinstance(command, BufferStore) or isinstance(command, BufferLoad):
                if axis not in command.indices and inner_axis in command.indices:
                    name = command.buffer.tensor + inner_axis.name
                    command.comm.append(RingComm(axis = inner_axis, num_cards = num_cards, name=name, shard_dim=start_dim))
                    if command.buffer.tensor not in divided:
                        divided.add(command.buffer.tensor)
                        cut_buffer(command.buffer, inner_axis, start_dim, start_dim + 1, num_cards)
                        inner_axis.ring_comm.append(name)
                        need_ring = True
            elif isinstance(command, ReduceOp):
                if not command.buffer.has_axis(axis) and command.buffer.has_axis(inner_axis) and axis in command.axes:
                    name = command.buffer.tensor + inner_axis.name
                    command.comm.append(RingComm(axis = inner_axis, num_cards = num_cards, name=name, shard_dim=start_dim, write_back=True))
                    if command.buffer.tensor not in divided:
                        divided.add(command.buffer.tensor)
                        cut_buffer(command.buffer, inner_axis, start_dim, start_dim + 1, num_cards)
                        inner_axis.ring_comm.append(name)
                        need_ring = True
        
        if need_ring:
            inner_axis.max_block_size = min(inner_axis.max_block_size, inner_axis.size // num_cards)
            inner_axis.ring_comm_cards = num_cards
            if inner_axis.max_block_size % ring_divide == 0:
                inner_axis.max_block_size //= ring_divide

            # divide the other buffers related to the axis
            for command in all_buffer_commands:
                if command.buffer.tensor not in divided and command.buffer.has_axis(inner_axis): # inner_axis in command.indices:
                    divided.add(command.buffer.tensor)
                    cut_buffer(command.buffer, inner_axis, start_dim, start_dim + 1, num_cards)
            
            start_dim += 1

        buffer_active = copy.deepcopy(next_axis_active)
    

def get_loop_axis(program: Program) -> List[Axis]:
    """get the axes in loops"""
        
    def collect_inner_axis(node):
        if not isinstance(node, GridLoop):
            return None
            
        ret_axis = []
        for cur_axis in node.axes:
            ret_axis.append(cur_axis)
        
        return ret_axis if ret_axis else None
            
    ret_axis = program.visit(collect_inner_axis)
    if ret_axis is None:
        return []
    
    ret = []
    for sublist in ret_axis:
        ret.extend(sublist)
    return ret

def cut_buffer(buffer: Buffer, axis: Axis, start_dim: int, end_dim: int, num_cards: int):
    """
    cut the buffer in the axis
    """
    if buffer.has_axis(axis):
        dim_id, _ = buffer.get_axis(axis)
        if isinstance(buffer.shape[dim_id], int): # dynamic shape doesn't need to be cut, as it will be cut along with the axis
            buffer.shape[dim_id] = int(buffer.shape[dim_id] // num_cards)
        dim_list = list(range(start_dim, end_dim))
        if buffer.shard_spec.specs[dim_id] == ShardType.REPLICATE:
            buffer.shard_spec.specs[dim_id] = (ShardType.SHARD, dim_list)
        else:
            old_shard_dim = buffer.shard_spec.specs[dim_id][1]
            old_shard_dim.extend(dim_list)
            new_shard_dim = list(set(old_shard_dim))
            buffer.shard_spec.specs[dim_id] = (ShardType.SHARD, new_shard_dim)
        buffer.shard_spec.validate()
    else:
        raise ValueError("The buffer doesn't have the axis")

def parallelize(program: Program, loop: GridLoop, axis: Axis, mesh: DeviceMesh, start_dim: int, end_dim: int, used_axes:set = set()) -> bool:
    """
    parallelize the loop in the axis
    if success, return True and the loop will be refactored
    otherwise return False, and the loop will not be changed
    """
    if start_dim >= end_dim:
        return True
    num_cards = np.prod(mesh.shape[start_dim:end_dim])

    axis_type = loop.get_axis_type(axis)

    # check the axis can be splited
    if not check_split(axis, num_cards):
        return False
    
    if axis_type == 'r':
        return False

    buffer_commands = identify_buffer_commands(program)
    
    related_buffer_commands = []
    for command in buffer_commands:
        if command.buffer.has_axis(axis):
            related_buffer_commands.append(command)
    if len(related_buffer_commands) == 0:
        raise ValueError("No buffer related to the axis, can't be a spatial axis")
    
    for command in related_buffer_commands:
        for axes in command.buffer.bound_axes:
            used_axes.update(related_axis.name for related_axis in axes)
        cut_buffer(command.buffer, axis, start_dim, end_dim, num_cards)

    if axis_type == 'm':
        # managed reduction axis
        reduce_ops = program.visit(collect_reduce)
        related_reduce_ops = []
        for reduce_op in reduce_ops:
            if axis in reduce_op.axes:
                related_reduce_ops.append(reduce_op)

        if len(related_reduce_ops) == 0:
            raise ValueError("No reduce operation related to the axis, can't be a managed axis")
        # elif len(related_reduce_ops) > 1:
        #     raise ValueError("More than one reduce operation related to the axis, can't be a managed axis")
        
        for reduce_op in related_reduce_ops:
            reduce_op.shard_dim.extend(list(range(start_dim, end_dim)))
        # related_reduce_ops[0].shard_dim.extend(list(range(start_dim, end_dim)))

    # split the axis
    axis.size = axis.size // num_cards
    axis.parallel_info = (num_cards, start_dim, end_dim)
    axis.max_block_size = min(axis.max_block_size, axis.size)
    return True
    