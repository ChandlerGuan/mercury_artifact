# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

from typing import Optional
from mercury.ir.elements import Axis, Buffer
from mercury.ir.nodes import BufferMatch, IRNode
from mercury.ir.utils import collect_reduce, get_buffers, get_element_size


def get_buffer_size(program: IRNode) -> int:
    """
    return the per card size of the buffers in the ir
    Notice: if you used some temp buffer in your code
    you should add them in the return value, because
    they are not tracked by the ir
    """
        
    buffers = program.visit(get_buffers)
    reduce_ops = program.visit(collect_reduce)

    total_size = 0
    for buffer in buffers:
        buffer_size = 1

        is_reduce = False
        ring_factor = 1

        # check if is reduce buffer
        for reduce_op in reduce_ops:
            if buffer.tensor == reduce_op.buffer.tensor:
                is_reduce = True
                break
        
        for dim_len, dim_axes in zip(buffer.shape, buffer.bound_axes):
            if isinstance(dim_len, Axis):
                dim_len = dim_len.min_block_size
            
            is_ring = False
            for axis in dim_axes:
                if buffer.tensor + axis.name in axis.ring_comm:
                    is_ring = True
                    break

            if is_ring:
                # when buffer is write, then it must be a reduce buffer
                # in this case, we will communicate the buffer in the last round to get the final result
                # so there is no need to keep the original buffer
                # otherwise, to reduce the last round communication, we will keep the buffer
                # and the factor is twice the block size
                if buffer.write:
                    double_buffer_factor = dim_axes[-1].min_block_size
                else:
                    double_buffer_factor = 2 * dim_axes[-1].min_block_size
                ring_factor *= 1 + double_buffer_factor / dim_len

            buffer_size *= dim_len

        buffer_size *= get_element_size(buffer.dtype)

        total_size += buffer_size * ring_factor # ring factor is the double buffer factor

        if is_reduce:
            total_size += buffer_size # if is reduce buffer, we would need another buffer to store local res

    return total_size