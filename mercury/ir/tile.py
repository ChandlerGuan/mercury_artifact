""" Split one entire axis into multiple axis """

from mercury.ir.elements import Axis
from mercury.ir.nodes import AxisDef, BufferMatch, Program, GridLoop, IRNode, BufferLoad, BufferStore, ReduceOp
import copy
from typing import List, Optional, Tuple


def tile_loop(program: Program, axis: Axis, split_len: int):
    """
    Split a loop axis into two nested axes.
    note: program should not be initialized as a distributed program
    
    Args:
        program: The Program IR node to transform
        axis: The axis to split
        split_len: The size of the inner axis after splitting
        
    Returns:
        The transformed Program with the axis split into two nested axes
    """
    if split_len <= 0:
        raise ValueError(f"Split length must be positive, got {split_len}")
    
    if axis.size % split_len != 0:
        raise ValueError(f"Axis size {axis.size} must be divisible by split length {split_len}")
    
    if program.mesh is not None:
        raise ValueError("Program should not be initialized as a distributed program")
    
    # Create new axes for the split
    outer_size = axis.size // split_len
    outer_axis = Axis(name=f"{axis.name}_outer", 
                      size=outer_size, 
                      min_block_size=1,
                      max_block_size=1)
    
    inner_axis = Axis(name=f"{axis.name}_inner", 
                      size=split_len, 
                      min_block_size=min(axis.min_block_size, split_len),
                      max_block_size=min(axis.max_block_size, split_len))
    
    # Find all GridLoops that use this axis
    loops = _find_loops_with_axis(program, axis)
    if not loops:
        raise ValueError(f"Axis {axis.name} not found in any loop")
    
    # Replace the axis in each loop
    _replace_axis_in_program(program, axis, outer_axis, inner_axis)


def _find_loops_with_axis(program: Program, axis: Axis) -> List[GridLoop]:
    """Find all GridLoop nodes that contain the specified axis."""
    result = []
    
    def visitor(node: IRNode):
        if isinstance(node, GridLoop) and axis in node.axes:
            result.append(node)
        return None
    
    program.visit(visitor)
    return result


def _replace_axis_in_program(program: Program, old_axis: Axis, outer_axis: Axis, inner_axis: Axis):
    """Replace an axis with two new axes throughout the program."""
    
    def replace_axis_in_loop(node: IRNode):
        if isinstance(node, GridLoop):
            # If this loop contains the old axis, replace it with two new axes
            if old_axis in node.axes:
                idx = node.axes.index(old_axis)
                axis_type = node.axis_types[idx]
                
                # Replace the axis with two new axes
                node.axes = node.axes[:idx] + [outer_axis, inner_axis] + node.axes[idx+1:]
                node.axis_types = node.axis_types[:idx] + axis_type * 2 + node.axis_types[idx+1:]
        
        elif isinstance(node, (BufferLoad, BufferStore)):
            # Replace axis in buffer indices
            id = None
            for i, idx in enumerate(node.indices):
                if idx == old_axis:
                    # Replace with a tuple of the two new axes
                    if id is not None:
                        raise ValueError("Multiple occurrences of axis in buffer indices")
                    id = i

            if id is not None:
                node.indices = node.indices[:id] + [outer_axis, inner_axis] + node.indices[id+1:]
            
            # the buffer would be updated in the buffer match command
        
        elif isinstance(node, ReduceOp):
            # Replace the reduction axis if needed
            id = None
            for i, axis in enumerate(node.axes):
                if axis == old_axis:
                    if id is not None:
                        raise ValueError("Multiple occurrences of axis in reduction axes")
                    id = i

            if id is not None:
                node.axes = node.axes[:id] + [outer_axis, inner_axis] + node.axes[id+1:]

            if node.indices is not None:
                # Replace axis in buffer indices
                id = None
                for i, idx in enumerate(node.indices):
                    if idx == old_axis:
                        # Replace with a tuple of the two new axes
                        if id is not None:
                            raise ValueError("Multiple occurrences of axis in buffer indices")
                        id = i

                if id is not None:
                    node.indices = node.indices[:id] + [outer_axis, inner_axis] + node.indices[id+1:]
                
                # the buffer would be updated in the buffer match command

        elif isinstance(node, BufferMatch):
            # Replace axis in buffer shape
            for id, dim in enumerate(node.buffer.shape):
                if dim == old_axis:
                    node.buffer.shape[id] = inner_axis

            # Replace define axis if needed
            if node.buffer.def_axis is not None and node.buffer.def_axis == old_axis:
                node.buffer.def_axis = outer_axis

            # Replace axis in bound axes
            if node.buffer.has_axis(old_axis):
                dim, pos = node.buffer.get_axis(old_axis)

                # Update the bound axes
                node.buffer.bound_axes[dim] = node.buffer.bound_axes[dim][:pos] \
                    + [outer_axis, inner_axis] + node.buffer.bound_axes[dim][pos+1:]

                # Also update the axes_factor outer axis be 1, inner axis be the original factor
                node.buffer.axes_factor[dim] = node.buffer.axes_factor[dim][:pos] \
                    + [1, node.buffer.axes_factor[dim][pos]] + node.buffer.axes_factor[dim][pos+1:]
                
        elif isinstance(node, Program):
            # replace the axis defined in the program
            idx = None
            for i, command in enumerate(node.body):
                if isinstance(command, AxisDef) and command.axis == old_axis:
                    if idx is not None:
                        raise ValueError("Multiple occurrences of axis in program")
                    idx = i

            if idx is None:
                raise ValueError("Axis not defined in program")
            node.body = node.body[:idx] + [AxisDef(axis=outer_axis), AxisDef(axis=inner_axis)] + node.body[idx+1:]
            
    program.visit(replace_axis_in_loop)