"""
Loop elimination pass that removes GridLoops containing optional axes.
"""
from .nodes import IRNode, Program, GridLoop
from typing import List, Optional

def eliminate_loops(program: Program):
    """
    Remove GridLoops that contain optional axes.
    
    Args:
        program: The program to transform
        
    Returns:
        Program with optional loops eliminated
    """
    def gather_loops(node: IRNode) -> Optional[GridLoop]:
        if isinstance(node, GridLoop):
            return node
        return None
    
    # collect all loops in the program
    loops = program.visit(gather_loops)
    
    # iterate through all loops
    for loop in loops:
        for axis, type_ in zip(loop.axes, loop.axis_types):
            # if axis.max_block_size != axis.size:
            axis.min_block_size = axis.max_block_size
            # kept_axes.append(axis)
            # kept_types += type_
        
        # loop.axes = kept_axes
        # loop.axis_types = kept_types