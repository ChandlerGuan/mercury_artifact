from typing import List, Optional
import torch

from mercury.ir.elements import Axis, Buffer
from mercury.ir.nodes import AxisDef, BufferLoad, BufferMatch, BufferStore, GridLoop, IRNode, Program, ReduceOp

def get_element_size(dtype):
    try:
        dtype_info = torch.finfo(dtype)
        dtype_size = dtype_info.bits // 8
    except TypeError:
        try:
            dtype_info = torch.iinfo(dtype)
            dtype_size = dtype_info.bits // 8
        except TypeError:
            raise ValueError(f"Unsupported dtype: {dtype}")
    return dtype_size

def get_buffers(node: IRNode) -> Optional[Buffer]:
    if isinstance(node, BufferMatch):
        return node.buffer

def get_io_buffers(node: IRNode) -> Optional[Buffer]:
    if isinstance(node, BufferMatch) and node.tensor_name is not None:
        return node.buffer
    
def collect_parallelizeable_axes(node: IRNode) -> Optional[List[Axis]]:
    if isinstance(node, GridLoop):
        res = []
        for id, axis in enumerate(node.axes):
            if node.axis_types[id] == "s" or node.axis_types[id] == "m":
                res.append(axis)
        return res
    return None

def collect_loops(node: IRNode) -> Optional[GridLoop]:
    return node if isinstance(node, GridLoop) else None

def collect_reduce(node: IRNode) -> Optional[ReduceOp]:
    return node if isinstance(node, ReduceOp) else None

def collect_axis(node: IRNode) -> Optional[Axis]:
    return node.axis if isinstance(node, AxisDef) else None

def get_inner_buffer(program: Program, axis: Axis) -> List[IRNode]:
    """get the buffer ops inner to the input, notice that axis must in loop"""

    def outer_func(init_met_axis: bool):
        # use a dict to store the state of met_axis
        state = {'met_axis': init_met_axis}
        
        def collect_inner_buffer(node):
            if isinstance(node, GridLoop):
                if axis in node.axes:
                    state['met_axis'] = True
            elif isinstance(node, BufferLoad) or isinstance(node, BufferStore) or isinstance(node, ReduceOp):
                if state['met_axis']:
                    return node
            return None
        
        def undo_met_axis(node):
            if isinstance(node, GridLoop):
                if axis in node.axes:
                    state['met_axis'] = False

        return collect_inner_buffer, undo_met_axis

    collect_inner_buffer, undo_met_axis = outer_func(False)
    return program.visit(collect_inner_buffer, undo_met_axis)

def get_potential_ring(node: IRNode) -> Optional[Buffer]:
    if isinstance(node, BufferLoad) or isinstance(node, BufferStore) or isinstance(node, ReduceOp):
        return node