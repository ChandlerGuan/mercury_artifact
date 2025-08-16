# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""
Frontend parser for converting Python AST to auto-ring IR.
"""
import ast
import inspect
from typing import Dict, List, Any, Union, Optional, Tuple
import textwrap
import torch

from ..ir.elements import Axis, Buffer
from ..ir.nodes import (
    Program, AxisDef, BufferMatch, GridLoop,
    BufferStore, BufferLoad, IRNode, PyNode, ReduceOp
)

class IRBuilder(ast.NodeVisitor):
    """AST visitor that builds auto-ring IR nodes."""
    
    def __init__(self):
        self.axes: Dict[str, Axis] = {}  # Map of axis name to Axis object
        self.buffers: Dict[str, Buffer] = {}  # Map of buffer name to Buffer object
        self.current_program = None
        self.loop_vars: Dict[str, Axis] = {}  # Map of loop var name to Axis
        self.cur_loop: Optional[GridLoop] = None

    def is_buffer_name(self, name: str) -> bool:
        """Check if a name refers to a buffer."""
        return name in self.buffers
    
    def get_vars_in_scope(self) -> Dict[str, Union[Axis, Buffer]]:
        """Get all variables currently in scope."""
        vars = {}
        vars.update(self.axes)
        vars.update(self.buffers)
        vars.update(self.loop_vars)
        return vars
    
    def visit(self, node: ast.AST) -> Any:
        """Override visit to wrap unknown nodes in PyNode."""
        try:
            result = super().visit(node)
            if result is None and not isinstance(node, ast.Constant):
                # Node wasn't handled by any specific visitor
                # Wrap it in PyNode with current scope
                return PyNode(node=node, vars_in_scope=self.get_vars_in_scope())
            return result
        except NotImplementedError:
            # No specific visitor found, preserve as Python
            return PyNode(node=node, vars_in_scope=self.get_vars_in_scope())
        
    def get_indices(self, indice_raw: List[Union[Axis, Tuple[Axis]]]) -> List[Axis]:
        indices = []
        for indice in indice_raw:
            if isinstance(indice, tuple):
                for axis in indice:
                    indices.append(axis)
            else:
                indices.append(indice)
        return indices
    
    def visit_Buffer(self, node) -> Buffer:
        """Process buffer"""
        buffer_name = self.visit(node)
        if buffer_name not in self.buffers:
            raise ValueError(f"Buffer {buffer_name} not found")
        buffer = self.buffers[buffer_name]
        return buffer
    
    def visit_IndexedBuffer(self, node):
        """Process buffer with indices"""
        buffer = self.visit_Buffer(node.value)
        indices = self.get_indices(self.visit(node.slice))
        return buffer, indices
    
    def visit_BufferLoad(self, target: ast.Name, call: ast.Call) -> BufferLoad:
        """Process buffer loading."""
        if len(call.args) != 1:
            raise ValueError("load_buffer requires a single argument")
        
        name = target.id
        buffer, indices = self.visit_IndexedBuffer(call.args[0])
        buffer.read = True
        return BufferLoad(buffer=buffer, indices=indices, target=name)
    
    def visit_BufferStore(self, target: ast.Name, call: ast.Call) -> BufferStore:
        """Process buffer storing."""
        if len(call.args) != 1:
            raise ValueError("store_buffer requires a single argument")
        
        name = self.visit(call.args[0])
        buffer, indices = self.visit_IndexedBuffer(target)
        buffer.write = True
        return BufferStore(buffer=buffer, indices=indices, value=name)

    def visit_Assign(self, node: ast.Assign) -> Optional[Union[IRNode, ast.Assign]]:
        """Process assignments, handling both buffer and regular assignments."""
        if len(node.targets) != 1:
            return None
            
        target = node.targets[0]
        value = node.value
        
        # Handle special assignments first
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id == "Axis":
                return self.visit_AxisDef(target, value)
            elif value.func.id == "match_buffer":
                return self.visit_BufferMatch(target, value)
            elif value.func.id == "load_buffer":
                return self.visit_BufferLoad(target, value)
            elif value.func.id == "store_buffer":
                return self.visit_BufferStore(target, value)
            elif value.func.id == "temp_buffer":
                value.args = [target] + value.args # add the name to reuse buffermatch code
                buffer_match = self.visit_BufferMatch(target, value)
                buffer_match.tensor_name = None # remove the tensor name
                return buffer_match
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> Tuple[Axis, int]:
        """used to process axis zooming e.g. axis i * 2"""
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(left, Axis) and isinstance(right, int) and isinstance(node.op, ast.Mult):
            return node.op, left, right
        else:
            return self.generic_visit(node)

    def visit_AxisDef(self, target: ast.Name, call: ast.Call) -> AxisDef:
        """Process axis definition."""
        if len(call.args) < 2:
            raise ValueError("Axis requires name and size arguments")
            
        name = self.visit(call.args[0])
        size = self.visit(call.args[1])
        
        # Get optional block_size
        min_block_size = 1
        if len(call.args) > 2:
            min_block_size = self.visit(call.args[2])
        for kw in call.keywords:
            if kw.arg == "min_block_size":
                min_block_size = self.visit(kw.value)
                
        # Create and register axis
        axis = Axis(name=name, size=size, min_block_size=min_block_size)
        self.axes[target.id] = axis
        
        return AxisDef(axis=axis)
    
    def visit_Tuple(self, node) -> Tuple:
        return tuple(self.visit(idx) for idx in node.elts)
    
    def visit_List(self, node) -> List:
        return [self.visit(idx) for idx in node.elts]
    
    def get_type(self, node) -> torch.dtype:
        str = ast.unparse(node)
        ret = eval(str)
        assert isinstance(ret, torch.dtype)
        return ret

    def visit_BufferMatch(self, target: ast.Name, call: ast.Call) -> BufferMatch:
        """Process buffer matching."""
        if len(call.args) < 3:
            raise ValueError("match_buffer requires tensor, shape, and bound_axes arguments")
        
        tensor = self.visit(call.args[0])
        shape = self.visit(call.args[1])
        # can be a tuple or a optional 
        # the content inside an be a axis or a tuple of axis and zooming factor
        bound_axes_in: List[Union[Tuple[Union[Axis, Tuple[ast.Mult, Axis, int]]], Optional[Union[Axis, Tuple[ast.Mult, Axis, int]]]]] = self.visit(call.args[2])
        if len(shape) != len(bound_axes_in):
            raise ValueError("mismatch dimensions in shape and bound_axes")
        bound_axes: List[List[Axis]] = []
        zooming_factors: List[List[int]] = []

        for axis_in in bound_axes_in:
            if isinstance(axis_in, tuple) and not isinstance(axis_in[0], ast.Mult): # use mult to distinguish between tuple of axis and zooming factor
                dim_axes = []
                dim_zooming_factors = []
                for id, axis in enumerate(axis_in):
                    cur_axis = axis
                    zooming_factor = 1
                    if isinstance(axis, tuple):
                        _, cur_axis, zooming_factor = axis
                    dim_axes.append(cur_axis)
                    dim_zooming_factors.append(zooming_factor)
                    if id != len(bound_axes) - 1:
                        cur_axis.max_block_size = 1

                bound_axes.append(dim_axes)
                zooming_factors.append(dim_zooming_factors)
            else:
                if axis_in is not None:
                    cur_axis = axis_in
                    zoom_factor = 1
                    if isinstance(axis_in, tuple):
                        _, cur_axis, zoom_factor = axis_in
                    bound_axes.append([cur_axis])
                    zooming_factors.append([zoom_factor])
                else:
                    bound_axes.append([])
                    zooming_factors.append([])

        # Get optional dtype
        dtype = torch.bfloat16
        # if the dtype is provided, we will use it
        if len(call.args) == 4:
            dtype = self.get_type(call.args[3])
        for kw in call.keywords:
            if kw.arg == "dtype":
                dtype = self.get_type(kw.value)

        cur_axis = None
        if self.cur_loop is not None:
            cur_axis = self.cur_loop.axes[-1]
        # Create buffer and register it
        buffer = Buffer(tensor=tensor, shape=shape, bound_axes=bound_axes, dtype=dtype, axes_factor=zooming_factors, def_axis=cur_axis)
        self.buffers[target.id] = buffer
        
        return BufferMatch(
            buffer=buffer,
            tensor_name=target.id,
        )

    def visit_For(self, node: ast.For) -> GridLoop:
        """Process for loop with grid."""
        if not (isinstance(node.iter, ast.Call) and
                isinstance(node.iter.func, ast.Name) and
                node.iter.func.id == "grid"):
            raise ValueError("Only grid-based for loops are supported")
            
        # Extract grid parameters
        axes = self.visit(node.iter.args[0])  # List of axes
        axis_types = self.visit(node.iter.args[1])  # "ssrs" string
        
        # Verify axis types
        if not all(t in "srm" for t in axis_types):
            raise ValueError("Axis types must be 's' or 'r' or 'm'")
        
        # Process targets
        if isinstance(node.target, ast.Tuple):
            target_names = [t.id for t in node.target.elts]
        else:
            target_names = [node.target.id]
            
        # Verify number of axes matches targets
        if len(target_names) != len(axes):
            raise ValueError(f"Number of loop variables ({len(target_names)}) doesn't match number of axes ({len(axes)})")
        
        # Set up loop variable bindings
        for name, axis in zip(target_names, axes):
            self.loop_vars[name] = axis
        
        # Process loop body
        body = []

        cur_loop = GridLoop(axes=axes, axis_types=axis_types, body=body)

        old_grid_loop = self.cur_loop
        self.cur_loop = cur_loop
        
        for stmt in node.body:
            ir_node = self.visit(stmt)
            if ir_node is not None:
                if isinstance(ir_node, list):
                    body.extend(ir_node)
                else:
                    body.append(ir_node)
                    
        # remove the loop vars
        for name in target_names:
            del self.loop_vars[name]

        self.cur_loop = old_grid_loop
        return cur_loop

    def visit_Reduce(self, node: ast.Call) -> ReduceOp:
        """Process reduce calls."""
        op, src, axis, buffer, collective_op, indices = None, None, None, None, None, None
        for kw in node.keywords:
            if kw.arg == "op":
                op = self.visit(kw.value)
            elif kw.arg == "buffer":
                if isinstance(kw.value, ast.Name):
                    buffer = self.visit_Buffer(kw.value)
                else:
                    buffer, indices = self.visit_IndexedBuffer(kw.value)
            elif kw.arg == "collective_op":
                collective_op = self.visit(kw.value)
            elif kw.arg == "src":
                src = self.visit(kw.value)
            elif kw.arg == "axis":
                axis = self.visit(kw.value)
            else:
                raise ValueError(f"Unknown reduce keyword argument {kw.arg}")
            
        if op is None or buffer is None or src is None or axis is None or collective_op is None:
            raise ValueError("reduce requires collective_op, op, src, axis, and buffer arguments")

        if not isinstance(op, str):
            assert isinstance(op, PyNode), "if op is not str, it should be a pynode"
            op = ast.unparse(op.node)
        
        # if isinstance(op, str):
        #     if op in ["sum", "max", "min"]:
        #         if collective_op is not None:
        #             raise ValueError("collective_op not needed for built-in reduce ops")
        #     else:
        #         if collective_op is None:
        #             raise ValueError("collective_op needed for custom reduce ops")
        # else:
        #     raise ValueError("reduce op must be a func name")

        if isinstance(axis, Axis):
            axis = [axis]
        buffer.read = True
        buffer.write = True
        return ReduceOp(op=op, src=src, axes=axis, buffer=buffer, collective_op=collective_op, indices=indices)

    def visit_Expr(self, node: ast.Expr) -> Optional[IRNode]:
        """Process expression statements."""
        # Check if the expression is a function call
        if isinstance(node.value, ast.Call):
            # Check if the function is a reduce call
            if isinstance(node.value.func, ast.Name) and node.value.func.id == "reduce":
                # Process the reduce call
                return self.visit_Reduce(node.value)
        # If not a reduce call, return generic visit
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Union[Axis, str]:
        """Process name references, resolving axes and preserving regular variables."""
        if node.id in self.axes:
            return self.axes[node.id]
        elif node.id in self.loop_vars:
            return self.loop_vars[node.id]
        return node.id

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Process constants."""
        return node.value

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Program:
        """Process function definition and build Program IR."""
        # Create program node
        self.current_program = Program(
            name=node.name,
            inputs=[],
            outputs=node.returns,
            body=[],
            defaults=[]
        )

        for input in node.args.args:
            self.current_program.inputs.append(input.arg)

        for default in node.args.defaults:
            self.current_program.defaults.append(self.visit(default))
        
        # Process function body
        for stmt in node.body:
            ir_node = self.visit(stmt)
            if ir_node is not None:
                if isinstance(ir_node, list):
                    self.current_program.body.extend(ir_node)
                else:
                    self.current_program.body.append(ir_node)
                    
        return self.current_program

def auto_schedule(extern_funcs: Optional[Dict[str, Any]] = None):
    """Decorator to convert a function to auto-ring IR."""
    def decorator(func):
        # Get function source
        source = inspect.getsource(func)
        source = textwrap.dedent(source)
        
        # Create IR builder
        builder = IRBuilder()
        if extern_funcs:
            builder.extern_funcs = extern_funcs
            
        # Parse function and build IR
        tree = ast.parse(source)
        program = None
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                program = builder.visit(node)
                break
                
        if program is None:
            raise ValueError("Could not find function definition")
            
        # Wrap the original function
        def wrapper(*args, **kwargs):
            # Here we would:
            # 1. Run optimization passes
            # 2. Generate code
            # 3. Execute optimized version
            raise NotImplementedError("Code generation not implemented yet")
            
        return wrapper
    return decorator