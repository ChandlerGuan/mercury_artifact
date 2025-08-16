# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""PyTorch code generator implementation."""

import copy
from typing import Dict, List, Optional, Tuple, Union
from mercury.ir.nodes import (
    IRNode, Program, GridLoop, BufferLoad, BufferStore,
    PyNode, ReduceOp, AxisDef, BufferMatch
)
from mercury.ir.elements import Axis
from mercury.ir.utils import collect_reduce
import textwrap

class PyTorchCodegen:
    """PyTorch code generator."""
    def __init__(self):
        self.indent_level = 0
        self.code_lines: List[str] = []
        self.temp_var_counter = 0
        self.active_axis: List[Axis] = [] # we want to also keep the order of the axis
        self.axis_vars: Dict[str, List] = {}
        self.process_group_name: str = "process_group"
        self.next_prefix: str = "_next_"
        self.ring_idx_suffix: str = "_ring_idx"
        self.ring_comm_prefix: str = "ring_comm_"
        self.mesh_shape: Tuple[int, ...] = ()
        self.active_ring_axes = []
        self.reduce_buffers: Dict[str, ReduceOp] = {}

    def emit_ring_indice(self, axis: Axis, mesh_dim: int):
        self.emit(f"send_to = shift_tuple_element(indices, {mesh_dim}, 1, {axis.ring_comm_cards})")
        self.emit(f"send_to = n_dim_to_one_dim(send_to, {self.mesh_shape})")
        self.emit(f"recv_from = shift_tuple_element(indices, {mesh_dim}, -1, {axis.ring_comm_cards})")
        self.emit(f"recv_from = n_dim_to_one_dim(recv_from, {self.mesh_shape})")
    
    def emit(self, line: str) -> None:
        """Emit a line of code with proper indentation."""
        self.code_lines.append(textwrap.indent(line, "    " * self.indent_level))
        # self.code_lines.append("    " * self.indent_level + line)
    
    def get_temp_var(self) -> str:
        """Get a new temporary variable name."""
        self.temp_var_counter += 1
        return f"_tmp{self.temp_var_counter}"

    def get_inner_active_axis(self, ring_axes: List[Axis]) -> List[Axis]:
        """ Get the inner active axis from a list of axes. """
        inner_active_axis = []
        ring_axes_names = [axis.name + self.ring_idx_suffix for axis in ring_axes]
        for axis in self.active_axis:
            if len(ring_axes_names) == 0:
                inner_active_axis.append(axis)
            elif axis.name in ring_axes_names:
                ring_axes_names.remove(axis.name)
        return inner_active_axis

    def visit(self, node: IRNode) -> Optional[str]:
        """Visit an IR node and generate code.
        
        Returns:
            Optional string representing the computed value/variable name.
        """
        method = f'visit_{node.__class__.__name__}'
        if hasattr(self, method):
            return getattr(self, method)(node)
        return str(node)
    
    def visit_tuple(self, node: Tuple) -> str:
        """Generate code for a Tuple node."""
        return f"({', '.join(self.visit(item) for item in node)})"

    def visit_AxisDef(self, node: AxisDef) -> None:
        """Process axis definition."""
        # Store axis size for later use
        return None

    def visit_ReduceOp(self, node: ReduceOp) -> None:
        """Generate code for a ReduceOP node."""
        self.reduce_buffers[node.buffer.tensor] = node

        # Get reduce target
        tensor_name = node.buffer.tensor
        indice_str = ""
        if node.indices is not None:
            indices = self.gen_indice(node)
            indice_str = f"[{', '.join(indices)}]"
        if tensor_name is None:
            raise ValueError(f"No mapping found for buffer {node.buffer.tensor}")
        
        # Get reduce src
        src = self.visit(node.src)

        # if first round:
        #     target = src
        # else:
        #     target = op(target, src)

        if len(node.comm) == 0:
            var_cond = []
            for axis in node.axes:
                if axis.name in self.axis_vars:
                    for var in self.axis_vars[axis.name]:
                        var_cond.append(f"{var} == 0")
            if len(var_cond) == 0:
                self.emit(f"{tensor_name}{indice_str} = {src}")
            else:
                self.emit(f"if {' and '.join(var_cond)}:")
                self.indent_level += 1
                self.emit(f"{tensor_name}{indice_str} = {src}")
                self.indent_level -= 1
                self.emit(f"else:")
                self.indent_level += 1
                self.emit(f"{tensor_name}{indice_str} = {node.op}({tensor_name}{indice_str}, {src})")
                self.indent_level -= 1
        else:
            axes:List[Axis] = [comm.axis for comm in node.comm]
            mesh_dims: List[int] = [comm.shard_dim for comm in node.comm]
            indice_names = [axis.name + self.ring_idx_suffix for axis in axes]
            comm_name = self.ring_comm_prefix + node.buffer.tensor
            all_indices_zero = " and ".join([f"{indice_name} == 0" for indice_name in indice_names])

            # if inner axis == 0:
            #     if id1 == 0 and id2 == 0 and ...:
            #         target = expr
            #         comm = SendRecv()
            #     else:
            #         comm.wait()
            #         target = reduce(_next_target, expr)
            # else:
            #     local_reduce
            inner_active_axis = self.get_inner_active_axis(axes)
            inner_axis_all_zero = " and ".join([f"{axis.name} == 0" for axis in inner_active_axis])
            if len(inner_active_axis) > 0:
                self.emit(f"if {inner_axis_all_zero}:")
                self.indent_level += 1
            self.emit(f"if {all_indices_zero}:")
            self.indent_level += 1
            self.emit(f"{tensor_name}{indice_str} = {src}.contiguous()")
            self.emit(f"{comm_name} = SendRecv()")
            self.indent_level -= 1
            self.emit(f"else:")
            self.indent_level += 1
            self.emit(f"{comm_name}.wait()")
            self.emit(f"{tensor_name}{indice_str} = {node.op}({self.next_prefix}{tensor_name}, {src})")
            self.indent_level -= 1
            if len(inner_active_axis) > 0:
                self.indent_level -= 1
                self.emit(f"else:")
                self.indent_level += 1
                self.emit(f"{tensor_name}{indice_str} = {node.op}({tensor_name}{indice_str}, {src})")
                self.indent_level -= 1

            # if inner_axis is last
            #     if id_inner + 1 != inner_size:
            #         _next_target = ring in inner axis
            #         comm.commit()
            #     elif id_outer + 1 != outer_size:
            #         _next_target = ring in outer axis
            #         comm.commit()

            inner_axis_all_last = " and ".join([f"{axis.name} + {axis.min_block_size} == {axis.size // axis.ring_comm_cards}" for axis in inner_active_axis])
            if len(inner_active_axis) > 0:
                self.emit(f"if {inner_axis_all_last}:")
                self.indent_level += 1
            for id, (axis, mesh_dim) in enumerate(zip(axes, mesh_dims)):
                if id == 0:
                    self.emit(f"if {indice_names[id]} + 1 != {axis.ring_comm_cards}:")
                else:
                    self.emit(f"elif {indice_names[id]} + 1 != {axis.ring_comm_cards}:")

                self.indent_level += 1

                self.emit_ring_indice(axis, mesh_dim)
                self.emit(f"{self.next_prefix}{tensor_name} = {comm_name}.send_recv(send_to, recv_from, {tensor_name}{indice_str})")
                self.emit(f"{comm_name}.commit()")
                self.indent_level -= 1
                last_axis, last_mesh_dim = axis, mesh_dim

            # transfer back to the corresponding rank
            self.emit("else:")
            self.indent_level += 1
            self.emit(f"recv_from, send_to = get_src_dst_ranks(indices, {self.mesh_shape}, {mesh_dims})")
            self.emit(f"{self.next_prefix}{tensor_name} = {comm_name}.send_recv(send_to, recv_from, {tensor_name}{indice_str})")
            self.emit(f"{comm_name}.commit()")
            self.emit(f"{comm_name}.wait()")
            self.emit(f"{tensor_name}{indice_str} = {self.next_prefix}{tensor_name}")
            self.indent_level -= 1
            if len(inner_active_axis) > 0:
                self.indent_level -= 1
            
            
    def visit_BufferMatch(self, node: BufferMatch) -> None:
        """Process buffer matching."""
        
        if isinstance(node.tensor_name, str) and isinstance(node.buffer.tensor, str):
            # Generate shape assertion
            shape_str = str(tuple(s for s in node.buffer.shape))
            self.emit(f"assert {node.buffer.tensor}.shape == {shape_str}, 'Shape mismatch for {node.buffer.tensor}'")
        elif node.tensor_name is None and isinstance(node.buffer.tensor, str):
            # Generate temporary buffer
            if len(self.active_ring_axes) > 0:
                # when in a ring, only the first step need to load the data
                all_indices_zero = " and ".join([f"{axis.name}{self.ring_idx_suffix} == 0" for axis in self.active_ring_axes])
                self.emit(f"if {all_indices_zero}:")
                self.indent_level += 1
                self.emit(f"{node.buffer.tensor} = torch.empty({tuple(node.buffer.get_shape())}, dtype={node.buffer.dtype}, device=device)")
                self.indent_level -= 1
            else:
                self.emit(f"{node.buffer.tensor} = torch.empty({tuple(node.buffer.get_shape())}, dtype={node.buffer.dtype}, device=device)")

    def visit_Program(self, node: Program) -> None:
        """Generate code for a Program node."""
        # Import os
        self.emit("import os")
        # Import torch
        self.emit("import torch")
        self.emit("import torch.distributed as dist")
        self.emit("from mercury.backend import *")
        self.emit("")

        self.emit(f"def {node.name}(")

        self.indent_level += 1

        non_default_num = len(node.inputs) - len(node.defaults)
        
        # Function signature with proper commas
        for id, inp in enumerate(node.inputs):
            if id < non_default_num:
                var = f"{inp}"
            else:
                default = node.defaults[id - non_default_num]
                default_str = self.visit(default)
                var = f"{inp}={default_str}"

            if id < len(node.inputs) - 1:
                var += ","
            self.emit(var)
                        
        self.indent_level -= 1
        self.emit(f"):")
        
        # Function body
        self.indent_level += 1

        if node.mesh is not None:
            # get the indices of the machine in the mesh
            self.mesh_shape = node.mesh.shape
            self.emit(f"indices = one_dim_to_n_dim(dist.get_rank(), {self.mesh_shape})")
            self.emit("local_rank = int(os.environ['LOCAL_RANK'])")
            self.emit("""device = torch.device(f"cuda:{local_rank}")""")
        else:
            self.emit("""device = torch.device("cuda")""")

        # Process all nodes
        for child in node.body:
            result = self.visit(child)
            if isinstance(result, str):
                self.emit(result)
                
        self.indent_level -= 1

    def visit_GridLoop(self, node: GridLoop) -> None:
        """Generate code for a GridLoop node."""
        indent_add = []
        old_active_ring_axes = self.active_ring_axes
        self.active_ring_axes = copy.deepcopy(self.active_ring_axes)

        ring_axes = []
        # Generate nested loops
        for axis in node.axes:
            var = axis.name
            begin = 0
            end = axis.size
            stride = axis.min_block_size

            if end == stride:
                indent_add.append(0)
                continue

            # for comm_name in axis.ring_comm:
            #     self.emit(f"{self.ring_comm_prefix}{axis.name}{comm_name} = RingComm({self.process_group_name})")

            if len(axis.ring_comm) == 0:
                self.emit(f"for {var} in range({begin}, {end}, {stride}):")
                self.indent_level += 1
                indent_add.append(1)
                self.active_axis.append(axis)
                self.axis_vars[axis.name] = [var]
            else:
                # if there is a ring comm, we need to split the loop into two:
                # 1. the first loop is the loop over local data
                # 2. the second loop is the loop over ringed data
                # example:
                # for i in range(0, size//num_cards, min_block_size):
                #     for j in range(0, num_cards, 1):

                axis_var = []
                if begin + stride < end // axis.ring_comm_cards:
                    self.emit(f"for {var} in range({begin}, {end // axis.ring_comm_cards}, {stride}):")
                    self.indent_level += 1
                    self.active_axis.append(axis)
                    axis_var.append(var)

                self.emit(f"for {var}{self.ring_idx_suffix} in range({axis.ring_comm_cards}):")
                ring_axis = Axis(axis.name + self.ring_idx_suffix, axis.ring_comm_cards, 1)
                ring_axes.append(ring_axis)
                self.active_axis.append(ring_axis)
                self.indent_level += 1
                axis_var.append(f"{var}{self.ring_idx_suffix}")
                self.axis_vars[axis.name] = axis_var
                indent_add.append(len(axis_var))
                self.active_ring_axes.append(axis)
        
        # Generate loop body
        for child in node.body:
            result = self.visit(child)
            if isinstance(result, str):
                self.emit(result)
        
        # collect reduce ops
        reduce_ops = node.visit(collect_reduce)
        axis2reduce = {}
        for reduce_op in reduce_ops:
            for axis in reduce_op.axes:
                if axis.name not in axis2reduce:
                    axis2reduce[axis.name] = []
                axis2reduce[axis.name].append(reduce_op)

        # End loops
        # remove ring axes
        for axis in ring_axes:
            if axis in self.active_axis:
                self.active_axis.remove(axis)
        for axis, indent in zip(reversed(node.axes), reversed(indent_add)):
            end = axis.size
            stride = axis.min_block_size
            if end != stride:
                self.indent_level -= indent
                if axis.name in self.active_axis:
                    self.active_axis.remove(axis)
                    self.axis_vars[axis.name] = []
            # this is a ad-hoc solution, now we create the collective when the buffer is loaded
            # if axis.name in axis2reduce:
            #     reduce_ops = axis2reduce[axis.name]
            #     met_collective = False
            #     for reduce_op in reduce_ops:
            #         if reduce_op.shard_dim is not None and len(reduce_op.comm) == 0: # if sharded and not ringed
            #             if met_collective:
            #                 raise ValueError("Only one collective reduce is supported for now")
            #             met_collective = True
            #             self.emit(f"{reduce_op.collective_op}({reduce_op.buffer.tensor}, dist.all_reduce, group=get_device_group(indices, {self.mesh_shape}, {reduce_op.shard_dim}))")

        self.active_ring_axes = old_active_ring_axes

    def gen_indice(self, node: Union[BufferLoad, BufferStore]) -> List:
        # Handle indices
        indices = []

        for axis_list, zoom_list in zip(node.buffer.bound_axes, node.buffer.axes_factor):
            related_axis = []
            axis_stride = []
            zoom_axis = []
            for axis, zoom_factor in zip(axis_list, zoom_list):
                for i in range(len(axis_stride)):
                    axis_stride[i] *= (axis.size // axis.ring_comm_cards)
                if axis in node.indices and axis in self.active_axis:
                    related_axis.append(axis)
                    axis_stride.append(1)
                    zoom_axis.append(zoom_factor)

            indice = ""
            for id, axis in enumerate(related_axis):
                if indice != "":
                    indice += " + " 

                axis_name = axis.name
                zoom_factor = zoom_axis[id]
                if zoom_factor != 1:
                    axis_name += f" * {zoom_factor}"
                if axis_stride[id] != 1:
                    indice += f"{axis_name} * {axis_stride[id]}"
                else:
                    indice += f"{axis_name}"
                if id == len(related_axis) - 1:
                    indice = f"{indice} : {indice} + {axis.min_block_size * axis_stride[id] * zoom_factor}"
                else:
                    assert axis.min_block_size == 1, "block size > 1 is not supported for outer axis in one dimension"
            if len(related_axis) == 0:
                indices.append(":")
            else:
                indices.append(indice)

        return indices

    def visit_BufferLoad(self, node: BufferLoad) -> None:
        """Generate code for a BufferLoad node."""

        indices = self.gen_indice(node)
        # Get input tensor name from buffer map
        tensor_name = node.buffer.tensor
        
        # Generate indexing expression
        index_str = f"[{', '.join(indices)}]"

        expr = f"{tensor_name}{index_str}"
        target = node.target


        if len(node.comm) == 0:
            # if len(self.active_ring_axes) > 0:
            #     # when in a ring, only the first step need to load the data
            #     all_indices_zero = " and ".join([f"{axis.name}{self.ring_idx_suffix} == 0" for axis in self.active_ring_axes])
            #     self.emit(f"if {all_indices_zero}:")
            #     self.indent_level += 1
            #     self.emit(f"{target} = {expr}")
            #     self.indent_level -= 1
            # else:

            # tackle the reduce buffer
            if node.buffer.tensor in self.reduce_buffers:
                reduce_op = self.reduce_buffers[node.buffer.tensor]
                if reduce_op.buffer.tensor != node.buffer.tensor:
                    raise ValueError("reduce buffer tensor name mismatch")

                if len(reduce_op.shard_dim) > 0: # is sharded
                    shard_dims = set(reduce_op.shard_dim)
                    for comm in reduce_op.comm:
                        shard_dims.remove(comm.shard_dim)
                    if len(shard_dims) > 0: # if not fully ringed
                        if len(self.active_ring_axes) > 0:
                            # when in a ring, only the last step need to store the data
                            all_indices_max = " and ".join([f"{axis.name}{self.ring_idx_suffix} == {axis.ring_comm_cards - 1}" for axis in self.active_ring_axes])
                            self.emit(f"if {all_indices_max}:")
                            self.indent_level += 1
                            self.emit(f"{reduce_op.collective_op}({reduce_op.buffer.tensor}, dist.all_reduce, group=get_device_group(indices, {self.mesh_shape}, {tuple(shard_dims)}))")
                            self.indent_level -= 1
                        else:
                            self.emit(f"{reduce_op.collective_op}({reduce_op.buffer.tensor}, dist.all_reduce, group=get_device_group(indices, {self.mesh_shape}, {tuple(shard_dims)}))")

            self.emit(f"{target} = {expr}")
        else:
            assert node.buffer.write == False, "both read and write is only supported for reduction"
            axes:List[Axis] = [comm.axis for comm in node.comm]
            mesh_dims: List[int] = [comm.shard_dim for comm in node.comm]
            indice_names = [axis.name + self.ring_idx_suffix for axis in axes]
            comm_name = self.ring_comm_prefix + node.buffer.tensor
            all_indices_zero = " and ".join([f"{indice_name} == 0" for indice_name in indice_names])

            # if id1 == 0 and id2 == 0 and ...:
            #     target = expr
            #     comm = SendRecv()
            # else:
            #     comm.wait_last_round()
            #     target = _next_target
            inner_active_axis = self.get_inner_active_axis(axes)
            inner_axis_all_zero = " and ".join([f"{axis.name} == 0" for axis in inner_active_axis])
            if len(inner_active_axis) > 0:
                self.emit(f"if {inner_axis_all_zero}:")
                self.indent_level += 1
            self.emit(f"if {all_indices_zero}:")
            self.indent_level += 1
            self.emit(f"{target} = {expr}.contiguous()")
            self.emit(f"{comm_name} = SendRecv()")
            self.indent_level -= 1
            self.emit(f"else:")
            self.indent_level += 1
            self.emit(f"{comm_name}.wait()")
            self.emit(f"{target} = {self.next_prefix}{target}")
            self.indent_level -= 1

            # if id_inner + 1 != inner_size:
            #     _next_target = ring in inner axis
            #     comm.commit()
            # elif id_outer + 1 != outer_size:
            #     _next_target = ring in outer axis
            #     comm.commit()

            for id, (axis, mesh_dim) in enumerate(zip(axes, mesh_dims)):
                if id == 0:
                    self.emit(f"if {indice_names[id]} + 1 != {axis.ring_comm_cards}:")
                else:
                    self.emit(f"elif {indice_names[id]} + 1 != {axis.ring_comm_cards}:")
                self.indent_level += 1
                self.emit_ring_indice(axis, mesh_dim)


                self.emit(f"{self.next_prefix}{target} = {comm_name}.send_recv(send_to, recv_from, {target})")
                self.emit(f"{comm_name}.commit()")
                self.indent_level -= 1

            if len(inner_active_axis) > 0:
                self.indent_level -= 1
            
    def visit_BufferStore(self, node: BufferStore) -> None:
        """Generate code for a BufferStore node."""
        indices = self.gen_indice(node)
        
        # Get input tensor name from buffer map
        tensor_name = node.buffer.tensor
        if tensor_name is None:
            raise ValueError(f"No mapping found for buffer {node.buffer.tensor}")
        
        # Generate indexing expression
        index_str = f"[{', '.join(indices)}]"
        
        # Handle value
        if isinstance(node.value, str):
            value = node.value
        else:
            value = self.visit(node.value)
            
        if len(node.comm) == 0:
            if len(self.active_ring_axes ) > 0:
                # when in a ring, only the last step need to store the data
                all_indices_max = " and ".join([f"{axis.name}{self.ring_idx_suffix} == {axis.ring_comm_cards - 1}" for axis in self.active_ring_axes])
                self.emit(f"if {all_indices_max}:")
                self.indent_level += 1
                self.emit(f"{tensor_name}{index_str} = {value}")
                self.indent_level -= 1
            else:
                self.emit(f"{tensor_name}{index_str} = {value}")
        else:
            raise ValueError("Communication not supported for store operations")

    def visit_PyNode(self, node: PyNode) -> str:
        """Generate code for a PyNode."""
        # For now, just convert AST node to string
        import ast
        return ast.unparse(node.node)

    # def visit_BinaryOp(self, node: BinaryOp) -> str:
    #     """Generate code for a BinaryOp node."""
    #     # Handle operands
    #     left = self.visit(node.left) if isinstance(node.left, (IRNode, PyNode)) else str(node.left)
    #     right = self.visit(node.right) if isinstance(node.right, (IRNode, PyNode)) else str(node.right)
        
    #     # Generate operation
    #     if node.op == "@":  # Matrix multiplication
    #         return f"torch.matmul({left}, {right})"
    #     elif node.op == "+=":  # Accumulation
    #         return f"{left} = {left} + {right}"
    #     return f"{left} {node.op} {right}"

def generate_pytorch_code(program: Program) -> str:
    """Generate PyTorch code from IR program.
    
    Args:
        program: The IR program to convert.
        
    Returns:
        Generated PyTorch code as string.
    """
    codegen = PyTorchCodegen()
    codegen.visit(program)
    return "\n".join(codegen.code_lines)