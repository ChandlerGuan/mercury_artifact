# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

from typing import Optional
import pytest
import torch
from mercury.ir.elements import Axis, Buffer, grid, match_buffer, load_buffer, store_buffer
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh
from mercury.ir.init_distributed import init_distributed
from mercury.ir.nodes import (
    IRNode, Program, AxisDef, BufferMatch, GridLoop,
)
from mercury.ir.calculate_memory import get_buffer_size
import mercury.ir.primitives as sp
import mercury.ir.loop_eliminating as le
import ast
import textwrap
import inspect
import utils.flash_attn_dsl as flash_attn_dsl

def test_ir_parallize():

    # Create test inputs
    batch_size, seqlen, nheads, dim = 4, 4096, 5, 128
    q = torch.randn(batch_size, seqlen, nheads, dim)
    k = torch.randn(batch_size, seqlen, nheads, dim)
    v = torch.randn(batch_size, seqlen, nheads, dim)
    o = torch.zeros_like(q)
    lse = torch.zeros(batch_size, nheads, seqlen)

    # Get source and parse to IR
    source = inspect.getsource(flash_attn_dsl.flash_attn)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    # Parse function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        pytest.fail("Could not find function definition")

    print(program)

    # collect axes
    def collect_axes(node: IRNode) -> Optional[AxisDef]:
        return node if isinstance(node, AxisDef) else None
    
    axes = program.visit(collect_axes)

    axis = axes[2] # S_q

    # we want to get the loop
    def collect_loops(node: IRNode) -> Optional[GridLoop]:
        return node if isinstance(node, GridLoop) else None
    
    loops = program.visit(collect_loops)
    assert len(loops) == 2, "Should have 2 grid loops"
    
    # Outer loop should be spatial
    outer_loop = next(l for l in loops if len(l.axes) == 3)

    devices = [0, 1, 2, 3, 4, 5, 6, 7]
    mesh = DeviceMesh(devices, (8,))
    init_distributed(program, mesh)

    sp.parallelize(program, outer_loop, axis.axis, mesh, 0, 1)

    sp.shift(program, axis.axis, mesh, 0, 1, 1)

    print(program)

    le.eliminate_loops(program)

    print("After loop elimination:")
    print(program)
    print(get_buffer_size(program))

if __name__ == '__main__':
    test_ir_parallize()