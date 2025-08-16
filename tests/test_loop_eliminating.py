# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

from typing import List, Optional
import pytest
import torch
from mercury.ir.elements import Axis, Buffer, grid, match_buffer, load_buffer, store_buffer
from mercury.frontend.parser import IRBuilder, auto_schedule
from mercury.ir.nodes import (
    IRNode, Program, AxisDef, BufferMatch, GridLoop,
)
from mercury.ir import eliminate_loops
import ast
import textwrap
import inspect
import utils.flash_attn_dsl as flash_attn_dsl

def test_flash_attn_loop_eliminate():
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

    # 执行loop elimination
    original_loops: List[GridLoop] = []
    def collect_loops(node: IRNode):
        if isinstance(node, GridLoop):
            original_loops.append(node)
    program.visit(collect_loops)
    
    original_optional_count = sum(
        sum(1 for axis in loop.axes if axis.max_block_size == axis.size)
        for loop in original_loops
    )
    
    eliminate_loops(program)
    
    remaining_loops = []
    def collect_remaining_loops(node: IRNode):
        if isinstance(node, GridLoop):
            remaining_loops.append(node)
    program.visit(collect_remaining_loops)
            
    print("Original Program:")
    print(program)

def test_simple_loop_eliminate():
    optional_axis = Axis("opt", size=8, max_block_size=8)
    required_axis = Axis("req", size=4, max_block_size=2)
    
    loop = GridLoop(
        axes=[optional_axis, required_axis],
        axis_types="sr",
        body=[]
    )
    
    program = Program(
        name="simple_test",
        inputs=[],
        defaults=[],
        outputs=None,
        body=[loop]
    )
    
    eliminate_loops(program)

if __name__ == "__main__":
    test_flash_attn_loop_eliminate()
    test_simple_loop_eliminate()