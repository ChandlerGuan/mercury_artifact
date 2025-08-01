from typing import Optional
import pytest
import torch
from mercury.ir.init_distributed import init_distributed
from mercury.frontend.parser import IRBuilder, auto_schedule
from mercury.ir.calculate_memory import get_buffer_size
from mercury.ir.distributed import DeviceMesh
from mercury.ir.nodes import (
    IRNode, Program, AxisDef, BufferMatch, GridLoop,
)
import ast
import textwrap
import inspect
import utils.flash_attn_dsl as flash_attn_dsl

def test_flash_attn_ir_gen():

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

    devices = [0, 1, 2, 3, 4, 5, 6, 7]
    mesh = DeviceMesh(devices, (8,))
    init_distributed(program, mesh)

    print(program)

    # Test 1: Verify axis definitions
    def collect_axes(node: IRNode) -> Optional[AxisDef]:
        return node if isinstance(node, AxisDef) else None
    
    axes = program.visit(collect_axes)
    assert len(axes) == 4, "Should have 4 axis definitions"
    axis_names = {ax.axis.name for ax in axes}
    assert axis_names == {"B", "H", "S_q", "S_kv"}

    # Test 2: Verify buffer matching
    def collect_buffers(node: IRNode) -> Optional[BufferMatch]:
        return node if isinstance(node, BufferMatch) else None
    
    buffers = program.visit(collect_buffers)
    assert len(buffers) == 5, "Should have 5 buffer matches"
    buffer_names = {b.tensor_name for b in buffers}
    assert buffer_names == {"Q", "K", "V", "O", "LSE"}

    # Test 3: Verify grid loop structure
    def collect_loops(node: IRNode) -> Optional[GridLoop]:
        return node if isinstance(node, GridLoop) else None
    
    loops = program.visit(collect_loops)
    assert len(loops) == 2, "Should have 2 grid loops"
    
    # Outer loop should be spatial
    outer_loop = next(l for l in loops if len(l.axes) == 3)
    assert outer_loop.axis_types == "sss", "Outer loop should be spatial"
    
    # Inner loop should be reduction
    inner_loop = next(l for l in loops if len(l.axes) == 1)
    assert inner_loop.axis_types == "r", "Inner loop should be reduction"

    # Test: Calculate buffer size
    buffer_size = get_buffer_size(program)
    assert buffer_size == 84213760, "Buffer size should be 84213760"
if __name__ == "__main__":
    test_flash_attn_ir_gen()