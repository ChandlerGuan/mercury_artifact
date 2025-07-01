"""
Tests for IR generation and manipulation.
"""
import torch
import pytest
import ast
import inspect
import textwrap
from typing import Any

from mercury.ir.elements import Axis, Buffer, grid, match_buffer, store_buffer, load_buffer
from mercury.frontend.parser import IRBuilder, auto_schedule
from mercury.ir.nodes import (
    Program, AxisDef, BufferMatch, GridLoop,
)

def test_matmul_ir_gen():
    """Test IR generation for matrix multiplication."""

    def simple_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        # Define axes
        I = Axis("I", 128, min_block_size=32)  # M dimension
        J = Axis("J", 256, min_block_size=32)  # K dimension
        K = Axis("K", 128, min_block_size=32)  # N dimension

        # Match buffers
        A = match_buffer(a, [128, 256], [I, J])  # [M, K]
        B = match_buffer(b, [256, 128], [J, K])  # [K, N]
        C = match_buffer(c, [128, 128], [I, K])  # [M, N]

        # Grid pattern shows J is reduction axis
        for i, j, k in grid([I, J, K], "srs"):
            _c = load_buffer(C[i, k])
            _a = load_buffer(A[i, j])
            _b = load_buffer(B[j, k])
            _c += _a @ _b
            C[i, k] = store_buffer(_c)

    # Create test inputs
    a = torch.randn(128, 256)
    b = torch.randn(256, 128)
    c = torch.zeros(128, 128)

    # Get IR from function
    source = inspect.getsource(simple_matmul)
    source = textwrap.dedent(source)

    print(source)
    
    # Parse the function and build IR
    tree = ast.parse(source)
    builder = IRBuilder()

    # Find and parse the function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        pytest.fail("Could not find function definition")

    # print IR structure
    print(program)

    # Verify IR structure
    assert isinstance(program, Program)
    
    # Count node types
    node_types = {}
    for node in program.body:
        node_type = type(node).__name__
        node_types[node_type] = node_types.get(node_type, 0) + 1

    # Verify axis definitions
    assert node_types.get('AxisDef', 0) == 3, "Should have 3 axis definitions"
    axis_defs = [n for n in program.body if isinstance(n, AxisDef)]
    assert {ax.axis.name for ax in axis_defs} == {"I", "J", "K"}

    # Verify buffer matching
    assert node_types.get('BufferMatch', 0) == 3, "Should have 3 buffer matches"
    buffer_matches = [n for n in program.body if isinstance(n, BufferMatch)]
    assert {m.tensor_name for m in buffer_matches} == {"A", "B", "C"}

    # Verify grid loop
    grid_loops = [n for n in program.body if isinstance(n, GridLoop)]
    assert len(grid_loops) == 1, "Should have 1 grid loop"
    grid_loop = grid_loops[0]
    assert grid_loop.axis_types == "srs", "Grid should have srs pattern"

    print("✓ Matmul IR generation test passed")

def test_ir_errors():
    """Test error handling in IR generation."""
    def invalid_grid_func(a: torch.Tensor, b: torch.Tensor):
        # Test invalid grid pattern
        I = Axis("I", 10)
        for i in grid([I], "x"):  # Invalid axis type
            pass

    def invalid_buffer_func(a: torch.Tensor):
        # Test mismatched buffer dimensions
        I = Axis("I", 10)
        A = match_buffer(a, [10,], [I, I])  # Too many axes

    # Test invalid grid pattern
    with pytest.raises(ValueError, match="Axis types must be 's' or 'r'"):
        source = inspect.getsource(invalid_grid_func)
        tree = ast.parse(textwrap.dedent(source))
        builder = IRBuilder()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                builder.visit(node)

    # Test mismatched buffer dimensions
    with pytest.raises(ValueError, match="dimensions"):
        source = inspect.getsource(invalid_buffer_func)
        tree = ast.parse(textwrap.dedent(source))
        builder = IRBuilder()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                builder.visit(node)

if __name__ == "__main__":
    test_matmul_ir_gen()
    test_ir_errors()
    print("\nAll tests passed! ✨")