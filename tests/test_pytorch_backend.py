# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""Tests for PyTorch backend code generation."""

import torch
import pytest
import ast
import inspect
import textwrap
from mercury.ir.elements import Axis, grid, match_buffer, store_buffer, load_buffer
from mercury.ir import eliminate_loops
from mercury.frontend.parser import IRBuilder, auto_schedule
from mercury.backend.pytorch import generate_pytorch_code
from utils.flash_attn_dsl import *
from flash_attn.flash_attn_interface import _flash_attn_forward, flash_attn_func

def test_simple_matmul_codegen():
    """Test PyTorch code generation for matrix multiplication."""
    
    def simple_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
        # Define axes
        I = Axis("I", 128)  # M dimension
        J = Axis("J", 256)  # K dimension
        K = Axis("K", 128)  # N dimension

        # Match buffers
        A = match_buffer(a, (128, 256), [I, J])  # [M, K]
        B = match_buffer(b, (256, 128), [J, K])  # [K, N]
        C = match_buffer(c, (128, 128), [I, K])  # [M, N]

        # Grid pattern shows J is reduction axis
        for i, j, k in grid([I, J, K], "srs"):
            _c = load_buffer(C[i, k])
            _a = load_buffer(A[i, j])
            _b = load_buffer(B[j, k])
            _c += torch.matmul(_a, _b)
            C[i, k] = store_buffer(_c)

    # Get IR from function
    source = inspect.getsource(simple_matmul)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    # Parse function definition to IR
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        pytest.fail("Could not find function definition")

    print(program)
    eliminate_loops(program)
    print("After loop elimination:")
    print(program)

    # Generate PyTorch code
    code = generate_pytorch_code(program)
    print("\nGenerated PyTorch code:")
    print(code)

    # Execute generated code
    namespace = {}
    exec(code, namespace)
    matmul_func = namespace["simple_matmul"]

    # Test with sample inputs
    a = torch.randn(128, 256, device="cuda")
    b = torch.randn(256, 128, device="cuda")
    c = torch.zeros(128, 128, device="cuda")

    # Run generated code
    matmul_func(a, b, c)

    # Verify result
    expected = torch.matmul(a, b)
    assert torch.allclose(c, expected, rtol=1e-3, atol=1e-3), \
        "Generated matmul result doesn't match torch.matmul"

    print("✓ PyTorch code generation test passed")

def test_flash_attention_codegen():
    """Test PyTorch code generation for Flash Attention."""
    from flash_attn.flash_attn_interface import _flash_attn_forward
    
    # Get IR from flash_attn function
    source = inspect.getsource(flash_attn)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    # Parse function definition to IR
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        pytest.fail("Could not find function definition")

    print(program)

    eliminate_loops(program)

    print("After loop elimination:")
    print(program)

    # Generate PyTorch code
    code = generate_pytorch_code(program)
    print("\nGenerated PyTorch code:")
    print(code)

    # Execute generated code
    namespace = globals()
    
    exec(code, namespace)
    flash_attn_gen = namespace["flash_attn"]

    # Test with sample inputs
    batch_size = 4
    seq_len = 4096
    num_heads = 5
    head_dim = 128
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    o = torch.zeros(batch_size, seq_len, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    lse = torch.zeros(batch_size, num_heads, seq_len, device="cuda", dtype=torch.float32)

    dropout_p = 0
    causal = False
    deterministic = False

    # Run generated code
    flash_attn_gen(q, k, v, o, lse, softmax_scale=q.shape[-1] ** (-0.5))

    # Compare with reference implementation
    ref_out, ref_lse, _ = flash_attn_func(
        q,k,v,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    
    print("out diff")
    print(f"mean: {torch.mean((o - ref_out).abs())}")
    print(f"max: {torch.max((o - ref_out).abs())}")
    print("lse diff")
    print(f"mean: {torch.mean((lse - ref_lse).abs())}")
    print(f"max: {torch.max((lse - ref_lse).abs())}")

    assert torch.allclose(o, ref_out, rtol=1e-3, atol=1e-3), \
        "Generated flash attention output doesn't match reference"
    assert torch.allclose(lse, ref_lse, rtol=1e-3, atol=1e-3), \
        "Generated flash attention LSE doesn't match reference"

    print("✓ PyTorch flash attention code generation test passed")

if __name__ == "__main__":
    test_simple_matmul_codegen()
    test_flash_attention_codegen()