from typing import Optional
import pytest
from mercury.backend.pytorch.codegen import generate_pytorch_code
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.frontend.parser import IRBuilder, auto_schedule
import ast
import textwrap
from flash_attn.flash_attn_interface import _flash_attn_forward
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
from mercury.ir.distributed import DeviceMesh
from mercury.search.search import search, enumerate_mesh_shapes, enumerate_mesh_assignment
from mercury.search.dump import dump
from utils.utils import log
import torch.distributed as dist
from utils.flash_attn_dsl import *
from utils.gemm_dsl import *

def test_enumerate_mesh_shapes():
    shapes4 = list(enumerate_mesh_shapes(4, 3))
    expected4 = [(4,), (2, 2)]
    assert set(shapes4) == set(expected4), f"Expected {expected4}, got {shapes4}"

    shapes8 = list(enumerate_mesh_shapes(8, 3))
    expected8 = [(8,), (2, 4), (4, 2), (2, 2, 2)]
    assert set(shapes8) == set(expected8), f"Expected {expected8}, got {shapes8}"

    shapes7 = list(enumerate_mesh_shapes(7, 3))
    expected7 = [(7,)]
    assert set(shapes7) == set(expected7), f"Expected {expected7}, got {shapes7}"

    shapes16 = list(enumerate_mesh_shapes(16, 2))
    for shape in shapes16:
        assert len(shape) <= 2, f"Shape {shape} exceeds max dimension 2"

    for shape in shapes16:
        product = 1
        for dim in shape:
            product *= dim
        assert product == 16, f"Shape {shape} product does not equal 16"

def test_enumerate_mesh_assignment():
    # test 1: Basic case - 2D mesh with 2 axes
    assignments = list(enumerate_mesh_assignment(2, 2))
    expected = [
        [(0, 2), (0, 0)],  
        [(0, 0), (0, 2)],  
        [(0, 1), (1, 1)],  
        [(1, 1), (0, 1)]   
    ]
    assert set(map(tuple, assignments)) == set(map(tuple, expected)), \
        f"Expected {expected}, got {assignments}"

    assignments = list(enumerate_mesh_assignment(3, 2))
    for assignment in assignments:
        assert all(isinstance(a, tuple) and len(a) == 2 for a in assignment)
        dims_used = []
        for start, length in assignment:
            if length > 0:
                dims_used.extend(range(start, start + length))
        dims_used.sort()
        assert dims_used == list(range(min(dims_used), max(dims_used) + 1))

def test_search():
    batch_size, seqlen, nheads, dim = 4, 4096, 8, 128

    # Get source and parse to IR
    source = flash_attn_pack_kv_double_ring_template.format(
        BATCH=batch_size,
        SEQ_LEN=seqlen,
        HEADS=nheads,
        HEAD_DIM=dim,
        SEQ_LEN_IN = seqlen // 2,
        SEQ_LEN_OUT = 2,
    )

    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    # Parse function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        pytest.fail("Could not find function definition")

    devices = [i for i in range(16)]
    mesh = DeviceMesh(devices, (16,))
    for res_program in search(program, mesh):
        eliminate_loops(res_program)
        print(generate_pytorch_code(res_program))

def test_search_attn_reduce():
    batch_size, seqlen, nheads, dim = 4, 4096, 8, 128
    # Get source and parse to IR
    source = flash_attn_manage_reduction.format(
        BATCH=batch_size,
        SEQ_LEN=seqlen,
        HEADS=nheads,
        HEAD_DIM=dim,
        RED_DIM=dim + 1,
    )

    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    # Parse function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        pytest.fail("Could not find function definition")

    devices = [i for i in range(8)]
    mesh = DeviceMesh(devices, (8,))
    for idx, res_program in enumerate(search(program, mesh, ["S_q", "S_kv"])):
        eliminate_loops(res_program)
        print(f"Program {idx}:")
        print(generate_pytorch_code(res_program))
        dump(res_program)

def test_search_gemm():
    m, n, k = 1024, 4096, 512
    source = gemm_manage_reduction.format(
        M_LEN = m,
        N_LEN = n,
        K_LEN = k,
    )

    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    # Parse function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        pytest.fail("Could not find function definition")

    devices = [i for i in range(8)]
    mesh = DeviceMesh(devices, (8,))
    for idx, res_program in enumerate(search(program, mesh, ["I", "J", "K"])):
        eliminate_loops(res_program)
        print(f"Program {idx}:")
        print(generate_pytorch_code(res_program))
        dump(res_program)


if __name__ == "__main__":
    test_search_attn_reduce()
    test_search_gemm()
    test_search()
    test_enumerate_mesh_shapes()
    test_enumerate_mesh_assignment()