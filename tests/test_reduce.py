# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

from typing import Optional
import pytest
import torch
from mercury.frontend.parser import IRBuilder, auto_schedule
from mercury.ir.calculate_memory import get_buffer_size
from mercury.backend.pytorch import generate_pytorch_code
from mercury.ir.distributed import DeviceMesh
from mercury.ir.init_distributed import init_distributed
from mercury.ir.nodes import (
    IRNode, Program, AxisDef, BufferMatch, GridLoop,
)
import mercury.ir.primitives as sp
import mercury.ir.loop_eliminating as le
import ast
import textwrap
from flash_attn.flash_attn_interface import _flash_attn_forward
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
from utils.utils import log
import torch.distributed as dist
from utils.flash_attn_dsl import *

batch_size, seqlen, nheads, dim = 4, 4096, 5, 128

def test_reduce(world_size=8):

    # Get source and parse to IR
    source = flash_attn_manage_reduction.format(
        BATCH=batch_size,
        SEQ_LEN=seqlen,
        HEADS=nheads,
        HEAD_DIM=dim,
        RED_DIM=dim + 1,
    )
    # print(source)

    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    # Parse function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        pytest.fail("Could not find function definition")

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size,))

    init_distributed(program, mesh)

    # print(program)

    # collect axes
    def collect_axes(node: IRNode) -> Optional[AxisDef]:
        return node if isinstance(node, AxisDef) else None
    
    axes = program.visit(collect_axes)

    axis = axes[3] # S_q

    # we want to get the loop
    def collect_loops(node: IRNode) -> Optional[GridLoop]:
        return node if isinstance(node, GridLoop) else None
    
    loops = program.visit(collect_loops)
    assert len(loops) == 2, "Should have 2 grid loops"
    
    inner_loop = loops[1]

    sp.parallelize(program, inner_loop, axis.axis, mesh, 0,1)

    # print(program)

    le.eliminate_loops(program)

    # print(program)

    print(get_buffer_size(program) / 1024 / 1024, "MB")

    code = generate_pytorch_code(program)

    # print(code)

    # Execute generated code
    namespace = globals()
    
    exec(code, namespace)
    return namespace["flash_attn_expose_reduce"]

def test_reduce_ring(world_size=8):
    # Get source and parse to IR
    source = flash_attn_manage_reduction.format(
        BATCH=batch_size,
        SEQ_LEN=seqlen,
        HEADS=nheads,
        HEAD_DIM=dim,
        RED_DIM=dim + 1,
    )
    # print(source)

    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    # Parse function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        pytest.fail("Could not find function definition")

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size,))

    init_distributed(program, mesh)

    # print(program)

    # collect axes
    def collect_axes(node: IRNode) -> Optional[AxisDef]:
        return node if isinstance(node, AxisDef) else None
    
    axes = program.visit(collect_axes)

    axis = axes[3] # S_q

    # we want to get the loop
    def collect_loops(node: IRNode) -> Optional[GridLoop]:
        return node if isinstance(node, GridLoop) else None
    
    loops = program.visit(collect_loops)
    assert len(loops) == 2, "Should have 2 grid loops"
    
    inner_loop = loops[1]

    sp.parallelize(program, inner_loop, axis.axis, mesh, 0,1)

    sp.shift(program, axis.axis, mesh, 0, 1, 1)

    # print(program)

    le.eliminate_loops(program)

    # print(program)

    print(get_buffer_size(program) / 1024 / 1024, "MB")

    code = generate_pytorch_code(program)

    # print(code)

    # Execute generated code
    namespace = globals()
    
    exec(code, namespace)
    return namespace["flash_attn_expose_reduce"]

def test_parallel_q_and_kv(world_size=8):
     # Get source and parse to IR
    source = flash_attn_manage_reduction.format(
        BATCH=batch_size,
        SEQ_LEN=seqlen,
        HEADS=nheads,
        HEAD_DIM=dim,
        RED_DIM=dim + 1,
    )
    # print(source)

    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    # Parse function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        pytest.fail("Could not find function definition")

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size//2, 2)) # (kv, q)

    init_distributed(program, mesh)

    # print(program)

    # collect axes
    def collect_axes(node: IRNode) -> Optional[AxisDef]:
        return node if isinstance(node, AxisDef) else None
    
    axes = program.visit(collect_axes)

    axis_q = axes[2].axis
    axis_kv = axes[3].axis # S_kv

    # we want to get the loop
    def collect_loops(node: IRNode) -> Optional[GridLoop]:
        return node if isinstance(node, GridLoop) else None
    
    loops = program.visit(collect_loops)
    assert len(loops) == 2, "Should have 2 grid loops"
    
    outer_loop = loops[0]
    inner_loop = loops[1]

    sp.parallelize(program, inner_loop, axis_kv, mesh, 0, 1)
    sp.parallelize(program, outer_loop, axis_q, mesh, 1, 2)

    # print(program)

    le.eliminate_loops(program)

    # print(program)

    print(get_buffer_size(program) / 1024 / 1024, "MB")

    code = generate_pytorch_code(program)

    print(code)

    # Execute generated code
    namespace = globals()
    
    exec(code, namespace)
    return namespace["flash_attn_expose_reduce"]

def run_tree_attn():
    rank = dist.get_rank()

    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    dropout_p = 0
    causal = False
    deterministic = False

    assert seqlen % world_size == 0
    assert dim % 8 == 0

    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, dim, device=device, dtype=dtype,
    )
    dist.broadcast(qkv, src=0)

    # devices = [i for i in range(world_size)]
    # mesh = DeviceMesh(devices, (world_size,))
    func = test_reduce(world_size)

    local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv.requires_grad = True

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )


    q = qkv[:, :, 0].contiguous()
    local_kv = local_qkv[:, :, 1:3]
    local_kv = local_kv.permute(2, 0, 1, 3, 4).contiguous()
    ring_out = torch.zeros_like(q)
    ring_lse = torch.zeros(batch_size, nheads, seqlen, device=device, dtype=torch.float32)

    func(
        q,
        local_kv,
        ring_out,
        ring_lse,
        q.shape[-1] ** (-0.5),
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
    )

    log("out diff", out - ring_out)
    log("lse diff", lse - ring_lse)

    assert torch.allclose(out, ring_out, atol=1e-3)
    assert torch.allclose(lse, ring_lse, atol=1e-3)

    dist.barrier()

def run_ring_over_q():
    rank = dist.get_rank()

    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    dropout_p = 0
    causal = False
    deterministic = False

    assert seqlen % world_size == 0
    assert dim % 8 == 0

    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, dim, device=device, dtype=dtype,
    )
    dist.broadcast(qkv, src=0)

    # devices = [i for i in range(world_size)]
    # mesh = DeviceMesh(devices, (world_size,))
    func = test_reduce_ring(world_size)

    local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()
    local_qkv.requires_grad = True

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out = out.chunk(world_size, dim=1)[rank]
    local_lse = lse.chunk(world_size, dim=-1)[rank]

    local_q, local_kv = local_qkv[:, :, 0], local_qkv[:, :, 1:3]
    local_q, local_kv = local_q.contiguous(), local_kv.permute(2, 0, 1, 3, 4).contiguous()
    ring_out = torch.zeros_like(local_q)
    ring_lse = torch.zeros(batch_size, nheads, seqlen // world_size, device=device, dtype=torch.float32)


    func(
        local_q,
        local_kv,
        ring_out,
        ring_lse,
        local_q.shape[-1] ** (-0.5),
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
    )

    log("out diff", local_out - ring_out)
    log("lse diff", local_lse - ring_lse)

    assert torch.allclose(local_out, ring_out, atol=1e-3)
    assert torch.allclose(local_lse, ring_lse, atol=1e-3)
    
    dist.barrier()

if __name__ == "__main__":
    # test_reduce()
    # test_reduce_ring()
    # test_parallel_q_and_kv()
    dist.init_process_group("nccl")
    run_ring_over_q()
    run_tree_attn()