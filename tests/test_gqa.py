# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

from typing import Optional
import pytest
import torch
from mercury.ir.elements import Axis, Buffer, grid, match_buffer, load_buffer, store_buffer
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
from flash_attn import flash_attn_kvpacked_func
from utils.utils import log
import torch.distributed as dist
from utils.flash_attn_dsl import *

batch_size, seqlen, nheads, n_kv_heads, dim = 4, 4096, 32, 8, 128

def test_gqa(world_size=8):


    # Get source and parse to IR
    source = gqa_pack_kv_template.format(
        BATCH=batch_size,
        SEQ_LEN=seqlen,
        HEADS=nheads,
        HEAD_DIM=dim,
        KV_HEADS=n_kv_heads,
        HEADS_PER_GROUP=nheads // n_kv_heads,
    )
    print(source)
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

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size,))

    init_distributed(program, mesh)

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

    sp.parallelize(program, outer_loop, axis.axis, mesh, 0,1)

    print(program)

    sp.shift(program, axis.axis, mesh, 0, 1, 1)

    print(generate_pytorch_code(program))

    le.eliminate_loops(program)

    print(program)

    code = generate_pytorch_code(program)

    print(code)

    # Execute generated code
    namespace = globals()
    
    exec(code, namespace)
    return namespace["gqa_pack_kv"]

def run_func():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    dropout_p = 0
    causal = False
    deterministic = False

    assert seqlen % world_size == 0
    assert dim % 8 == 0

    q = torch.randn(
        batch_size, seqlen, nheads, dim, device=device, dtype=dtype
    )

    kv = torch.randn(
        batch_size, seqlen, 2, n_kv_heads, dim, device=device, dtype=dtype
    )

    dist.broadcast(q, src=0)
    dist.broadcast(kv, src=0)

    func = test_gqa(world_size)

    local_q = q.chunk(world_size, dim=1)[rank].detach().clone()
    local_kv = kv.chunk(world_size, dim=1)[rank].detach().clone()

    dist.barrier()
    if rank == 0:
        print("#" * 30)
        print("# forward:")
        print("#" * 30)

    out, lse, _ = flash_attn_kvpacked_func(
        q,
        kv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    local_out = out.chunk(world_size, dim=1)[rank]
    local_lse = lse.chunk(world_size, dim=-1)[rank]


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

    log("out", out, rank0_only=True)
    log("lse", lse, rank0_only=True)
    log("out diff", local_out - ring_out)
    log("lse diff", local_lse - ring_lse)

    dist.barrier()

if __name__ == "__main__":
    # test_gqa()
    run_func()