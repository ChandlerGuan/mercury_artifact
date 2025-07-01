from typing import Optional
import pytest
import torch
import torch.distributed as dist
from mercury.frontend.parser import IRBuilder
from mercury.ir.init_distributed import init_distributed
from mercury.ir.nodes import (
    IRNode, AxisDef, GridLoop
)
from mercury.backend import *
import mercury.ir.primitives as sp
import mercury.ir.loop_eliminating as le
import ast
import textwrap
import inspect
from flash_attn.flash_attn_interface import _flash_attn_forward, flash_attn_qkvpacked_func
from mercury.ir.tile import tile_loop
from utils.flash_attn_dsl import *
from utils.utils import log
from mercury.ir.distributed import DeviceMesh

batch, heads, seq_len, head_dim = 4, 8, 4096, 128

def test_split_double_reduce_ring(world_size=8):

    # Get source and parse to IR
    source = flash_attn_manage_reduction.format(
        BATCH=batch,
        HEADS=heads, 
        SEQ_LEN=seq_len, 
        HEAD_DIM=head_dim,
        RED_DIM=head_dim + 1
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

    # collect axes
    def collect_axes(node: IRNode) -> Optional[AxisDef]:
        return node.axis if isinstance(node, AxisDef) else None    
    
    axes = program.visit(collect_axes)

    axis_q = axes[2]

    tile_loop(program, axis_q, seq_len // 2)

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size // 2,2))

    init_distributed(program, mesh)

    axes = program.visit(collect_axes)

    axis_kv = axes[4]

    # we want to get the loop
    def collect_loops(node: IRNode) -> Optional[GridLoop]:
        return node if isinstance(node, GridLoop) else None
    
    loops = program.visit(collect_loops)
    assert len(loops) == 2, "Should have 2 grid loops"
    
    inner_loop = loops[1]

    sp.parallelize(program, inner_loop, axis_kv, mesh, 0,2)

    sp.shift(program, axis_kv, mesh, 0, 2, 1)

    le.eliminate_loops(program)


    code = generate_pytorch_code(program)

    print(code)

    # Execute generated code
    namespace = globals()
    
    exec(code, namespace)
    return namespace[program.name]

def test_split_double_ring(world_size=8):

    # Get source and parse to IR
    source = flash_attn_manage_reduction.format(
        BATCH=batch,
        HEADS=heads, 
        SEQ_LEN=seq_len, 
        HEAD_DIM=head_dim,
        RED_DIM=head_dim + 1
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

    # collect axes
    def collect_axes(node: IRNode) -> Optional[AxisDef]:
        return node.axis if isinstance(node, AxisDef) else None    
    
    axes = program.visit(collect_axes)

    axis_kv = axes[3]

    tile_loop(program, axis_kv, seq_len // 2)

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size // 2,2))

    init_distributed(program, mesh)

    axis_q = axes[2] # S_q

    # we want to get the loop
    def collect_loops(node: IRNode) -> Optional[GridLoop]:
        return node if isinstance(node, GridLoop) else None
    
    loops = program.visit(collect_loops)
    assert len(loops) == 2, "Should have 2 grid loops"
    
    # Outer loop should be spatial
    outer_loop = next(l for l in loops if len(l.axes) == 3)

    sp.parallelize(program, outer_loop, axis_q, mesh, 0,2)

    sp.shift(program, axis_q, mesh, 0, 2, 1)

    le.eliminate_loops(program)

    code = generate_pytorch_code(program)

    print(code)

    # Execute generated code
    namespace = globals()
    
    exec(code, namespace)
    return namespace[program.name]

def test_tree_over_ring(world_size=8):
    # Get source and parse to IR
    source = flash_attn_manage_reduction.format(
        BATCH=batch,
        HEADS=heads, 
        SEQ_LEN=seq_len, 
        HEAD_DIM=head_dim,
        RED_DIM=head_dim + 1
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

    # collect axes
    def collect_axes(node: IRNode) -> Optional[AxisDef]:
        return node.axis if isinstance(node, AxisDef) else None    
    
    # split both q and kv

    axes = program.visit(collect_axes)

    axis_kv = axes[3]

    tile_loop(program, axis_kv, seq_len // 2)

    axes = program.visit(collect_axes)

    axis_q = axes[2] # S_q

    tile_loop(program, axis_q, seq_len // 2)

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size // 2,2))

    init_distributed(program, mesh)

    axes = program.visit(collect_axes)

        # we want to get the loop
    def collect_loops(node: IRNode) -> Optional[GridLoop]:
        return node if isinstance(node, GridLoop) else None
    
    loops = program.visit(collect_loops)
    assert len(loops) == 2, "Should have 2 grid loops"
    
    # Outer loop should be spatial
    outer_loop = loops[0]
    inner_loop = loops[1]

    # ring in inner kv
    # tree in outer kv
    inner_q = axes[3]
    outer_kv = axes[4]

    sp.parallelize(program, outer_loop, inner_q, mesh, 0, 1)

    sp.shift(program, inner_q, mesh, 0, 1, 1)

    sp.parallelize(program, inner_loop, outer_kv, mesh, 1,2)

    le.eliminate_loops(program)

    code = generate_pytorch_code(program)

    print(code)

    # Execute generated code
    namespace = globals()
    
    exec(code, namespace)
    return namespace[program.name]
    
def test_split_reduce_inner_ring(world_size=4):
    # Get source and parse to IR
    source = flash_attn_manage_reduction.format(
        BATCH=batch,
        HEADS=heads, 
        SEQ_LEN=seq_len, 
        HEAD_DIM=head_dim,
        RED_DIM=head_dim + 1
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

    # collect axes
    def collect_axes(node: IRNode) -> Optional[AxisDef]:
        return node.axis if isinstance(node, AxisDef) else None    
    
    axes = program.visit(collect_axes)

    axis_q = axes[2]

    tile_loop(program, axis_q, seq_len // 2)

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size // 2,2))

    init_distributed(program, mesh)

    axes = program.visit(collect_axes)

    axis_kv = axes[4]

    # we want to get the loop
    def collect_loops(node: IRNode) -> Optional[GridLoop]:
        return node if isinstance(node, GridLoop) else None
    
    loops = program.visit(collect_loops)
    assert len(loops) == 2, "Should have 2 grid loops"
    
    inner_loop = loops[1]

    used_axes = set()
    sp.parallelize(program, inner_loop, axis_kv, mesh, 0, 2, used_axes)

    axis_q_outer = axes[2]

    used_axes.add(axis_q_outer.name)

    sp.shift(program, axis_kv, mesh, 0, 2, 1, used_axes)

    le.eliminate_loops(program)

    print(program)
    code = generate_pytorch_code(program)

    print(code)

    # Execute generated code
    namespace = globals()
    
    exec(code, namespace)
    return namespace[program.name]

def run_double_ring():
    rank = dist.get_rank()

    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    dropout_p = 0
    causal = False
    deterministic = False

    assert seq_len % world_size == 0
    assert head_dim % 8 == 0

    qkv = torch.randn(
        batch, seq_len, 3, heads, head_dim, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)

    # devices = [i for i in range(world_size)]
    # mesh = DeviceMesh(devices, (world_size,))
    func = test_split_double_ring(world_size)

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
    ring_lse = torch.zeros(batch, heads, seq_len // world_size, device=device, dtype=torch.float32)

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

    assert torch.allclose(local_out, ring_out, atol=1e-3)
    assert torch.allclose(local_lse, ring_lse, atol=1e-3)
    dist.barrier()

def run_ring_over_q():
    rank = dist.get_rank()

    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    dropout_p = 0
    causal = False
    deterministic = False

    qkv = torch.randn(
        batch, seq_len, 3, heads, head_dim, device=device, dtype=dtype,
    )
    dist.broadcast(qkv, src=0)

    # devices = [i for i in range(world_size)]
    # mesh = DeviceMesh(devices, (world_size,))
    func = test_split_double_reduce_ring(world_size)

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
    ring_lse = torch.zeros(batch, heads, seq_len // world_size, device=device, dtype=torch.float32)


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

def run_tree_over_ring():
    rank = dist.get_rank()

    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    dropout_p = 0
    causal = False
    deterministic = False

    qkv = torch.randn(
        batch, seq_len, 3, heads, head_dim, device=device, dtype=dtype,
    )
    dist.broadcast(qkv, src=0)

    # devices = [i for i in range(world_size)]
    # mesh = DeviceMesh(devices, (world_size,))
    func = test_tree_over_ring(world_size)

    local_q = qkv[:, :, 0].chunk(world_size // 2, dim=1)[rank % 2].detach().clone()
    local_kv = qkv[:, :, 1:3].chunk(world_size, dim=1)[rank].detach().permute(2, 0, 1, 3, 4).clone()

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

    local_out = out.chunk(world_size // 2, dim=1)[rank % 2]
    local_lse = lse.chunk(world_size // 2, dim=-1)[rank % 2]

    ring_out = torch.zeros_like(local_q)
    ring_lse = torch.zeros(batch, heads, seq_len * 2 // world_size, device=device, dtype=torch.float32)


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

def run_inner_ring_reduce():
    rank = dist.get_rank()

    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    dropout_p = 0
    causal = False
    deterministic = False

    qkv = torch.randn(
        batch, seq_len, 3, heads, head_dim, device=device, dtype=dtype,
    )
    dist.broadcast(qkv, src=0)

    # devices = [i for i in range(world_size)]
    # mesh = DeviceMesh(devices, (world_size,))
    func = test_split_reduce_inner_ring(world_size)

    local_q = qkv[:, :, 0].chunk(world_size // 2, dim=1)[rank % 2].detach().clone()
    local_kv = qkv[:, :, 1:3].chunk(world_size, dim=1)[rank].detach().permute(2, 0, 1, 3, 4).clone()

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

    local_out = out.chunk(world_size // 2, dim=1)[rank % 2]
    local_lse = lse.chunk(world_size // 2, dim=-1)[rank % 2]

    ring_out = torch.zeros_like(local_q)
    ring_lse = torch.zeros(batch, heads, seq_len * 2 // world_size, device=device, dtype=torch.float32)


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

if __name__ == '__main__':
    dist.init_process_group("nccl")
    run_inner_ring_reduce()
    run_ring_over_q()
    run_double_ring()
    run_tree_over_ring()
