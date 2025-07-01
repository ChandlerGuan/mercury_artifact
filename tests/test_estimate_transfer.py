
import ast
import textwrap
from typing import Optional

import torch
from mercury.backend.pytorch.codegen import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh
from mercury.ir.init_distributed import init_distributed
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.ir.nodes import AxisDef, GridLoop, IRNode
from mercury.ir.primitives import shift, parallelize
from mercury.ir.tile import tile_loop
from mercury.ir.utils import get_io_buffers
from mercury.search.estimate_transfer import estimate_transfer_time
from mercury.search.search import search
from utils.flash_attn_dsl import flash_attn_manage_reduction
import torch.distributed as dist

batch_size, seqlen, nheads, dim = 8, 4096, 8, 128

def get_input_output_buffers(program):
    buffers = program.visit(get_io_buffers)

    input_buffers = []
    output_buffers = []
    
    for buffer in buffers:
        if not buffer.write:
            input_buffers.append(buffer)
        else:
            output_buffers.append(buffer)
    return input_buffers, output_buffers


def get_seq_parallel_input_output(world_size = 8, batch_size = batch_size,
                                  seqlen = seqlen, nheads = nheads, dim = dim):
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

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size,))

    init_distributed(program, mesh)

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

    parallelize(program, outer_loop, axis.axis, mesh, 0, len(mesh.shape))

    shift(program, axis.axis, mesh, 0, len(mesh.shape), 1)

    return get_input_output_buffers(program)

def test_estimate_no_transfer(world_size=8):
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

    # collect axes
    def collect_axes(node: IRNode) -> Optional[AxisDef]:
        return node.axis if isinstance(node, AxisDef) else None    
    
    axes = program.visit(collect_axes)

    axis_kv = axes[3]

    tile_loop(program, axis_kv, seqlen // 2)

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

    parallelize(program, outer_loop, axis_q, mesh, 0,2)

    shift(program, axis_q, mesh, 0, 2, 1)

    eliminate_loops(program)

    input_buffers, output_buffers = get_input_output_buffers(program)
    origin_input_buffers, origin_output_buffers = get_seq_parallel_input_output(world_size)

    assert len(input_buffers) == len(origin_input_buffers)
    assert len(output_buffers) == len(origin_output_buffers)

    time_in = estimate_transfer_time(origin_input_buffers, input_buffers)
    time_out = estimate_transfer_time(output_buffers, origin_output_buffers)
    time = time_in + time_out
    if dist.get_rank() == 0:
        print(f"no transfer time {time} ms")

def test_estimate_seq2batch(world_size=8):
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

    # collect axes
    def collect_axes(node: IRNode) -> Optional[AxisDef]:
        return node.axis if isinstance(node, AxisDef) else None    
    
    axes = program.visit(collect_axes)

    axis_kv = axes[3]

    tile_loop(program, axis_kv, seqlen // 2)

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size // 2,2))

    init_distributed(program, mesh)

    axis_batch = axes[0]

    # we want to get the loop
    def collect_loops(node: IRNode) -> Optional[GridLoop]:
        return node if isinstance(node, GridLoop) else None
    
    loops = program.visit(collect_loops)
    assert len(loops) == 2, "Should have 2 grid loops"
    
    # Outer loop should be spatial
    outer_loop = next(l for l in loops if len(l.axes) == 3)

    parallelize(program, outer_loop, axis_batch, mesh, 0,2)

    eliminate_loops(program)

    input_buffers, output_buffers = get_input_output_buffers(program)
    origin_input_buffers, origin_output_buffers = get_seq_parallel_input_output(world_size)

    assert len(input_buffers) == len(origin_input_buffers)
    assert len(output_buffers) == len(origin_output_buffers)

    time_in = estimate_transfer_time(origin_input_buffers, input_buffers)
    time_out = estimate_transfer_time(output_buffers, origin_output_buffers)
    time = time_in + time_out
    if dist.get_rank() == 0:
        print(f"seq to batch time: {time} ms")

    # compare with collective

    device = torch.device(f"cuda:{dist.get_rank()}")
    src_data = torch.randn(world_size, seqlen // world_size, batch_size // world_size, nheads, dim, 3, device=device, dtype = torch.bfloat16)
    dst_data = torch.empty_like(src_data)

    rounds = 50
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(rounds):
        dist.all_to_all_single(dst_data, src_data)
    end_event.record()
    torch.cuda.synchronize()
    time = start_event.elapsed_time(end_event) / rounds
    if dist.get_rank() == 0:
        print(f"collective time: {time * 2} ms")

def test_search_seq2any(world_size=8):
    source = flash_attn_manage_reduction.format(
        BATCH=batch_size,
        SEQ_LEN=seqlen,
        HEADS=nheads,
        HEAD_DIM=dim,
        RED_DIM=dim + 1,
    )
    rank = dist.get_rank()

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size,))

    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    # Parse function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size,))
    searched_programs = list(search(program, mesh,  ["S_q", "S_kv"]))

    # for some unkonwn reason, the search result's order is not stable across different devices
    searched_programs.sort(key=lambda x: generate_pytorch_code(x))

    origin_input_buffers, origin_output_buffers = get_seq_parallel_input_output(world_size)

    for idx, searched_program in enumerate(searched_programs):
        if rank == 0:
            print(f"\ntest program {idx + 1}")
        input_buffers, output_buffers = get_input_output_buffers(searched_program)

        time_in = estimate_transfer_time(origin_input_buffers, input_buffers)
        time_out = estimate_transfer_time(output_buffers, origin_output_buffers)
        time = time_in + time_out
        if dist.get_rank() == 0:
            print(f"time: {time} ms")
        dist.barrier()


if __name__ == "__main__":
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    test_search_seq2any(world_size)
    test_estimate_no_transfer(world_size)
    test_estimate_seq2batch(world_size)

    