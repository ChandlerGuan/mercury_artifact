
import ast
import textwrap
from typing import List, Optional, Tuple

import torch
from mercury.backend.pytorch.codegen import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.elements import Buffer
from mercury.ir.distributed import DeviceMesh, ShardType, ShardingSpec
from mercury.ir.init_distributed import init_distributed
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.ir.nodes import AxisDef, GridLoop, IRNode
from mercury.ir.primitives import shift, cut_buffer, parallelize
from mercury.ir.tile import tile_loop
from mercury.ir.utils import get_io_buffers
from mercury.search.estimate_transfer import estimate_transfer_time
from mercury.search.search import search
from utils.flash_attn_dsl import flash_attn_manage_reduction
import torch.distributed as dist
from utils.gemm_dsl import *

m, n, k = 256, 256, 256

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

def get_divided_buffer(
    buffer_shape: List[int],
    divide_dim: List[Tuple[int, int]],
    buffer_name: str,
    dtype,
    mesh: DeviceMesh,
):
    """
    buffer_shape: shape of the buffer
    divide_dim: list of tuples, each tuple contains the dimension and the corresponding mesh dimension
    buffer_name: name of the buffer
    dtype: data type of the buffer
    mesh: device mesh
    """
    buffer = Buffer(
        tensor=buffer_name,
        shape=buffer_shape,
        bound_axes=[],
        dtype=dtype,
        axes_factor=[],
        shard_spec=ShardingSpec(mesh, [ShardType.REPLICATE] * 2)
    )

    for dim_id, mesh_dim in divide_dim:
        num_cards = mesh.shape[mesh_dim]
        buffer.shape[dim_id] = int(buffer.shape[dim_id] // num_cards)

        dim_list = [mesh_dim]
        if buffer.shard_spec.specs[dim_id] == ShardType.REPLICATE:
            buffer.shard_spec.specs[dim_id] = (ShardType.SHARD, dim_list)
        else:
            old_shard_dim = buffer.shard_spec.specs[dim_id][1]
            old_shard_dim.extend(dim_list)
            new_shard_dim = list(set(old_shard_dim))
            buffer.shard_spec.specs[dim_id] = (ShardType.SHARD, new_shard_dim)
        buffer.shard_spec.validate()

    return buffer

def ag_tp(world_size, m=m, n=n, k=k) -> Tuple[List[Buffer], List[Buffer]]:
    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size,))

    a_shape = [m, k]
    b_shape = [k, n]
    c_shape = [m, n]

    a_buffer = get_divided_buffer(a_shape, [(0, 0)], "a", torch.float16, mesh) # dim 0 (m) is sharded along mesh dim 0
    b_buffer = get_divided_buffer(b_shape, [(1, 0)], "b", torch.float16, mesh) # dim 1 (n) is sharded along mesh dim 0
    c_buffer = get_divided_buffer(c_shape, [(1, 0)], "c", torch.float16, mesh)

    return [a_buffer, b_buffer], [c_buffer]

def rs_tp(world_size, m=m, n=n, k=k) -> Tuple[List[Buffer], List[Buffer]]:
    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size,))

    a_shape = [m, k]
    b_shape = [k, n]
    c_shape = [m, n]

    a_buffer = get_divided_buffer(a_shape, [(1, 0)], "a", torch.float16, mesh) # dim 1 (k) is sharded along mesh dim 0
    b_buffer = get_divided_buffer(b_shape, [(0, 0)], "b", torch.float16, mesh) # dim 0 (k) is sharded along mesh dim 0
    c_buffer = get_divided_buffer(c_shape, [(0, 0)], "c", torch.float16, mesh) # dim 0 (m) is sharded along mesh dim 0

    return [a_buffer, b_buffer], [c_buffer]


def test_search(world_size, get_origin_io):
    source = gemm_manage_reduction.format(
        M_LEN=m,
        N_LEN=n,
        K_LEN=k,
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
    searched_programs = list(search(program, mesh,  ["I", "J", "K"]))

    # for some unkonwn reason, the search result's order is not stable across different devices
    searched_programs.sort(key=lambda x: generate_pytorch_code(x))

    origin_input_buffers, origin_output_buffers = get_origin_io(world_size)

    for idx, searched_program in enumerate(searched_programs):
        if rank == 0:
            print(f"\ntest program {idx + 1}")
        input_buffers, output_buffers = get_input_output_buffers(searched_program)
        time_in = estimate_transfer_time(origin_input_buffers, input_buffers, 1)
        time_out = estimate_transfer_time(output_buffers, origin_output_buffers, 1)
        time = time_in + time_out
        if dist.get_rank() == 0:
            print(f"time: {time} ms")
        dist.barrier()    


if __name__ == "__main__":
    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    test_search(world_size, ag_tp)
    test_search(world_size, rs_tp)



