import ast
import textwrap
from typing import Optional
import pytest
import torch
import torch.distributed as dist
from mercury.frontend.parser import IRBuilder
from mercury.ir.init_distributed import init_distributed
from mercury.ir.nodes import Buffer
from mercury.backend import *
from mercury.ir.distributed import DeviceMesh
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.search.dump import dump
from mercury.search.search import search
from flash_attn.flash_attn_interface import flash_attn_kvpacked_func, _flash_attn_forward
from utils.flash_attn_dsl import *
from utils.utils import log
from mercury.ir.utils import get_io_buffers
from mercury.ir.distributed import ShardType
from utils.gemm_dsl import *

m = 512
n = 256
k = 1024

def run_validation(source):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    a = torch.randn(
        m, k, device = device, dtype = dtype
    )

    a = a / 100
    
    b = torch.randn(
        k, n, device = device, dtype = dtype
    )

    b = b / 100

    dist.broadcast(a, src=0)
    dist.broadcast(b, src=0)
    old_a = a.detach().clone()
    old_b = b.detach().clone()

    res = torch.matmul(a, b)
    old_res = res.detach().clone()

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
    searched_programs = list(search(program, mesh, ["I", "J", "K"]))

    # for some unkonwn reason, the search result's order is not stable across different devices
    searched_programs.sort(key=lambda x: generate_pytorch_code(x))

    for idx, res_program in enumerate(searched_programs):
        if rank == 0:
            print(f"\nTesting program {idx + 1}/{len(searched_programs)}")

        eliminate_loops(res_program)
        code = generate_pytorch_code(res_program)

        if rank ==0:
            print(code)
            dump(res_program)
        
        namespace = globals()
        exec(code, namespace)
        func = namespace[program.name]

        buffers = res_program.visit(get_io_buffers)

        local_tensors = {}
        for buffer in buffers:
            device_coords = []

            for dim, spec in enumerate(buffer.shard_spec.specs):
                if isinstance(spec, tuple) and spec[0] == ShardType.SHARD:
                    # we will change the way to determin the device coords by using the high dim mesh

                    indices = one_dim_to_n_dim(rank, res_program.mesh.shape)

                    # only use the sharded cords
                    shard_coord = tuple([indices[i] for i in spec[1]])
                    shard_mesh = tuple([res_program.mesh.shape[i] for i in spec[1]])

                    device_coords.append(n_dim_to_one_dim(shard_coord, shard_mesh))

                else:
                    device_coords.append(0)

            try:
                is_output = buffer.write

                if buffer.tensor == "a":
                    full_tensor = a
                elif buffer.tensor == "b":
                    full_tensor = b
                elif buffer.tensor == "c":
                    full_tensor = res
                else:
                    raise ValueError(f"Unknown input buffer: {buffer.tensor}")

                local_tensor = full_tensor.detach().clone()
                for dim, coord in enumerate(device_coords):
                    size = buffer.shape[dim]
                    local_tensor = local_tensor.narrow(dim, coord * size, size)

                local_tensor = local_tensor.contiguous()


                if is_output:
                    local_tensors[buffer.tensor + "truth"] = local_tensor
                    if buffer.tensor == "c":
                        dtype = torch.bfloat16
                    else:  # lse
                        dtype = torch.float32
                    local_tensor = torch.zeros(
                        tuple(buffer.shape),
                        device=device,
                        dtype=dtype
                    )

                local_tensors[buffer.tensor] = local_tensor

            except Exception as e:
                if rank == 0:
                    print(f"Error processing buffer {buffer.tensor}: {str(e)}")
                    print(f"Buffer info: {buffer}")
                raise

        dist.barrier()
        if rank == 0:
            print("#" * 30)
            print(f"# Testing Forward Pass for Program {idx + 1}:")
            print("#" * 30)

        local_res_truth = local_tensors["ctruth"]

        local_a = local_tensors["a"].contiguous()
        local_b = local_tensors["b"].contiguous()
        local_res  = local_tensors["c"]

        assert old_a.equal(a), "a has been modified"
        assert old_b.equal(b), "b has been modified"
        assert old_res.equal(res), "res has been modified"

        func(
            local_a,
            local_b,
            local_res,
        )

        res_diff = torch.abs(local_res_truth - local_res)
        
        max_res_diff = res_diff.max().item()
        
        log(f"Program {idx+1} max res diff", local_res_truth - local_res)
        
        assert max_res_diff < 1e-3, f"Res difference too large: {max_res_diff}"
        
        dist.barrier()

if __name__ == "__main__":
    dist.init_process_group("nccl")
    source = gemm_manage_reduction.format(
        M_LEN = m,
        N_LEN = n,
        K_LEN = k,
    )
    run_validation(source)
    