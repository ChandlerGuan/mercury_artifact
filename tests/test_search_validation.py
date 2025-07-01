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
from mercury.ir.utils import get_buffers
from mercury.ir.distributed import ShardType

batch_size = 4
seqlen = 4096
nheads = 16
n_kv_heads = 8
d = 128

def run_validation(source):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    
    dropout_p = 0
    causal = False
    deterministic = False

    q = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype
    )

    kv = torch.randn(
        batch_size, seqlen, 2, n_kv_heads, d, device=device, dtype=dtype
    )

    dist.broadcast(q, src=0)
    dist.broadcast(kv, src=0)
    old_q = q.detach().clone()
    old_kv = kv.detach().clone()

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

    old_out = out.detach().clone()
    old_lse = lse.detach().clone()

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size,))

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
    searched_programs = list(search(program, mesh, ["S_q", "S_kv"]))

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

        buffers = res_program.visit(get_buffers)

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

                if buffer.tensor == "q":
                    full_tensor = q
                elif buffer.tensor == "kv":
                    full_tensor = kv.permute(2, 0, 1, 3, 4)
                elif buffer.tensor == "o":
                    full_tensor = out
                elif buffer.tensor == "lse":
                    full_tensor = lse
                elif buffer.tensor == "reduce_buf":
                    continue
                else:
                    raise ValueError(f"Unknown input buffer: {buffer.tensor}")

                local_tensor = full_tensor.detach().clone()
                for dim, coord in enumerate(device_coords):
                    size = buffer.shape[dim]
                    local_tensor = local_tensor.narrow(dim, coord * size, size)

                local_tensor = local_tensor.contiguous()


                if is_output:
                    local_tensors[buffer.tensor + "truth"] = local_tensor
                    if buffer.tensor == "o":
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

        local_out = local_tensors["otruth"]
        local_lse = local_tensors["lsetruth"]

        local_q, local_kv = local_tensors["q"], local_tensors["kv"]
        ring_out, ring_lse = local_tensors["o"], local_tensors["lse"]

        assert old_q.equal(q), "q has been modified"
        assert old_kv.equal(kv), "kv has been modified"
        assert old_out.equal(out), "out has been modified"
        assert old_lse.equal(lse), "lse has been modified"

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

        out_diff = torch.abs(local_out - ring_out)
        lse_diff = torch.abs(local_lse - ring_lse)
        
        max_out_diff = out_diff.max().item()
        max_lse_diff = lse_diff.max().item()
        
        log(f"Program {idx+1} max out diff", local_out - ring_out)
        log(f"Program {idx+1} max lse diff", local_lse - ring_lse)
        
        assert max_out_diff < 1e-3, f"Output difference too large: {max_out_diff}"
        assert max_lse_diff < 1e-3, f"LSE difference too large: {max_lse_diff}"
        
        dist.barrier()

if __name__ == "__main__":
    dist.init_process_group("nccl")
    # Get source and parse to IR
    # source1 = flash_attn_pack_kv_double_ring_template.format(
    #     BATCH=batch_size,
    #     SEQ_LEN=seqlen,
    #     HEADS=nheads,
    #     HEAD_DIM=d,
    #     SEQ_LEN_IN = seqlen // 2,
    #     SEQ_LEN_OUT = 2,
    # )
    # source2 = flash_attn_manage_reduction.format(
    #     BATCH=batch_size,
    #     SEQ_LEN=seqlen,
    #     HEADS=nheads,
    #     HEAD_DIM=d,
    #     RED_DIM=d+1,
    # )
    source = gqa_manage_reduction.format(
        BATCH=batch_size,
        SEQ_LEN=seqlen,
        HEADS=nheads,
        HEAD_DIM=d,
        RED_DIM=d+1,
        KV_HEADS=n_kv_heads,
        HEADS_PER_GROUP=nheads // n_kv_heads,
    )
    # # test the double ring
    # run_validation(source1)
    # test the reduce
    run_validation(source)
    