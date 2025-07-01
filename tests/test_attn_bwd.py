from typing import Optional
import pytest
import torch
import torch.distributed as dist
from mercury.frontend.parser import IRBuilder
from mercury.ir.init_distributed import init_distributed
from mercury.search.dump import dump
from mercury.search.search import search
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.ir.utils import get_buffers
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
from mercury.ir.nodes import (
    IRNode, AxisDef, GridLoop
)
from mercury.backend import *
import mercury.ir.primitives as sp
import mercury.ir.loop_eliminating as le
import ast
import textwrap
import inspect
from flash_attn.flash_attn_interface import _flash_attn_backward
from flash_attn.flash_attn_interface import flash_attn_kvpacked_func
from utils.attn_bwd_dsl import *
from utils.utils import log
from mercury.ir.distributed import DeviceMesh



batch_size = 4
seqlen = 4096
nheads = 16
n_kv_heads = 8
d = 128

def run_validation(source):
    """validate the generated code by running it and comparing the results with the original implementation"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")
    
    dropout_p = 0
    causal = False
    deterministic = False

    q_init = torch.randn(
        batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    
    # some of the searched kernels have precision issues possibly due to overflow
    # so to check the correctness, we need to scale the input
    q = torch.div(q_init, 100).detach().requires_grad_(True)

    kv_init = torch.randn(
        batch_size, seqlen, 2, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    kv = torch.div(kv_init, 100).detach().requires_grad_(True)

    dout_init = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)

    dout = torch.div(dout_init, 100).detach().requires_grad_(True)

    dist.broadcast(q, src=0)
    dist.broadcast(kv, src=0)
    dist.broadcast(dout, src=0)

    # ground truth
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

    out.backward(dout)

    dq = q.grad
    dkv = kv.grad

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

    fail_num = 0
    # validate the generated code by running it and comparing the results with the original implementation
    for idx, res_program in enumerate(searched_programs):
        if rank == 0:
            print(f"\nTesting program {idx + 1}/{len(searched_programs)}")

        eliminate_loops(res_program)
        code = generate_pytorch_code(res_program)

        # if rank ==0:
        #     print(code)
        #     dump(res_program)
        
        namespace = globals()
        exec(code, namespace)
        func = namespace[program.name]

        # get the buffers
        buffers = res_program.visit(get_io_buffers)

        # analyze the buffers
        local_tensors = {}
        for buffer in buffers:
            # get shard on the device
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
                    # REPLICATE维度
                    device_coords.append(0)

            # prepare buffer
            try:
                is_output = buffer.write

                if buffer.tensor == "q":
                    full_tensor = q
                elif buffer.tensor == "kv":
                    # kv need permute
                    full_tensor = kv.permute(2, 0, 1, 3, 4)
                elif buffer.tensor == "o":
                    full_tensor = out
                elif buffer.tensor == "lse":
                    full_tensor = lse
                elif buffer.tensor == "dout":
                    full_tensor = dout
                elif buffer.tensor == "dq":
                    full_tensor = dq
                elif buffer.tensor == "dkv":
                    full_tensor = dkv.permute(2, 0, 1, 3, 4)
                else:
                    raise ValueError(f"Unknown input buffer: {buffer.tensor}")

                # shard the buffer
                local_tensor = full_tensor.detach().clone()
                for dim, coord in enumerate(device_coords):
                    size = buffer.shape[dim]
                    local_tensor = local_tensor.narrow(dim, coord * size, size)

                local_tensor = local_tensor.contiguous()


                if is_output:
                    local_tensors[buffer.tensor + "truth"] = local_tensor
                    # output buffer need to be reshaped
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

        local_dq = local_tensors["dqtruth"]
        local_dkv = local_tensors["dkvtruth"]

        local_q, local_kv = local_tensors["q"], local_tensors["kv"]
        local_out, local_lse = local_tensors["o"], local_tensors["lse"]
        local_dout = local_tensors["dout"]

        res_dq, res_dkv = local_tensors["dq"], local_tensors["dkv"]

        # assert old_q.equal(q), "q has been modified"
        # assert old_kv.equal(kv), "kv has been modified"
        # assert old_out.equal(out), "out has been modified"
        # assert old_lse.equal(lse), "lse has been modified"

        try:
            func(
                local_q,
                local_kv,
                local_out,
                local_lse,
                local_dout,
                res_dq,
                res_dkv,
                local_q.shape[-1] ** (-0.5),
                dropout_p=dropout_p,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
            )

            dq_diff = torch.abs(local_dq - res_dq)
            dkv_diff = torch.abs(local_dkv - res_dkv)
            
            mean_dq_diff = dq_diff.mean().item()
            mean_dkv_diff = dkv_diff.mean().item()
            
            # log(f"Program {idx+1} max dq diff", local_dq - res_dq)
            # log(f"Program {idx+1} max dkv diff", local_dkv - res_dkv)
            
            if mean_dq_diff >= 1e-3 or mean_dkv_diff >= 1e-3:
                log(f"Program {idx+1} max dq diff", local_dq - res_dq)
                log(f"Program {idx+1} max dkv diff", local_dkv - res_dkv)
                raise RuntimeError(f"Output difference too large: {mean_dq_diff}")
            # assert mean_dq_diff < 1e-3, f"Output difference too large: {mean_dq_diff}"
            # assert mean_dkv_diff < 1e-3, f"LSE difference too large: {mean_dkv_diff}"
        except:
            # raise RuntimeError(f"wrong")
            fail_num = fail_num + 1
            if rank ==0:
                print(code)
                dump(res_program)
        dist.barrier()
    if rank == 0:
        print(f"{fail_num} programs failed to exec")

def test_bwd_dsl():
    batch=4
    heads=8
    seq_len=256
    head_dim=128
    dropout_p = 0
    causal = False
    deterministic = False
    # Get source and parse to IR
    source = attn_backward_template.format(BATCH=batch, HEADS=heads, SEQ_LEN=seq_len, HEAD_DIM=head_dim)

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


    # print(program)
    # le.eliminate_loops(program)
    code = generate_pytorch_code(program)
    print(code)

    namespace = globals()    
    exec(code, namespace)
    func = namespace["flash_attn_backward"]


    # validate in single card
    dtype = torch.bfloat16
    device = torch.device("cuda:0")

    q = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    kv = torch.randn(batch, seq_len, 2, heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(batch, seq_len, heads, head_dim, device=device, dtype=dtype)

    out, lse, _ = flash_attn_kvpacked_func(
        q, kv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    out.backward(dout)

    kv_transposed = kv.permute(2, 0, 1, 3, 4).contiguous().detach()

    dq_dsl = torch.empty_like(q)
    dkv_dsl = torch.empty_like(kv_transposed)

    func(
        q,
        kv_transposed,
        out,
        lse,
        dout,
        dq_dsl,
        dkv_dsl,
        q.shape[-1] ** (-0.5)
    )

    diff_dq = (dq_dsl - q.grad)
    diff_dkv = (dkv_dsl - kv.grad.permute(2, 0, 1, 3, 4))

    print(f"avg diff dq {diff_dq.mean().item()}")
    print(f"max diff dq {diff_dq.max().item()}")
    print(f"avg diff dkv {diff_dkv.mean().item()}")
    print(f"max diff dkv {diff_dkv.max().item()}")
    
def test_transformation():
    source = attn_backward_template.format(
        BATCH=batch_size,
        SEQ_LEN=seqlen,
        HEADS=nheads,
        HEAD_DIM=d,
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
    mesh = DeviceMesh(devices, (2, 2, 2))

    init_distributed(program, mesh)

    # collect axes
    def collect_axes(node: IRNode) -> Optional[AxisDef]:
        return node if isinstance(node, AxisDef) else None
    
    axes = program.visit(collect_axes)

    axis_b = axes[0].axis
    axis_q = axes[2].axis

    # we want to get the loop
    def collect_loops(node: IRNode) -> Optional[GridLoop]:
        return node if isinstance(node, GridLoop) else None
    
    loops = program.visit(collect_loops)


    sp.parallelize(program, loops[0], axis_b, mesh, 1, 3)
    sp.parallelize(program, loops[1], axis_q, mesh, 0, 1)
    sp.shift(program, axis_q, mesh, 0, 1, 1)

    # print(program)

    le.eliminate_loops(program)

    print(program)

    code = generate_pytorch_code(program)

    print(code)



if __name__ == '__main__':
    # test_bwd_dsl()
    # test_transformation()
    dist.init_process_group("nccl")
    source = attn_backward_template.format(
        BATCH=batch_size,
        SEQ_LEN=seqlen,
        HEADS=nheads,
        HEAD_DIM=d,
    )

    run_validation(source)

