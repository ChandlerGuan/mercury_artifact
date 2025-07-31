from mercury.frontend.parser import IRBuilder
from mercury.ir.init_distributed import init_distributed
from mercury.backend import *
from mercury.ir.primitives import parallelize, shift
from mercury.ir.loop_eliminating import eliminate_loops
import ast
import textwrap
from mercury.ir.utils import collect_axis, collect_loops
from utils.flash_attn_dsl import *
from mercury.ir.distributed import DeviceMesh
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func, _flash_attn_forward
from utils.utils import log

def get_ring_attn(world_size):

    # Get source and parse to IR
    source = flash_attn_pack_kv_template.format(BATCH=4, HEADS=5, SEQ_LEN=4096, HEAD_DIM=128)
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()

    program = builder.visit(tree.body[0])

    axes = program.visit(collect_axis)
    loops = program.visit(collect_loops)

    axis = axes[2] # S_q
    outer_loop = loops[0]

    devices = list(range(world_size))
    mesh = DeviceMesh(devices, (world_size,))

    # add distributed info
    init_distributed(program, mesh)

    # parallelize q
    parallelize(program, outer_loop, axis, mesh, 0, len(mesh.shape))

    # shift kv
    shift(program, axis, mesh, 0, len(mesh.shape), 1)

    # eliminate loops
    eliminate_loops(program)

    # gen pytorch code
    code = generate_pytorch_code(program)

    return code

def run_func():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    world_size = dist.get_world_size()
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")

    batch_size = 4
    seqlen = 4096
    nheads = 5
    d = 128
    dropout_p = 0
    causal = False
    deterministic = False

    assert seqlen % world_size == 0
    assert d % 8 == 0

    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)

    code = get_ring_attn(world_size)
    if rank == 0:
        print("#" * 30)
        print("# generated code:")
        print("#" * 30)
        print(code)
        print("#" * 30)
        print("# run generated code:")
        print("#" * 30)

    namespace = globals()
    exec(code, namespace)
    func = namespace["flash_attn_pack_kv"]

    local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()

    dist.barrier()

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

    log("out", out, rank0_only=True)
    log("lse", lse, rank0_only=True)
    log("out diff", local_out - ring_out)
    log("lse diff", local_lse - ring_lse)

    dist.barrier()

if __name__ == '__main__':
    run_func()