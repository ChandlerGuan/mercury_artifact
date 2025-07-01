import ast
import textwrap
from flash_attn.flash_attn_interface import _flash_attn_forward
import os
import torch
import torch.distributed as dist
from mercury.backend.pytorch.codegen import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.calculate_memory import get_buffer_size
from mercury.ir.distributed import DeviceMesh
from mercury.ir.init_distributed import init_distributed
from mercury.ir.nodes import AxisDef, GridLoop, IRNode
from mercury.backend.pytorch import generate_pytorch_code
from mercury.ir import eliminate_loops
from mercury.ir import parallelize
from mercury.ir.primitives import shift
from utils.flash_attn_dsl import *

batch_size = 4
deterministic = False
seqlen = 4096 * 8
num_heads = 32
head_dim = 128
causal = False

def benchmark(f, num_iter=100, log=True, profile=False):
    dtype = torch.bfloat16
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    q = torch.randn(
        batch_size,
        seqlen // world_size,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    kv = torch.randn(
        2,
        batch_size,
        seqlen // world_size,
        num_heads,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    ring_out = torch.zeros_like(q)
    ring_lse = torch.zeros(batch_size, num_heads, seqlen // world_size, device=device, dtype=torch.float32)

    if profile:
        torch.backends.cudnn.benchmark = True
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=5,
                warmup=5,
                active=5,
            ),
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(
                    f"./benchmark/logs/{f.__name__}", f"rank_{dist.get_rank()}"
                )
            ),
        )

    if profile:
        profiler.start()

    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    with torch.no_grad():
        for _ in range(num_iter):
            _ = f(
                q,
                kv,
                ring_out,
                ring_lse,
                q.shape[-1] ** (-0.5),
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=deterministic,
            )
            if profile:
                profiler.step()


    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0

    if profile:
        profiler.stop()

    if rank == 0 and log:
        print(f"{num_iter / time:.3f} iter/s, {time:.3f} sec")

def gen_func(ring_divide: int, world_size: int = 8):
    # Get source and parse to IR
    source = flash_attn_pack_kv_template.format(
        BATCH=batch_size,
        HEADS=num_heads,
        SEQ_LEN=seqlen,
        HEAD_DIM=head_dim
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

    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size,))

    init_distributed(program, mesh)
    
    parallelize(program, outer_loop, axis.axis, mesh, 0, len(mesh.shape))
    shift(program, axis.axis, mesh, 0, len(mesh.shape), ring_divide)

    eliminate_loops(program)

    code = generate_pytorch_code(program)

    # Execute generated code
    namespace = globals()
    
    exec(code, namespace)
    return namespace["flash_attn_pack_kv"], get_buffer_size(program)


if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    profile = False
    num_iter = 100

    for i in range(4):
        ring_divide = 2 ** i
        # we neglect the temp buffer for _out and _lse
        # as they are small and not tracked by the ir
        # for more accurate memory usage, you should add them
        f, memory_cost = gen_func(ring_divide, world_size)
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"# {f.__name__} memory cost: {memory_cost / 1024 / 1024:.2f} MB")
        benchmark(f, num_iter=num_iter, log=False)
        benchmark(f, num_iter=num_iter, log=True, profile=profile)

