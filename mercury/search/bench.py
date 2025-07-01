import os

import torch
import torch.distributed as dist
from mercury.backend.pytorch.codegen import generate_pytorch_code
from mercury.ir.calculate_memory import get_buffer_size
from mercury.ir.distributed import ShardType
from mercury.ir.loop_eliminating import eliminate_loops
from mercury.ir.nodes import Program
from mercury.ir.utils import get_io_buffers


def bench_program(program: Program, namespace, num_iter=100, kwargs=None, log=True):
    """
    benchmark for the given program
    Args:
        program: the IR program to be benchmarked
        num_iter: number of iterations to test
        kwargs: extra arguments to pass to the program
        log: whether to print the log
    Returns:
        float: iterations per second
        float: total time (seconds)
        float: memory usage (bytes)
    """
    rank = dist.get_rank()
    local_rank = int(os.getenv("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    if kwargs is None:
        kwargs = {}

    eliminate_loops(program)
    code = generate_pytorch_code(program)

    if rank == 0 and log:
        print(code)

    exec(code, namespace)
    func = namespace[program.name]

    # get buffers
    buffers = program.visit(get_io_buffers)
    memory_cost = get_buffer_size(program)

    # prepare input tensors
    local_tensors = {}
    for buffer in buffers:
        is_output = buffer.write
        dtype = buffer.dtype
        if is_output:
            # output buffers use zeros initialization
            local_tensor = torch.zeros(tuple(buffer.shape), device=device, dtype=dtype)
        else:
            # input buffers use random initialization
            local_tensor = torch.randn(tuple(buffer.shape), device=device, dtype=dtype)

        local_tensors[buffer.tensor] = local_tensor

    kwargs.update(local_tensors)

    # warmup
    for _ in range(num_iter):
        func(**kwargs)

    # exec benchmark
    begin = torch.cuda.Event(enable_timing=True)
    begin.record()

    with torch.no_grad():
        for _ in range(num_iter):
            func(**kwargs)

    end = torch.cuda.Event(enable_timing=True)
    end.record()
    torch.cuda.synchronize(device=device)
    time = begin.elapsed_time(end) / 1000.0  # convert to seconds

    if rank == 0 and log:
        print(f"Memory cost: {memory_cost / 1024 / 1024:.2f} MB")
        print(f"{num_iter / time:.3f} iter/s, {time:.3f} sec")

    return num_iter / time, time, memory_cost
