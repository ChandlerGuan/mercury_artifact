# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

import ast
import random
import textwrap
import torch
import torch.distributed as dist
from mercury.backend.pytorch.codegen import generate_pytorch_code
from mercury.frontend.parser import IRBuilder
from mercury.ir.distributed import DeviceMesh
from mercury.search.search import search
from mercury.search.bench import bench_program
from flash_attn.flash_attn_interface import _flash_attn_forward
from utils.flash_attn_dsl import *

def run_search_bench(source: str):
    """benchmark search"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    dropout_p = 0
    causal = False
    deterministic = False
    num_iter = 100

    namespace = globals()
    
    tree = ast.parse(textwrap.dedent(source))
    builder = IRBuilder()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            program = builder.visit(node)
            break
    else:
        raise RuntimeError("Could not find function definition")
    
    devices = [i for i in range(world_size)]
    mesh = DeviceMesh(devices, (world_size,))
    searched_programs = list(search(program, mesh, ["S_q", "S_kv"]))

    # for some unkonwn reason, the search result's order is not stable across different devices
    searched_programs.sort(key=lambda x: generate_pytorch_code(x))

    # benchmark
    for idx, searched_program in enumerate(searched_programs):
        if rank == 0:
            print(f"\nprogram {idx + 1}")
        
        kwargs = {
            "softmax_scale": random.random() * 0.5 + 0.5,
            "dropout_p": dropout_p,
            "causal": causal,
            "window_size": (-1, -1),
            "alibi_slopes": None,
            "deterministic": deterministic,
        }
        
        iter_per_sec, total_time, memory = bench_program(
            program=searched_program,
            namespace=namespace,
            num_iter=num_iter,
            kwargs=kwargs,
            log=False
        )

        if rank == 0:
            print(f"program {idx + 1} performance:")
            print(f"iter per second: {iter_per_sec:.2f}")
            print(f"time: {total_time:.3f} s")
            print(f"memory: {memory / 1024 / 1024:.2f} MB")
        
        torch.cuda.empty_cache()
        dist.barrier()

def search_on_code():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    batch_size = 4
    seqlen = 4096 * 8
    nheads = 32
    head_dim = 128
    
    # single ring with reduction
    source = flash_attn_manage_reduction.format(
        BATCH=batch_size,
        SEQ_LEN=seqlen,
        HEADS=nheads,
        HEAD_DIM=head_dim,
        RED_DIM=head_dim + 1,
    )
    run_search_bench(source)


if __name__ == "__main__":
    search_on_code()
