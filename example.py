# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

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
    """
    This function serves as the core example of the Mercury compiler, demonstrating how to
    automatically convert a standard Attention implementation (based on a Flash Attention DSL)
    into a distributed Ring Attention implementation.
    It injects distributed logic through a series of IR (Intermediate Representation) transformations.

    Args:
    - world_size (int): The total number of devices (e.g., GPUs) in the distributed environment.

    Returns:
    - code (str): A string containing the generated PyTorch code for the distributed Ring Attention.

    How to Extend:
    - Replace `flash_attn_pack_kv_template`: You can substitute this with another Python implementation
      of Attention (as a string). As long as the code is well-structured with identifiable loops
      and tensor operations, Mercury can parse and transform it.
    - Adjust `source.format`: Modify parameters like BATCH, HEADS, SEQ_LEN, HEAD_DIM to fit
      different model scales.
    - Modify the parallelization strategy:
        - `axis = axes[2]`: Currently, the partitioning is done along the sequence length axis (S_q).
          You could choose another axis for parallelization, such as the Head axis.
    """

    # Step 1: Get the source code and parse it into Mercury IR.
    # `flash_attn_pack_kv_template` is a Python code template for a standard Flash Attention implementation.
    # `.format(...)` fills in the specific dimensions, making it a complete, parsable function definition.
    #
    # --- How to Extend ---
    # You can change the parameters here (BATCH, HEADS, SEQ_LEN, HEAD_DIM) to test code generation
    # for different scales. Alternatively, replace `flash_attn_pack_kv_template` with a string of your
    # own Attention implementation.
    source = flash_attn_pack_kv_template.format(BATCH=4, HEADS=5, SEQ_LEN=4096, HEAD_DIM=128)
    # `ast.parse` converts the Python source code string into an Abstract Syntax Tree (AST).
    tree = ast.parse(textwrap.dedent(source))
    # `IRBuilder` is a core component of Mercury that transforms the Python AST into Mercury's
    # custom Intermediate Representation (IR).
    builder = IRBuilder()
    # `builder.visit` traverses the AST to generate a structured Program IR object, which contains
    # all the information about the computation graph.
    program = builder.visit(tree.body[0])

    # Step 2: Extract key information (axes and loops) from the IR.
    # `collect_axis` and `collect_loops` are IR visitors used to extract all Axis and Loop
    # information from the Program IR. This is crucial for the subsequent parallelization
    # transformations, as we need to specify which loops and tensor axes to operate on.
    axes = program.visit(collect_axis)
    loops = program.visit(collect_loops)

    # Step 3: Define the parallelization strategy.
    # Here, we select the axis and loop for parallelization. `axes[2]` corresponds to the
    # sequence length axis (S_q), and `loops[0]` is the outermost loop.
    # This is the key to implementing Ring Attention: we partition the computation along the sequence length.
    axis = axes[2] # S_q
    outer_loop = loops[0]

    # Define the Device Mesh, which describes the physical topology of the distributed computation.
    # Here, we create a one-dimensional device mesh of size `world_size`.
    devices = list(range(world_size))
    mesh = DeviceMesh(devices, (world_size,))

    # Step 4: Apply distributed transformations to the IR.
    # `init_distributed` initializes the distributed information, associating the device mesh with the Program IR.
    init_distributed(program, mesh)

    # `parallelize` is a core IR transformation. It partitions tensors and computations across
    # different devices according to the specified axis and device mesh.
    # Here, we partition the Q tensor along the sequence axis `axis`.
    parallelize(program, outer_loop, axis, mesh, 0, len(mesh.shape))

    # `shift` is another core IR transformation used to implement Ring-style communication.
    # It inserts communication operations (like send/recv) into the computation graph,
    # enabling each device to receive data (the KV block, in this case) from its neighbor.
    shift(program, axis, mesh, 0, len(mesh.shape), 1)

    # `eliminate_loops` is an optimization step that removes loops that become redundant after
    # the distributed transformations, simplifying the computation graph.
    eliminate_loops(program)

    # Step 5: Generate the target code from the transformed IR.
    # `generate_pytorch_code` compiles the distributed and optimized IR back into PyTorch code.
    code = generate_pytorch_code(program)

    return code

def run_func():
    """
    This function is responsible for setting up the distributed environment,
    running the code generated by `get_ring_attn`, and comparing its output
    with a standard Flash Attention implementation to verify its correctness.
    """
    # Initialize the PyTorch distributed process group.
    dist.init_process_group("nccl")
    # Get the rank of the current process and the total world size.
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # --- Key Variable Definitions ---
    # Define model parameters and data types.
    # --- How to Extend ---
    # You can modify these parameters to test performance and correctness with different configurations.
    # Note: These parameters should be consistent with the values in `source.format` within
    # `get_ring_attn` to ensure logical correctness.
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{rank}")
    batch_size = 4
    seqlen = 4096
    nheads = 5
    d = 128 # head_dim
    dropout_p = 0
    causal = False
    deterministic = False

    # Ensure the sequence length is divisible by the world size, a basic requirement for Ring Attention.
    assert seqlen % world_size == 0
    assert d % 8 == 0

    # Create the full QKV tensor and broadcast it to all devices to ensure consistent initial data.
    qkv = torch.randn(
        batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dist.broadcast(qkv, src=0)

    # Call the core function to generate the PyTorch code for Ring Attention.
    code = get_ring_attn(world_size)
    # On rank 0, print the generated code for debugging and inspection.
    if rank == 0:
        print("#" * 30)
        print("# generated code:")
        print("#" * 30)
        print(code)
        print("#" * 30)
        print("# run generated code:")
        print("#" * 30)

    # Use `exec` to dynamically execute the generated code string, loading the defined function
    # into the current namespace.
    namespace = globals()
    exec(code, namespace)
    # Get the function handle from the namespace.
    func = namespace["flash_attn_pack_kv"]

    # --- Prepare Local Data ---
    # Each rank processes only a portion of the full QKV tensor.
    # The `chunk` operation simulates data parallelism by splitting the tensor along the sequence length dimension.
    local_qkv = qkv.chunk(world_size, dim=1)[rank].detach().clone()

    # Synchronize all processes to ensure data is ready.
    dist.barrier()

    # --- Run the Baseline (Standard Flash Attention) ---
    # Run the original, non-distributed Flash Attention on the full QKV tensor to serve as a
    # ground truth for correctness verification.
    out, lse, _ = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
        return_attn_probs=True,
    )

    # Extract the portion of the baseline result that corresponds to the current rank.
    local_out = out.chunk(world_size, dim=1)[rank]
    local_lse = lse.chunk(world_size, dim=-1)[rank]

    # --- Run the Generated Ring Attention Code ---
    # Prepare the inputs for the generated distributed function.
    local_q, local_kv = local_qkv[:, :, 0], local_qkv[:, :, 1:3]
    local_q, local_kv = local_q.contiguous(), local_kv.permute(2, 0, 1, 3, 4).contiguous()
    # Initialize tensors to store the results of the Ring Attention.
    ring_out = torch.zeros_like(local_q)
    ring_lse = torch.zeros(batch_size, nheads, seqlen // world_size, device=device, dtype=torch.float32)

    # Call the distributed function generated by Mercury.
    func(
        local_q,
        local_kv,
        ring_out,
        ring_lse,
        local_q.shape[-1] ** (-0.5), # scale
        dropout_p=dropout_p,
        causal=causal,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=deterministic,
    )

    # --- Verify the Results ---
    # Print the difference between the baseline result and the Ring Attention result.
    # A very small difference (close to zero) indicates that the generated distributed code is correct.
    log("out", out, rank0_only=True)
    log("lse", lse, rank0_only=True)
    log("out diff", local_out - ring_out)
    log("lse diff", local_lse - ring_lse)

    # Synchronize all processes to ensure all computations and logging are complete.
    dist.barrier()

if __name__ == '__main__':
    # Entry point of the program.
    run_func()