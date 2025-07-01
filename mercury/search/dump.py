""" in this file we dump the critical information of the search result ir """

from mercury.ir.nodes import BufferLoad, BufferStore, Program
from mercury.ir.utils import collect_axis, get_buffers, get_potential_ring


def dump(program: Program):
    print("Dumping IR:")
    print(f"  Mesh:{program.mesh}")
    print("  Axes:")
    axes = program.visit(collect_axis)
    for axis in axes:
        axis_str = f"    {axis.name}: size={axis.size}, step={axis.min_block_size}"
        if len(axis.ring_comm) > 0:
            axis_str += f", ring_comm={axis.ring_comm}"
        if axis.parallel_info is not None:
            axis_str += f", parallel_info(cards, start_dim, end_dim)={axis.parallel_info}"
        print(axis_str)

    print("  Buffers:")
    buffers = program.visit(get_buffers)
    for buffer in buffers:
        print(f"    {buffer.tensor}: {buffer.shape}, dtype={buffer.dtype}, shard_spec={buffer.shard_spec.specs}")

    print("  Ring communication and all reduces:")
    buffer_commands = program.visit(get_potential_ring)
    for command in buffer_commands:
        if isinstance(command, BufferStore) or isinstance(command, BufferLoad):
            if len(command.comm) > 0:
                print(f"    ring for {command.buffer.tensor}: {command.indices} -> {command.comm}")
        else:
            if len(command.comm) > 0:
                print(f"    ring for reduce{command.buffer.tensor}: {command.axes} -> {command.comm}")
            elif len(command.shard_dim) > 0:
                print(f"    all reduce for {command.buffer.tensor}: {command.axes} -> {command.shard_dim}")

    