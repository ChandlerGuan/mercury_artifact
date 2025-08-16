# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

from mercury.ir.distributed import DeviceMesh, ShardType, ShardingSpec
from mercury.ir.nodes import Program
from mercury.ir.utils import get_buffers


def init_distributed(program: Program, mesh: DeviceMesh):
    """Initialize distributed IR with device mesh.

    Args:
        program: IR program
        mesh: device mesh
    """
    # Add device mesh to program
    program.mesh = mesh

    # Initialize distributed buffers
    buffers = program.visit(get_buffers)
    for buffer in buffers:
        buffer.shard_spec = ShardingSpec(mesh, [ShardType.REPLICATE] * len(buffer.shape))