# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""
auto-ring: A compiler for auto-parallelizing tensor computations.
"""
from .ir.elements import Axis, Buffer, grid, match_buffer
from .frontend.parser import auto_schedule
from .ir.nodes import BufferStore, BufferLoad
from .backend import *
from .search import *

__version__ = "0.1.0"
__all__ = [
    "Axis",
    "Buffer",
    "grid",
    "match_buffer",
    "auto_schedule",
    "store_buffer",
    "load_buffer",
    "BufferStore",
    "BufferLoad",
    "RingComm"
]