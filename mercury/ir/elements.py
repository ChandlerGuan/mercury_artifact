# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

"""
Axis and buffer binding definitions for auto-ring compiler.
"""
from typing import Any, Callable, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch.distributed
from .distributed import ShardingSpec
import copy
import torch

@dataclass
class Axis:
    """Represents a loop iteration axis with optional blocking."""
    name: str
    size: int
    min_block_size: int = 1
    max_block_size: Optional[int] = None # when the axis is 'ringed', max block size will reduce to the size of data on each card
    ring_comm: List[str] = field(default_factory=list)
    ring_comm_cards: int = 1
    parallel_info: Optional[Any] = None

    def __post_init__(self):
        if self.max_block_size is None:
            self.max_block_size = self.size
        if self.min_block_size <= 0:
            raise ValueError(f"Block size must be positive, got {self.min_block_size}")
        if self.min_block_size > self.size:
            raise ValueError(f"Block size {self.min_block_size} exceeds axis size {self.size}")
        
    def __str__(self) -> str:
        return f"Axis(name={self.name}, size={self.size}, min_block_size={self.min_block_size}, max_block_size={self.max_block_size}, ring_comm={self.ring_comm}, ring_comm_cards={self.ring_comm_cards})"

    @property
    def num_blocks(self) -> int:
        """Number of blocks if blocked, or 1 if not blocked."""
        if self.min_block_size is None:
            return 1
        return (self.size + self.min_block_size - 1) // self.min_block_size

    def __deepcopy__(self, memo: dict) -> 'Axis':
        """Customize deep copy behavior."""
        if id(self) in memo:
            return memo[id(self)]
            
        result = Axis(
            name=self.name,
            size=self.size, 
            min_block_size=self.min_block_size,
            max_block_size=self.max_block_size,
            ring_comm=copy.deepcopy(self.ring_comm, memo),
            ring_comm_cards=self.ring_comm_cards,
            parallel_info=self.parallel_info)
        memo[id(self)] = result
        return result

class GridIterator:
    """Iterator for a set of axes with reduction marking."""
    def __init__(self, axes: List[Axis], axis_types: str):
        """
        Args:
            axes: List of axes to iterate over
            axis_types: String of s/r marking each axis as spatial/reduction
                e.g. "ssrs" for 4 axes where the third is reduction
        """
        if len(axes) != len(axis_types):
            raise ValueError(f"Got {len(axes)} axes but {len(axis_types)} type markers")
        if not all(t in "sr" for t in axis_types):
            raise ValueError(f"Axis types must be 's' or 'r', got {axis_types}")
            
        self.axes = axes
        self.axis_types = axis_types
        self._reduction_axes = [i for i, t in enumerate(axis_types) if t == 'r']

    def __iter__(self):
        # Get ranges for each axis based on blocking
        ranges = [range(ax.num_blocks) for ax in self.axes]
        
        # Return indices tuple for each combination
        from itertools import product
        return product(*ranges)

    @property
    def reduction_axes(self) -> List[int]:
        """Return indices of reduction axes."""
        return self._reduction_axes

def grid(axes: List[Axis], axis_types: str) -> GridIterator:
    """Create a grid iterator over the given axes.
    
    Args:
        axes: List of axes to iterate over
        axis_types: String of s/r marking each axis as spatial/reduction
    
    Returns:
        GridIterator for use in for loops
    """
    return GridIterator(axes, axis_types)

@dataclass
class Buffer:
    """A tensor buffer with axis binding information."""
    tensor: str
    shape: List[Union[int, Axis]] # int for fixed size, Axis for dynamic size according to the axis stride
    bound_axes: List[List[Axis]]  # None for unbound dimensions
    axes_factor: List[List[int]] # factor for each axis in each dimension
    shard_spec: Optional[ShardingSpec] = None
    read: bool = False
    write: bool = False
    dtype: torch.dtype = torch.bfloat16
    def_axis: Optional[Axis] = None # the loop axis that the buffer is defined in, for temporary buffer

    def get_shape(self) -> List[int]:
        return [int(dim.min_block_size) if isinstance(dim, Axis) else int(dim) for dim in self.shape]

    def __str__(self) -> str:
        axis_name = [[axis.name for axis in axes] if len(axes) > 0 else None for axes in self.bound_axes]
        return f"Buffer(name = {self.tensor}, shape={self.shape}, axes={axis_name}, shard_spec={self.shard_spec} , read={self.read}, write={self.write}, dtype={self.dtype})"
        
    def __deepcopy__(self, memo: dict) -> 'Buffer':
        """Customize deep copy behavior."""
        if id(self) in memo:
            return memo[id(self)]
            
        result = Buffer(
            tensor=self.tensor,
            shape=copy.deepcopy(self.shape, memo),
            bound_axes=[[copy.deepcopy(axis, memo) for axis in axes] 
                       for axes in self.bound_axes],
            axes_factor=copy.deepcopy(self.axes_factor, memo),
            shard_spec=copy.deepcopy(self.shard_spec, memo) if self.shard_spec else None,
            read=self.read,
            write=self.write,
            dtype=self.dtype)
        memo[id(self)] = result
        return result

    def get_axis(self, target_axis: Axis) -> Tuple[int, int]:
        for i, axes_dim in enumerate(self.bound_axes):
            for j, axis in enumerate(axes_dim):
                if axis == target_axis:
                    return i, j
        raise ValueError("axis not bound to buffer")
    
    def has_axis(self, target_axis: Axis) -> bool:
        for axes_dim in self.bound_axes:
            for axis in axes_dim:
                if axis == target_axis:
                    return True
        return False
    
    @property
    def ndim(self) -> int:
        return len(self.shape)

def match_buffer(tensor: torch.Tensor, shape: List[int], 
                bound_axes: List[Optional[Union[Axis, Tuple[Axis, ...]]]], dtype: torch.dtype = torch.bfloat16) -> Buffer:
    """Create a Buffer from a tensor with axis bindings.
    
    Args:
        tensor: Input tensor
        shape: Expected shape
        bound_axes: List of axes to bind to each dimension (None for unbound)
        
    Returns:
        Buffer with axis bindings
    """
    pass
    # if tensor.shape != shape:
    #     raise ValueError(f"Tensor shape {tensor.shape} doesn't match expected {shape}")
    # if len(bound_axes) != len(shape):
    #     raise ValueError(f"Got {len(bound_axes)} axes for {len(shape)} dimensions")
        
    # # Verify bound axes match corresponding dimension sizes
    # for dim, (size, axis) in enumerate(zip(shape, bound_axes)):
    #     if axis is not None and axis.size != size:
    #         raise ValueError(
    #             f"Axis {axis.name} size {axis.size} doesn't match dim {dim} size {size}")
            
    # return Buffer(tensor, shape, bound_axes, dtype=dtype)

def load_buffer(buffer) -> torch.Tensor:
    """Load from a buffer with given indices.
    
    Args:
        buffer: Buffer to load from
        indices: List of indices or axes to load from
        
    Returns:
        Tensor slice from the buffer
    """
    pass
    # if len(indices) != buffer.ndim:
    #     raise ValueError(f"Got {len(indices)} indices for {buffer.ndim} dimensions")
        
    # # Convert axes to indices
    # final_indices = []
    # for idx, axis in zip(indices, buffer.bound_axes):
    #     if isinstance(idx, Axis):
    #         if axis is None:
    #             raise ValueError(f"Axis {idx.name} not bound in buffer")
    #         idx = idx.name
    #     final_indices.append(idx)
        
    # return buffer[final_indices]

def store_buffer(target: torch.Tensor) -> Buffer:
    pass

def temp_buffer(shape, bound_axes, dtype=torch.bfloat16) -> Buffer:
    pass

def reduce(target: Buffer, source: torch.Tensor, axis: Axis, op: Callable, collective_op: Callable):
    pass