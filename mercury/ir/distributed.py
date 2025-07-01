"""
device grid and sharding
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Dict, Optional
from enum import Enum
import copy
import numpy as np

class ShardType(str, Enum):
    """shard type enum"""
    REPLICATE = 'R'  # replicate
    SHARD = 'S'      # shard

    def __str__(self):
        if self == ShardType.REPLICATE:
            return 'R'
        elif self == ShardType.SHARD:
            return 'S'
        else:
            raise ValueError(f"unknown shard type: {self}")

@dataclass
class DeviceMesh:
    """N-dimension device mesh

    supports arbitrary dimensional device organization, e.g.:
    - 1D: [8] means 8 devices
    - 2D: [4, 2] means a 4x2 device grid
    - 3D: [2, 2, 2] means a 2x2x2 device cube

    Currently devices must be [0, 1, 2, ...]
    """
    devices: List[int]  # device ID list
    shape: Tuple[int, ...]  # N-dimensional shape of the device grid

    def __post_init__(self):
        if len(self.devices) != np.prod(self.shape):
            raise ValueError(f"device number {len(self.devices)} not match with {self.shape}")
        
        # reshape the device list into an N-dimensional array for easy indexing
        self._device_grid = np.array(self.devices).reshape(self.shape)
    
    def __deepcopy__(self, memo: dict) -> 'DeviceMesh':
        """Customize deep copy behavior."""
        if id(self) in memo:
            return memo[id(self)]
            
        result = DeviceMesh(
            devices=copy.deepcopy(self.devices, memo),
            shape=self.shape)
        result._device_grid = np.copy(self._device_grid)
        memo[id(self)] = result
        return result
    
    def reshape(self, shape: Tuple[int, ...]) -> 'DeviceMesh':
        """
        reshape the device list into an N-dimensional grid
        Args:
            shape: target grid shape
        Returns:
            reshaped device grid
        Raises:
            ValueError: if the number of devices does not match the grid shape
        """
        if np.prod(shape) != len(self.devices):
            raise ValueError(f"device num {len(self.devices)} not match with grid shape {shape}")
        return DeviceMesh(devices=self.devices, shape=shape)

    def get_device(self, coords: Tuple[int, ...]) -> int:
        """
        get device ID for the given coordinates
        Args:
            coords: device coordinates, must match the grid dimensions
        Returns:
            device ID
        Raises:
            ValueError: if the coordinate dimensions do not match the grid dimensions
        """
        if len(coords) != len(self.shape):
            raise ValueError(f"cord dim {len(coords)} not match with grid dim {len(self.shape)}")
        return self._device_grid[coords]

    def get_slice(self, dim: int, idx: int) -> List[int]:
        """
        get device slice for the given dimension and index
        Args:
            dim: dimension index
            idx: index in that dimension
        Returns:
            list of device IDs
        """
        slicing = [slice(None)] * len(self.shape)
        slicing[dim] = idx
        return self._device_grid[tuple(slicing)].tolist()

    def merge_dims(self, start: int, end: int) -> 'DeviceMesh':
        """merge dimensions from start to end
        Args:
            start: start dim index
            end: end dim index
        Returns:
            merged device mesh
        """
        if start < 0 or end >= len(self.shape) or start > end:
            raise ValueError(f"invalid index: start={start}, end={end}, shape={self.shape}")
    
        # compute the new shape: keep the dimensions before start and after end,
        # and merge the dimensions from start to end into one dimension
        merged_dim_size = np.prod(self.shape[start:end+1])
        new_shape = self.shape[:start] + (merged_dim_size,) + self.shape[end+1:]
        
        return DeviceMesh(devices=self.devices, shape=new_shape)

    def all_coords(self) -> List[Tuple[int, ...]]:
        """
        get all device coordinates in the mesh
        Args:
            shape: target grid shape
        Returns:
            list of device coordinates, each coordinate is a tuple
        """
        return list(np.ndindex(*self.shape))

@dataclass
class ShardingSpec:
    """
    ShardingSpec represents the distribution of a tensor across a device mesh.
    It contains a mesh and a list of specifications for each dimension of the tensor.
    The specifications can be either:
    - ShardType.REPLICATE: the tensor is replicated across all devices in that dimension
    - (ShardType.SHARD, [mesh_dim, ...]): the tensor is sharded across the specified mesh dimensions.
    
    e.g.:
    - [R, R]: the tensor is replicated across all devices
    - [(S, [1]), R]: the tensor is sharded in dim 0 across the second mesh dimension and replicated across the rest
    """
    mesh: DeviceMesh
    specs: List[Union[ShardType, Tuple[ShardType, List[int]]]]

    def __str__(self) -> str:
        spec_str_list = []
        for spec in self.specs:
            if isinstance(spec, tuple):
                shard_type, mesh_dims_and_sizes = spec
                spec_str_list.append(f"({shard_type}, {mesh_dims_and_sizes})")
            else:
                spec_str_list.append(str(spec))
        spec_str = ", ".join(spec_str_list)
        return f"ShardingSpec(mesh={self.mesh}, specs={spec_str})"

    def __post_init__(self):
        self.validate()

    def __deepcopy__(self, memo: dict) -> 'ShardingSpec':
        """Customize deep copy behavior."""
        if id(self) in memo:
            return memo[id(self)]
            
        result = ShardingSpec(
            mesh=copy.deepcopy(self.mesh, memo),
            specs=[copy.deepcopy(spec, memo) if isinstance(spec, tuple)
                  else spec for spec in self.specs])
        memo[id(self)] = result
        return result

    def validate(self):
        """validate the sharding spec"""
        mesh_dim_used = [False] * len(self.mesh.shape)
        
        for spec in self.specs:
            if isinstance(spec, tuple):
                shard_type, mesh_dims_and_sizes = spec
                if shard_type != ShardType.SHARD:
                    raise ValueError(f"invalid shard type: {shard_type}")
                
                if not mesh_dims_and_sizes:
                    raise ValueError("mesh dimensions and sizes cannot be empty")
                    
                # check the legality of each mesh dimension
                for mesh_dim in mesh_dims_and_sizes:
                    if mesh_dim >= len(self.mesh.shape):
                        raise ValueError(f"mesh dim index{mesh_dim} out of range")
                    if mesh_dim_used[mesh_dim]:
                        raise ValueError(f"mesh dim {mesh_dim} already used")
                    mesh_dim_used[mesh_dim] = True
                    
            elif spec != ShardType.REPLICATE:
                raise ValueError(f"invaild spec: {spec}")
                
    def get_shard_info(self) -> Dict[int, Optional[List[int]]]:
        """
        get the mapping from tensor dimensions to device mesh dimensions
        Returns:
            Dict[int, Optional[List[int]]]
            when the value is None, it means the dimension is replicated
        """
        shard_info = {}
        for dim, spec in enumerate(self.specs):
            if isinstance(spec, tuple):
                _, mesh_dims = spec
                shard_info[dim] = mesh_dims
            else:
                shard_info[dim] = None
        return shard_info

    def fully_sharded(self) -> bool:
        """ check whether one buffer is fully sharded acrosss the mesh """
        remain_dims = len(self.mesh.shape)
        for spec in self.specs:
            if isinstance(spec, tuple) and spec[0] == ShardType.SHARD:
                remain_dims -= len(spec[1])
        assert remain_dims >= 0, "remaining dims must >= 0"
        return remain_dims == 0
        