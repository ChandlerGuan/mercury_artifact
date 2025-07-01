import json
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from mercury.ir.distributed import DeviceMesh, ShardingSpec, ShardType
from mercury.ir.nodes import Program
from mercury.ir.elements import Buffer

@dataclass
class BenchmarkResult:
    """dataclass for storing benchmark results."""
    program_id: int
    execution_time: float
    memory_usage: float
    input_buffers: List[Buffer]
    output_buffers: List[Buffer]
    mesh_info: DeviceMesh

    def to_dict(self) -> Dict:
        """
        transform the benchmark result into a dictionary format.
        """
        return {
            "program_id": self.program_id,
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "mesh_info": self.mesh_info.shape,
            "input_buffers": [self._buffer_to_dict(buf) for buf in self.input_buffers],
            "output_buffers": [self._buffer_to_dict(buf) for buf in self.output_buffers]
        }

    @staticmethod
    def _buffer_to_dict(buffer: Buffer) -> Dict:
        """transform a buffer into a dictionary format."""
        specs = []
        for spec in buffer.shard_spec.specs:
            if isinstance(spec, tuple):
                shard_type, dims = spec
                specs.append([str(shard_type), dims])
            else:
                specs.append(str(spec))

        return {
            "tensor_name": buffer.tensor,
            "shape": buffer.shape,
            "specs": specs,
            "dtype": str(buffer.dtype),
        }

@dataclass
class ShardingGroup:
    """
    sharding group for storing benchmark results. Each group contains
    best result of the same sharding pattern.
    """
    group_id: str
    input_buffers: List[Buffer]
    output_buffers: List[Buffer]
    mesh_shape: List[int]
    best_result: Optional[BenchmarkResult] = None

    def update_best_result(self, result: BenchmarkResult) -> bool:
        """
        update the best result of the group.
        Args:
            result: new benchmark result
        Returns:
            bool: return True if the best result is updated, otherwise False
        """
        if self.best_result is None or result.execution_time < self.best_result.execution_time:
            self.best_result = result
            return True
        return False

    def to_dict(self) -> Dict:
        """turn sharding group into a dictionary format"""
        return {
            "group_id": self.group_id,
            "mesh_shape": self.mesh_shape,
            "input_buffers": [BenchmarkResult._buffer_to_dict(buf) for buf in self.input_buffers],
            "output_buffers": [BenchmarkResult._buffer_to_dict(buf) for buf in self.output_buffers],
            "best_result": self.best_result.to_dict() if self.best_result else None
        }

class BenchmarkCollector:
    """benchmark result collector."""
    def __init__(self):
        self.groups: Dict[str, ShardingGroup] = {}

    def add_result(self, result: BenchmarkResult):
        """
        add a benchmark result to the collector.
        Args:
            result: BenchmarkResult instance
        """
        group_id = self._compute_group_id(result)
        if group_id not in self.groups:
            self.groups[group_id] = ShardingGroup(
                group_id=group_id,
                input_buffers=result.input_buffers,
                output_buffers=result.output_buffers,
                mesh_shape=list(result.mesh_info.shape)
            )
        self.groups[group_id].update_best_result(result)

    def _compute_group_id(self, result: BenchmarkResult) -> str:
        """
        group ID is computed based on the input and output buffers.
        """
        def _buffer_to_str(buffer: Buffer) -> str:
            # transform buffer sharding spec into a string
            specs_str = []
            for spec in buffer.shard_spec.specs:
                if isinstance(spec, tuple):
                    shard_type, dims = spec
                    specs_str.append(f"{shard_type}:{','.join(map(str, dims))}")
                else:
                    specs_str.append(str(spec))
            return f"{buffer.tensor}|{','.join(specs_str)}"

        # concatenate all input and output buffer specs
        input_str = "+".join(_buffer_to_str(buf) for buf in result.input_buffers)
        output_str = "+".join(_buffer_to_str(buf) for buf in result.output_buffers)
        
        return f"{input_str}#{output_str}"

    def to_dict(self) -> Dict:
        """turn the collector into a dictionary format"""
        return {
            "metadata": {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
            },
            "sharding_groups": [
                group.to_dict() for group in self.groups.values()
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BenchmarkCollector':
        """
        load data from a dictionary format.
        Args:
            data: dictionary containing collector data
        Returns:
            BenchmarkCollector: new collector instance
        """
        collector = cls()
        
        for group_data in data["sharding_groups"]:
            # build DeviceMesh
            mesh = DeviceMesh(
                devices=list(range(np.prod(group_data["mesh_shape"]))),
                shape=tuple(group_data["mesh_shape"])
            )

            def _buffer_from_dict(buf_data: Dict) -> Buffer:
                # transform buffer data into a Buffer instance
                specs = []
                for spec in buf_data["specs"]:
                    if isinstance(spec, list):
                        specs.append((ShardType.SHARD, spec[1]))
                    else:
                        specs.append(ShardType.REPLICATE)

                dtype = eval(buf_data["dtype"])
                
                return Buffer(
                    tensor=buf_data["tensor_name"],
                    shape=buf_data["shape"],
                    bound_axes=[],
                    dtype=dtype,
                    axes_factor=[],
                    shard_spec=ShardingSpec(mesh, specs)
                )
            
            # rebuild input buffers
            input_buffers = []
            for buf_data in group_data["input_buffers"]:
                buffer = _buffer_from_dict(buf_data)
                buffer.read = True
                input_buffers.append(buffer)
            
            # rebuild output buffers
            output_buffers = []
            for buf_data in group_data["output_buffers"]:
                buffer = _buffer_from_dict(buf_data)
                buffer.write = True
                output_buffers.append(buffer)
            
            # rebuild group
            group = ShardingGroup(
                group_id=group_data["group_id"],
                input_buffers=input_buffers,
                output_buffers=output_buffers,
                mesh_shape=group_data["mesh_shape"]
            )
            
            # if there is best result, rebuild it
            if group_data["best_result"]:
                best_data = group_data["best_result"]
                group.best_result = BenchmarkResult(
                    program_id=best_data["program_id"],
                    execution_time=best_data["execution_time"],
                    memory_usage=best_data["memory_usage"],
                    input_buffers=input_buffers,
                    output_buffers=output_buffers,
                    mesh_info=mesh
                )
            
            collector.groups[group.group_id] = group
            
        return collector