# Copyright (c) 2025 PICASSO LAB, Licensed under the MIT License.

import pytest
import torch
from mercury.ir.distributed import DeviceMesh, ShardingSpec, ShardType

def test_device_mesh_creation():
    """basic grid creation test"""
    devices = list(range(8))
    mesh_1d = DeviceMesh(devices=devices, shape=(8,))
    assert len(mesh_1d.devices) == 8
    assert mesh_1d.shape == (8,)
    
    mesh_2d = DeviceMesh(devices=devices, shape=(4, 2))
    assert len(mesh_2d.devices) == 8
    assert mesh_2d.shape == (4, 2)
    
    mesh_3d = DeviceMesh(devices=devices, shape=(2, 2, 2))
    assert len(mesh_3d.devices) == 8
    assert mesh_3d.shape == (2, 2, 2)

def test_device_mesh_reshape():
    """test reshaping of device mesh"""
    devices = list(range(8))

    mesh_1d = DeviceMesh(devices, shape=(8,))
    assert mesh_1d.shape == (8,)
    
    mesh_2d = mesh_1d.reshape(shape=(4, 2))
    assert mesh_2d.shape == (4, 2)
    
    mesh_3d = mesh_1d.reshape(shape=(2, 2, 2))
    assert mesh_3d.shape == (2, 2, 2)
    
    with pytest.raises(ValueError):
        mesh_1d.reshape(shape=(4, 3))

def test_device_mesh_indexing():
    """test indexing and slicing of device mesh"""
    devices = [0, 1, 2, 3, 4, 5, 6, 7]
    mesh = DeviceMesh(devices, shape=(2, 4))
    
    assert mesh.get_device((0, 0)) == 0
    assert mesh.get_device((0, 1)) == 1
    assert mesh.get_device((1, 0)) == 4
    
    assert mesh.get_slice(dim=0, idx=0) == [0, 1, 2, 3]
    assert mesh.get_slice(dim=1, idx=0) == [0, 4]

def test_device_mesh_operations():
    """test operations on device mesh"""
    devices = list(range(8))
    mesh_3d = DeviceMesh(devices, shape=(2, 2, 2))
    
    merged = mesh_3d.merge_dims(1, 2)
    assert merged.shape == (2, 4)

def test_sharding_spec():
    """test sharding spec creation and validation"""
    devices = list(range(8))
    mesh = DeviceMesh(devices, shape=(4, 2))
    
    spec1 = ShardingSpec(
        mesh=mesh,
        specs=[(ShardType.SHARD, [0]), ShardType.REPLICATE]
    )
    assert len(spec1.specs) == 2
    
    spec2 = ShardingSpec(
        mesh=mesh,
        specs=[ShardType.REPLICATE, (ShardType.SHARD, [1])]
    )
    assert len(spec2.specs) == 2
    
    # test invalid sharding spec
    with pytest.raises(ValueError):
        ShardingSpec(
            mesh=mesh,
            specs=[(ShardType.SHARD, [0]), (ShardType.SHARD, [0])]
        )
    
    # test invalid mesh shape
    with pytest.raises(ValueError):
        ShardingSpec(
            mesh=mesh,
            specs=[(ShardType.SHARD, [2]), ShardType.REPLICATE]
        )

if __name__ == '__main__':
    test_device_mesh_creation()
    test_device_mesh_reshape()
    test_device_mesh_indexing()
    test_device_mesh_operations()
    test_sharding_spec()
    print("All tests passed!")