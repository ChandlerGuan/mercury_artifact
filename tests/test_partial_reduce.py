import torch.distributed as dist
from mercury.backend.pytorch.utils import get_device_group, one_dim_to_n_dim

def test_get_device_group(rank):
    
    mesh_shape = (2, 2, 2)
    
    indices = one_dim_to_n_dim(rank, mesh_shape)
    
    group2 = get_device_group(indices, mesh_shape, tuple(range(0,1)), True)
    print(f"[Rank {rank}] group2:", group2)
    
    group3 = get_device_group(indices, mesh_shape, tuple(range(0,2)), True)
    print(f"[Rank {rank}] group3:", group3)

    group4 = get_device_group(indices, mesh_shape, tuple(range(1,2)), True)
    print(f"[Rank {rank}] group4:", group4)

    group5 = get_device_group(indices, mesh_shape, tuple(range(2,3)), True)
    print(f"[Rank {rank}] group5:", group5)
    
    group6 = get_device_group(indices, mesh_shape, tuple(range(1,3)), True)
    print(f"[Rank {rank}] group6:", group6)

    print(f"[Rank {rank}] Test completed")

if __name__ == "__main__":
    for rank in range(8):
        test_get_device_group(rank)