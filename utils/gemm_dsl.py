import torch
import torch.distributed as dist
def add_collective(tensor, op, group, dst=None):
    tensor_red = tensor.contiguous()
    if op == dist.all_reduce:
        dist.all_reduce(tensor_red, dist.ReduceOp.SUM, group=group)
    else:
        raise ValueError("not support other op than all reduce")

    tensor = tensor_red

gemm_manage_reduction = """
def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    # Define axes
    I = Axis("I", {M_LEN}, min_block_size=32)  # M dimension
    K = Axis("K", {K_LEN}, min_block_size=32)  # K dimension
    J = Axis("J", {N_LEN}, min_block_size=32)  # N dimension

    # Match buffers
    A = match_buffer(a, [{M_LEN}, {K_LEN}], [I, K])  # [M, K]
    B = match_buffer(b, [{K_LEN}, {N_LEN}], [K, J])  # [K, N]
    C = match_buffer(c, [{M_LEN}, {N_LEN}], [I, J])  # [M, N]

    # Grid pattern shows J is reduction axis
    for i, j in grid([I, J], "ss"):
        reduce_buf = temp_buffer([i, j], [I, J], dtype=torch.float32) # reduce = [stride_i, stride_j]
        for k in grid([K], "m"):
            _a = load_buffer(A[i, k])
            _b = load_buffer(B[k, j])
            _c = torch.matmul(_a, _b)
            block_res = _c.to(torch.float32)
            reduce(op=torch.add,
                   buffer = reduce_buf,
                   collective_op = add_collective,
                   src=block_res,
                   axis=k
            )
        reduce_res = load_buffer(reduce_buf[:, :])
        C[i, j] = store_buffer(reduce_res.to(c.dtype))
"""