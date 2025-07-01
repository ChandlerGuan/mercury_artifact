from functools import cache
import inspect
import torch
import torch.distributed as dist
from mercury.ir.elements import Axis, grid, match_buffer, load_buffer, store_buffer, temp_buffer, reduce
from flash_attn.flash_attn_interface import _flash_attn_backward
from utils.flash_attn_dsl import get_default_args

@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if "softcap" in args:
        args["softcap"] = 0.0
    return args

def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        # Use the origin _init_fn in CustomOpDef
        return _get_default_args(func._init_fn)

def add_collective(tensor, op, group, dst=None):
    tensor_red = tensor.contiguous()
    if op == dist.all_reduce:
        dist.all_reduce(tensor_red, dist.ReduceOp.SUM, group=group)
    else:
        raise ValueError("not support other op than all reduce")

    tensor = tensor_red

attn_backward_template = """
def flash_attn_backward(
    q: torch.Tensor,
    kv: torch.Tensor, 
    o: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    dq: torch.Tensor,
    dkv: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # Define axes
    B = Axis("B", {BATCH})  # Batch dimension
    H = Axis("H", {HEADS})  # Head dimension for K/V
    S_Q = Axis("S_q", {SEQ_LEN}, min_block_size=32)  # Query sequence length
    S_KV = Axis("S_kv", {SEQ_LEN}, min_block_size=32)  # Key/Value sequence length
    
    # Match buffers for inputs
    DOUT = match_buffer(dout, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H, None])
    Q = match_buffer(q, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H, None])
    KV = match_buffer(kv, [2, {BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [None, B, S_KV, H, None])
    O = match_buffer(o, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H, None])
    LSE = match_buffer(lse, [{BATCH}, {HEADS}, {SEQ_LEN}], [B, H, S_Q], dtype=torch.float32)
    
    # Match buffers for gradients
    DQ = match_buffer(dq, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H, None])
    DKV = match_buffer(dkv, [2, {BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [None, B, S_KV, H, None]) 
    
    for b, h in grid([B, H], "ss"):
        reduce_dkv = temp_buffer([2, b, {SEQ_LEN}, h, {HEAD_DIM}], [None, B, S_KV, H, None], dtype=torch.float32) # here S_KV has not been made into a loop, so we can't use its nick name
        for s_q in grid([S_Q], "m"):
            reduce_dq = temp_buffer([b, s_q, h, {HEAD_DIM}], [B, S_Q, H, None], dtype=torch.float32)
            for s_kv in grid([S_KV], "m"):
                _q = load_buffer(Q[b, s_q, h])
                _kv = load_buffer(KV[b, s_kv, h])
                _k = _kv[0]
                _v = _kv[1]
                _o = load_buffer(O[b, s_q, h])
                _lse = load_buffer(LSE[b, h, s_q])
                _dout = load_buffer(DOUT[b, s_q, h])

                block_dq = torch.empty(_q.shape, dtype=q.dtype, device=q.device)
                block_dkv = torch.empty(_kv.shape, dtype=kv.dtype, device=kv.device)

                params = get_default_args(_flash_attn_backward).copy()
                params.update(
                    {{
                        "dout": _dout,
                        "q": _q,
                        "k": _k,
                        "v": _v,
                        "out": _o,
                        "softmax_lse": _lse,
                        "dq": block_dq,
                        "dk": block_dkv[0],
                        "dv": block_dkv[1],
                        "softmax_scale": softmax_scale,
                        "dropout_p": dropout_p, 
                        "causal": False,
                        "deterministic": deterministic,
                        "alibi_slopes": alibi_slopes
                    }}
                )
                if "window_size" in params:
                    params.update({{"window_size": window_size}})
                else:
                    params.update(
                        {{
                            "window_size_left": window_size[0],
                            "window_size_right": window_size[1],
                        }}
                    )

                _flash_attn_backward(**params)

                reduce(op=torch.add,
                   buffer = reduce_dq,
                   collective_op = add_collective,
                   src=block_dq.to(torch.float32),
                   axis=s_kv
                )

                reduce(op=torch.add,
                   buffer = reduce_dkv[:, s_kv],
                   collective_op = add_collective,
                   src=block_dkv.to(torch.float32),
                   axis=s_q
                )
            # Store reduced gradients
            res_dq = load_buffer(reduce_dq[:, :, :, :])
            DQ[b, s_q, h] = store_buffer(res_dq.to(dq.dtype))
        res_dkv = load_buffer(reduce_dkv[:, :, :, :])
        DKV[b, : , h] = store_buffer(res_dkv.to(dkv.dtype))
"""
