import torch
import torch.distributed as dist
from mercury.ir.elements import Axis, grid, match_buffer, load_buffer, store_buffer# , reduce
from flash_attn.flash_attn_interface import _flash_attn_forward

from functools import cache
from typing import Callable, List, Optional, Tuple
import torch.nn.functional as F
import inspect


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

@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse

def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse

def flash_attn(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # Define axes
    B = Axis("B", 4)  # Batch dimension
    H = Axis("H", 5)  # Head dimension
    S_Q = Axis("S_q", 4096, min_block_size=32)  # Query sequence length
    S_KV = Axis("S_kv", 4096, min_block_size=32)  # Key/Value sequence length

    # Match buffers
    Q = match_buffer(q, [4, 4096, 5, 128], [B, S_Q, H, None])
    K = match_buffer(k, [4, 4096, 5, 128], [B, S_KV, H, None])
    V = match_buffer(v, [4, 4096, 5, 128], [B, S_KV, H, None])
    O = match_buffer(o, [4, 4096, 5, 128], [B, S_Q, H, None])
    LSE = match_buffer(lse, [4, 5, 4096], [B, H, S_Q], dtype=torch.float32)
    for b, h, s_q in grid([B, H, S_Q], "sss"):
        _out = None
        _lse = None
        for s_kv in grid([S_KV], "r"):
            _q = load_buffer(Q[b, s_q, h])
            _k = load_buffer(K[b, s_kv, h])
            _v = load_buffer(V[b, s_kv, h])
            params = get_default_args(_flash_attn_forward).copy()
            params.update(
                {
                    "q": _q,
                    "k": _k,
                    "v": _v,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }
            )
            outputs = _flash_attn_forward(**params)
            assert len(outputs) == 4
            block_out, block_lse, _, _ = outputs
            _out, _lse = update_out_and_lse(_out, _lse, block_out, block_lse)
        O[b, s_q, h] = store_buffer(_out.to(q.dtype))
        LSE[b, h, s_q] = store_buffer(_lse.squeeze(dim=-1).transpose(1, 2))

flash_attn_pack_kv_template = """
def flash_attn_pack_kv(
    q: torch.Tensor, 
    kv: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # Define axes
    B = Axis("B", {BATCH})  # Batch dimension
    H = Axis("H", {HEADS})  # Head dimension
    S_Q = Axis("S_q", {SEQ_LEN}, min_block_size=32)  # Query sequence length
    S_KV = Axis("S_kv", {SEQ_LEN}, min_block_size=32)  # Key/Value sequence length

    # Match buffers
    Q = match_buffer(q, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H, None])
    KV = match_buffer(kv, [2, {BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [None, B, S_KV, H, None])
    O = match_buffer(o, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H, None])
    LSE = match_buffer(lse, [{BATCH}, {HEADS}, {SEQ_LEN}], [B, H, S_Q], dtype=torch.float32)
    for b, h, s_q in grid([B, H, S_Q], "sss"):
        _out = None
        _lse = None
        for s_kv in grid([S_KV], "r"):
            _q = load_buffer(Q[b, s_q, h])
            _kv = load_buffer(KV[b, s_kv, h])
            _k = _kv[0]
            _v = _kv[1]
            params = get_default_args(_flash_attn_forward).copy()
            params.update(
                {{
                    "q": _q,
                    "k": _k,
                    "v": _v,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }}
            )
            outputs = _flash_attn_forward(**params)
            assert len(outputs) == 4
            block_out, block_lse, _, _ = outputs
            _out, _lse = update_out_and_lse(_out, _lse, block_out, block_lse)
        O[b, s_q, h] = store_buffer(_out.to(q.dtype))
        LSE[b, h, s_q] = store_buffer(_lse.squeeze(dim=-1).transpose(1, 2))
"""

gqa_pack_kv_template = """
def gqa_pack_kv(
    q: torch.Tensor, 
    kv: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # Define axes
    B = Axis("B", {BATCH})  # Batch dimension
    H = Axis("H", {KV_HEADS})  # Head dimension
    S_Q = Axis("S_q", {SEQ_LEN}, min_block_size=32)  # Query sequence length
    S_KV = Axis("S_kv", {SEQ_LEN}, min_block_size=32)  # Key/Value sequence length

    # Match buffers
    Q = match_buffer(q, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H * {HEADS_PER_GROUP}, None])
    KV = match_buffer(kv, [2, {BATCH}, {SEQ_LEN}, {KV_HEADS}, {HEAD_DIM}], [None, B, S_KV, H, None])
    O = match_buffer(o, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H * {HEADS_PER_GROUP}, None])
    LSE = match_buffer(lse, [{BATCH}, {HEADS}, {SEQ_LEN}], [B, H * {HEADS_PER_GROUP}, S_Q ], dtype=torch.float32)
    for b, h, s_q in grid([B, H, S_Q], "sss"):
        _out = None
        _lse = None
        for s_kv in grid([S_KV], "r"):
            _q = load_buffer(Q[b, s_q, h])
            _kv = load_buffer(KV[b, s_kv, h])
            _k = _kv[0]
            _v = _kv[1]
            params = get_default_args(_flash_attn_forward).copy()
            params.update(
                {{
                    "q": _q,
                    "k": _k,
                    "v": _v,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }}
            )
            outputs = _flash_attn_forward(**params)
            assert len(outputs) == 4
            block_out, block_lse, _, _ = outputs
            _out, _lse = update_out_and_lse(_out, _lse, block_out, block_lse)
        O[b, s_q, h] = store_buffer(_out.to(q.dtype))
        LSE[b, h, s_q] = store_buffer(_lse.squeeze(dim=-1).transpose(1, 2))
"""

flash_attn_pack_kv_double_ring_template = """
def flash_attn_pack_split_kv(
    q: torch.Tensor, 
    kv: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # Define axes
    B = Axis("B", {BATCH})  # Batch dimension
    H = Axis("H", {HEADS})  # Head dimension
    S_Q = Axis("q", {SEQ_LEN}, min_block_size=32)  # Query sequence length
    S_KV_OUTER = Axis("kv_outer", {SEQ_LEN_OUT})  # Key/Value sequence length
    S_KV_INNER = Axis("kv_inner", {SEQ_LEN_IN}, min_block_size=32)  # Key/Value sequence length

    # Match buffers
    Q = match_buffer(q, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H, None])
    KV = match_buffer(kv, [2, {BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [None, B, (S_KV_OUTER, S_KV_INNER), H, None])
    O = match_buffer(o, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H, None])
    LSE = match_buffer(lse, [{BATCH}, {HEADS}, {SEQ_LEN}], [B, H, S_Q], dtype=torch.float32)
    for b, h, s_q in grid([B, H, S_Q], "sss"):
        _out = None
        _lse = None
        for s_kv_outer, s_kv_inner in grid([S_KV_OUTER, S_KV_INNER], "rr"):
            _q = load_buffer(Q[b, s_q, h])
            _kv = load_buffer(KV[b, (s_kv_outer, s_kv_inner), h])
            _k = _kv[0]
            _v = _kv[1]
            params = get_default_args(_flash_attn_forward).copy()
            params.update(
                {{
                    "q": _q,
                    "k": _k,
                    "v": _v,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }}
            )
            outputs = _flash_attn_forward(**params)
            assert len(outputs) == 4
            block_out, block_lse, _, _ = outputs
            _out, _lse = update_out_and_lse(_out, _lse, block_out, block_lse)
        O[b, s_q, h] = store_buffer(_out.to(q.dtype))
        LSE[b, h, s_q] = store_buffer(_lse.squeeze(dim=-1).transpose(1, 2))
"""

@torch.jit.script
def flash_reduce(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    
    l_out, l_lse = lhs[..., :-1], lhs[..., -1:]
    r_out, r_lse = rhs[..., :-1], rhs[..., -1:]

    out = l_out - F.sigmoid(r_lse - l_lse) * (l_out - r_out)
    lse = l_lse - F.logsigmoid(l_lse - r_lse)

    return torch.cat([out, lse], dim=-1)

def update_out_and_lse_collective(tensor, op: Callable, group, dst=None):

    if op == dist.reduce or op == dist.all_reduce:
        local_out, local_lse = tensor[..., :-1], tensor[..., -1:]
        max_lse = local_lse.clone().contiguous()
        # the way in tree attention
        dist.all_reduce(max_lse, dist.ReduceOp.MAX, group=group)
        d = torch.exp(local_lse - max_lse).contiguous()
        sum_out = (local_out * d).contiguous()

        if op == dist.reduce:
            dist.reduce(sum_out, dst, dist.ReduceOp.SUM, group=group)
            dist.reduce(d, dst, dist.ReduceOp.SUM, group=group)
        elif op == dist.all_reduce:
            dist.all_reduce(sum_out, dist.ReduceOp.SUM, group=group)
            dist.all_reduce(d, dist.ReduceOp.SUM, group=group)

        tensor[..., :-1], tensor[..., -1:] = sum_out / d, max_lse + torch.log(d)
    
    elif op == dist.reduce_scatter:
        pass
        # assert isinstance(dst, List) , "dst should be a list of tensors"
        # local_outs = [t[..., :-1] for t in tensor]
        # local_lses = [t[..., -1] for t in tensor]
        # max_lses = [lse.clone() for lse in local_lses]
        # for max_lse in max_lses:
        #     dist.all_reduce(max_lse, dist.ReduceOp.MAX, group=group)
        # ds = [torch.exp(lse - max_lse) for lse, max_lse in zip(local_lses, max_lses)]
        # sum_outs = [out * d for out, d in zip(local_outs, ds)]
        # sum_out = torch.empty_like(sum_outs[0])
        # dist.reduce_scatter(sum_out, sum_outs, dist.ReduceOp.SUM, group=group)
        # d = torch.empty_like(ds[0])
        # dist.reduce_scatter(d, ds, dist.ReduceOp.SUM, group=group)
        # return sum_out / d, max_lses[dist.get_rank(group)] + torch.log(d)
    else:
        raise RuntimeError("Unsupported collective op")


flash_attn_manage_reduction = """
def flash_attn_expose_reduce(
    q: torch.Tensor, 
    kv: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # Define axes
    B = Axis("B", {BATCH})  # Batch dimension
    H = Axis("H", {HEADS})  # Head dimension
    S_Q = Axis("S_q", {SEQ_LEN}, min_block_size=32)  # Query sequence length
    S_KV = Axis("S_kv", {SEQ_LEN}, min_block_size=32)  # Key/Value sequence length

    # Match buffers
    Q = match_buffer(q, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H, None])
    KV = match_buffer(kv, [2, {BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [None, B, S_KV, H, None])
    O = match_buffer(o, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H, None])
    LSE = match_buffer(lse, [{BATCH}, {HEADS}, {SEQ_LEN}], [B, H, S_Q], dtype=torch.float32)

    for b, h, s_q in grid([B, H, S_Q], "sss"):
        reduce_buf = temp_buffer([b, s_q, h, {RED_DIM}], [B, S_Q, H, None], dtype=torch.float32) # b, h, s_q means this dim has the same size as the axis stride
        for s_kv in grid([S_KV], "m"): # m means managed reduction
            _q = load_buffer(Q[b, s_q, h])
            _kv = load_buffer(KV[b, s_kv, h])
            _k = _kv[0]
            _v = _kv[1]
            params = get_default_args(_flash_attn_forward).copy()
            params.update(
                {{
                    "q": _q,
                    "k": _k,
                    "v": _v,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }}
            )
            outputs = _flash_attn_forward(**params)
            assert len(outputs) == 4
            block_out, block_lse, _, _ = outputs

            block_res = torch.cat([block_out.to(torch.float32), block_lse.transpose(-2, -1).unsqueeze(dim=-1)], dim=-1)
            
            reduce(op=flash_reduce,
                   collective_op=update_out_and_lse_collective,
                   buffer = reduce_buf,
                   src=block_res,
                   axis=s_kv
            )

        local_res = load_buffer(reduce_buf[:, :, :, :])
        O[b, s_q, h] = store_buffer(local_res[..., :-1].to(q.dtype))
        LSE[b, h, s_q] = store_buffer(local_res[..., -1].transpose(1, 2))
"""

gqa_manage_reduction = """
def flash_attn_expose_reduce(
    q: torch.Tensor, 
    kv: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # Define axes
    B = Axis("B", {BATCH})  # Batch dimension
    H = Axis("H", {HEADS})  # Head dimension
    S_Q = Axis("S_q", {SEQ_LEN}, min_block_size=32)  # Query sequence length
    S_KV = Axis("S_kv", {SEQ_LEN}, min_block_size=32)  # Key/Value sequence length

    # Match buffers
    Q = match_buffer(q, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H * {HEADS_PER_GROUP}, None])
    KV = match_buffer(kv, [2, {BATCH}, {SEQ_LEN}, {KV_HEADS}, {HEAD_DIM}], [None, B, S_KV, H, None])
    O = match_buffer(o, [{BATCH}, {SEQ_LEN}, {HEADS}, {HEAD_DIM}], [B, S_Q, H * {HEADS_PER_GROUP}, None])
    LSE = match_buffer(lse, [{BATCH}, {HEADS}, {SEQ_LEN}], [B, H * {HEADS_PER_GROUP}, S_Q ], dtype=torch.float32)

    for b, h, s_q in grid([B, H, S_Q], "sss"):
        reduce_buf = temp_buffer([b, s_q, h, {RED_DIM}], [B, S_Q, H, None], dtype=torch.float32) # b, h, s_q means this dim has the same size as the axis stride
        for s_kv in grid([S_KV], "m"): # m means managed reduction
            _q = load_buffer(Q[b, s_q, h])
            _kv = load_buffer(KV[b, s_kv, h])
            _k = _kv[0]
            _v = _kv[1]
            params = get_default_args(_flash_attn_forward).copy()
            params.update(
                {{
                    "q": _q,
                    "k": _k,
                    "v": _v,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": causal,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,
                    "window_size_left": window_size[0],
                    "window_size_right": window_size[1],
                }}
            )
            outputs = _flash_attn_forward(**params)
            assert len(outputs) == 4
            block_out, block_lse, _, _ = outputs

            block_res = torch.cat([block_out.to(torch.float32), block_lse.transpose(-2, -1).unsqueeze(dim=-1)], dim=-1)
            
            reduce(op=flash_reduce,
                   collective_op=update_out_and_lse_collective,
                   buffer = reduce_buf,
                   src=block_res,
                   axis=s_kv
            )

        local_res = load_buffer(reduce_buf[:, :, :, :])
        O[b, s_q, h] = store_buffer(local_res[..., :-1].to(q.dtype))
        LSE[b, h, s_q] = store_buffer(local_res[..., -1].transpose(1, 2))
"""