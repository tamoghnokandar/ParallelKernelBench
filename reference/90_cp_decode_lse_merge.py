from typing import Optional, Tuple

import torch
import torch.distributed as dist


def _paged_decode_attn(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, num_q_heads, head_dim = q.shape
    _, page, num_kv_heads, _ = k_cache.shape
    max_tokens = block_table.shape[1] * page

    # Gather pages -> [B, T, Hkv, d]; Q heads are grouped onto their KV head (GQA).
    k = k_cache[block_table].reshape(B, max_tokens, num_kv_heads, head_dim).float()
    v = v_cache[block_table].reshape(B, max_tokens, num_kv_heads, head_dim).float()
    qg = q.float().reshape(B, num_kv_heads, num_q_heads // num_kv_heads, head_dim)

    scores = torch.einsum("bkgd,btkd->bkgt", qg, k) * scale
    valid = torch.arange(max_tokens, device=q.device).view(1, 1, 1, -1) < seq_lens.view(-1, 1, 1, 1)
    scores = scores.masked_fill(~valid, float("-inf"))

    # Empty local sequence: lse = -inf, out = 0.
    lse = torch.logsumexp(scores, dim=-1)
    safe_lse = torch.where(torch.isinf(lse), torch.zeros_like(lse), lse)
    probs = torch.exp(scores - safe_lse.unsqueeze(-1))
    out = torch.einsum("bkgt,btkd->bkgd", probs, v)
    return out.reshape(B, num_q_heads, head_dim), lse.reshape(B, num_q_heads)


@torch.no_grad()
def solution(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    local_seq_lens: torch.Tensor,
    softmax_scale: Optional[float] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    out, lse = _paged_decode_attn(
        q, k_cache, v_cache, block_table, local_seq_lens, float(softmax_scale)
    )

    # All-gather LSEs and rescale local partial outputs by exp(lse_local - lse_global).
    lses = torch.empty((world_size * lse.shape[0], *lse.shape[1:]), dtype=lse.dtype, device=lse.device)
    dist.all_gather_into_tensor(lses, lse.contiguous(), group=group)
    lses = lses.view(world_size, *lse.shape)
    global_lse = torch.logsumexp(lses, dim=0)
    scale = torch.exp(lse - global_lse).nan_to_num(nan=0.0)
    out = out * scale.unsqueeze(-1)

    # Reduce-scatter over heads: [B, Hq, d] -> [B, Hq / world_size, d].
    B, num_q_heads, head_dim = out.shape
    send = out.permute(1, 0, 2).contiguous()
    recv = torch.empty(
        (num_q_heads // world_size, B, head_dim), dtype=send.dtype, device=send.device
    )
    dist.reduce_scatter_tensor(recv, send, op=dist.ReduceOp.SUM, group=group)
    return recv.permute(1, 0, 2).contiguous().to(q.dtype)
