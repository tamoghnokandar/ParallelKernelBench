"""Decode-context all-to-all followed by stable attention-state merging.

Semantic adaptation of FlashInfer's decode_cp_a2a_alltoall, with an explicit
PyTorch LSE reduction. No FlashInfer or MNNVL dependency is required.
https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/dcp_alltoall.py
https://github.com/flashinfer-ai/flashinfer/issues/4575
"""
from typing import Optional

import torch
import torch.distributed as dist


@torch.no_grad()
def solution(
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
    is_lse_base_on_e: bool = True,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Inputs [B, H, P, D] / [B, H, P]; output [B, H, D].

    The P axis indexes destination ranks on input and source ranks after exchange.
    An LSE of -inf denotes an empty contribution; an entirely empty row yields zero.
    """
    group = group if group is not None else dist.group.WORLD
    size = dist.get_world_size(group)
    if partial_out.ndim != 4 or partial_out.shape[:-1] != partial_lse.shape:
        raise ValueError("Expected partial_out [B,H,P,D] and partial_lse [B,H,P]")
    if partial_out.shape[2] != size:
        raise ValueError("Destination axis must equal the context-parallel size")
    send_out = partial_out.movedim(2, 0).contiguous()
    send_lse = partial_lse.float().movedim(2, 0).contiguous()
    if size > 1:
        recv_out, recv_lse = torch.empty_like(send_out), torch.empty_like(send_lse)
        dist.all_to_all_single(recv_out, send_out, group=group)
        dist.all_to_all_single(recv_lse, send_lse, group=group)
    else:
        recv_out, recv_lse = send_out, send_lse
    maximum = recv_lse.amax(dim=0, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0.0)
    shifted = recv_lse - maximum
    weights = torch.exp(shifted) if is_lse_base_on_e else torch.exp2(shifted)
    denominator = weights.sum(dim=0).unsqueeze(-1)
    # Empty contributions must not contaminate a valid row with undefined values.
    values = torch.where(weights.unsqueeze(-1) > 0, recv_out.float(), 0.0)
    numerator = (values * weights.unsqueeze(-1)).sum(dim=0)
    return (numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)).to(partial_out.dtype)
