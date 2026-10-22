"""Weighted MoE unpermutation -> TP sum -> residual -> RMSNorm.

Semantic adaptation of kMoEFinalizeARResidualRMSNorm (no shared expert):
https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/allreduce.py
"""
from typing import Optional

import torch
import torch.distributed as dist


@torch.no_grad()
def solution(
    expert_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    routing_weights: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
    group: Optional[dist.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (residual_out, norm_out), both [tokens, hidden] in input dtype.

    expert_out is [permuted_rows, hidden], mapping/weights are [tokens, top_k].
    Mapping -1 is an absent contribution; other indices must be valid rows.
    Routing weights and residual/gamma are replicated across TP ranks; expert
    outputs are partial contributions. Residual is added once, after the TP sum.
    """
    group = group if group is not None else dist.group.WORLD
    if expanded_idx_to_permuted_idx.shape != routing_weights.shape:
        raise ValueError("Mapping and routing weights must have the same shape")
    if expert_out.shape[0] == 0:
        combined = torch.zeros_like(residual, dtype=torch.float32)
    else:
        indices = expanded_idx_to_permuted_idx.long()
        gathered = expert_out[indices.clamp_min(0)].float()
        gathered = torch.where((indices >= 0).unsqueeze(-1), gathered, 0.0)
        combined = (gathered * routing_weights.float().unsqueeze(-1)).sum(dim=1)
    if dist.get_world_size(group) > 1:
        dist.all_reduce(combined, group=group)
    residual_out = (combined + residual.float()).to(residual.dtype)
    x = residual_out.float()
    norm_out = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    norm_out = (norm_out * gamma.float()).to(residual.dtype)
    return residual_out, norm_out
