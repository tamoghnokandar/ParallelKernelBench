"""TP sum, residual/RMSNorm, and dynamic per-token E4M3 output quantization.

Semantic adaptation of FlashInfer's dynamic FP8 allreduce fusion pattern:
https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/allreduce.py
The collective transports FP32; FP8 is the consumer's output format.
"""
from typing import Optional

import torch
import torch.distributed as dist


@torch.no_grad()
def solution(
    partial: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
    group: Optional[dist.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (residual_out, fp8_out, scales); scales [tokens,1] dequantize FP8.

    Sum and RMS statistics use FP32. Residual and normalized activations are
    rounded to the input dtype at their boundaries before quantizing to E4M3.
    """
    group = group if group is not None else dist.group.WORLD
    reduced = partial.float().clone()
    if dist.get_world_size(group) > 1:
        dist.all_reduce(reduced, group=group)
    residual_out = (reduced + residual.float()).to(partial.dtype)
    x = residual_out.float()
    normalized = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    normalized = (normalized * gamma.float()).to(partial.dtype).float()
    scales = normalized.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
    quantized = (normalized / scales).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return residual_out, quantized, scales
