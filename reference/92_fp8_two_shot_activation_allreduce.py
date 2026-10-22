"""Two-shot all-reduce with E4M3 bytes and per-group FP32 scales on the wire.

Semantic adaptation of FlashInfer's quantized_all_reduce:
https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/quantized_allreduce.py
Uses NCCL-compatible byte exchanges instead of symmetric-memory peer loads.
"""
from typing import Optional

import torch
import torch.distributed as dist


def _quantize(x: torch.Tensor, scale_group: int) -> tuple[torch.Tensor, torch.Tensor]:
    groups = x.reshape(-1, scale_group).float()
    scales = groups.abs().amax(dim=-1).clamp_min(1e-12) / 448.0
    quantized = (groups / scales.unsqueeze(-1)).clamp(-448.0, 448.0)
    return quantized.to(torch.float8_e4m3fn).view(torch.uint8).flatten(), scales


def _dequantize(data: torch.Tensor, scales: torch.Tensor, scale_group: int) -> torch.Tensor:
    values = data.contiguous().view(torch.float8_e4m3fn).float().reshape(-1, scale_group)
    return (values * scales.unsqueeze(-1)).flatten()


@torch.no_grad()
def solution(
    activation: torch.Tensor,
    scale_group: int = 256,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Return a sum (not an average), preserving the activation shape/dtype.

    numel must be divisible by world_size * scale_group. Each rank owns one
    contiguous stripe. Input stripes are quantized, exchanged, and summed in
    source-rank order using FP32. The owner keeps its sum rounded to input dtype;
    other ranks receive an E4M3 round-trip of that sum, matching FlashInfer's
    owner-stripe behavior. Thus outputs may differ slightly between ranks.
    """
    group = group if group is not None else dist.group.WORLD
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    if scale_group <= 0 or activation.numel() % (size * scale_group):
        raise ValueError("numel must be divisible by world_size * scale_group")
    data, scales = _quantize(activation.flatten(), scale_group)
    if size > 1:
        received, received_scales = torch.empty_like(data), torch.empty_like(scales)
        dist.all_to_all_single(received, data, group=group)
        dist.all_to_all_single(received_scales, scales, group=group)
    else:
        received, received_scales = data, scales
    contributions = _dequantize(received, received_scales, scale_group).reshape(size, -1)
    stripe = torch.zeros_like(contributions[0])
    for source in range(size):
        stripe = stripe + contributions[source]
    stripe_data, stripe_scales = _quantize(stripe, scale_group)
    if size > 1:
        all_data = [torch.empty_like(stripe_data) for _ in range(size)]
        all_scales = [torch.empty_like(stripe_scales) for _ in range(size)]
        dist.all_gather(all_data, stripe_data, group=group)
        dist.all_gather(all_scales, stripe_scales, group=group)
        result = _dequantize(torch.cat(all_data), torch.cat(all_scales), scale_group)
    else:
        result = _dequantize(stripe_data, stripe_scales, scale_group)
    result = result.to(activation.dtype)
    stripe_size = stripe.numel()
    result[rank * stripe_size : (rank + 1) * stripe_size] = stripe.to(activation.dtype)
    return result.reshape_as(activation)
