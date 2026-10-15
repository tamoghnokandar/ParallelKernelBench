from typing import Optional

import torch
import torch.distributed as dist


@torch.no_grad()
def solution(
    x: torch.Tensor,
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    world_size = dist.get_world_size(group)
    C = x.shape[1]
    reduce_dims = [0] + list(range(2, x.dim()))
    stat_shape = [1, C] + [1] * (x.dim() - 2)

    xf = x.float()
    dyf = grad_out.float()
    w = weight.float()
    b = bias.float()

    # ---- forward: local (mean, invstd, count), all-gathered and combined ----
    count = x.numel() // C
    mean = xf.mean(dim=reduce_dims)
    var = xf.var(dim=reduce_dims, unbiased=False)
    invstd = torch.rsqrt(var + eps)
    local_stats = torch.cat(
        [mean, invstd, torch.full((1,), float(count), dtype=torch.float32, device=x.device)]
    )
    stats = torch.empty(world_size * (2 * C + 1), dtype=torch.float32, device=x.device)
    dist.all_gather_into_tensor(stats, local_stats, group=group)
    stats = stats.view(world_size, 2 * C + 1)
    means, invstds, counts = stats[:, :C], stats[:, C:2 * C], stats[:, 2 * C:]

    total = counts.sum()
    g_mean = (means * counts).sum(dim=0) / total
    vars_ = invstds.pow(-2) - eps
    g_var = (counts * (vars_ + (means - g_mean).pow(2))).sum(dim=0) / total
    g_invstd = torch.rsqrt(g_var + eps)

    xmu = xf - g_mean.view(stat_shape)
    y = xmu * g_invstd.view(stat_shape) * w.view(stat_shape) + b.view(stat_shape)

    # ---- backward: local reductions, all-reduced for grad_input ----
    sum_dy = dyf.sum(dim=reduce_dims)
    sum_dy_xmu = (dyf * xmu).sum(dim=reduce_dims)
    # Per-rank parameter grads (as in torch SyncBatchNorm; DDP reduces them later).
    grad_weight = sum_dy_xmu * g_invstd
    grad_bias = sum_dy

    sums = torch.cat([sum_dy, sum_dy_xmu])
    dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=group)
    mean_dy = (sums[:C] / total).view(stat_shape)
    mean_dy_xmu = (sums[C:] / total).view(stat_shape)

    inv = g_invstd.view(stat_shape)
    grad_x = (dyf - mean_dy - xmu * inv * inv * mean_dy_xmu) * inv * w.view(stat_shape)

    return torch.cat(
        [y.flatten(), grad_x.flatten(), grad_weight, grad_bias]
    ).to(x.dtype)
