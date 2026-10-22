"""Sum partial activations over TP, then gather token rows over DP.

Semantic reference for MixedCommOp.ALLREDUCE_ALLGATHER in FlashInfer:
https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/mixed_comm.py
Rank layout: global_rank = dp_rank * tp_size + tp_rank.
"""
from typing import Optional

import torch
import torch.distributed as dist


@torch.no_grad()
def solution(
    partial: torch.Tensor,
    tp_group: Optional[dist.ProcessGroup] = None,
    dp_group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """[local_tokens, hidden] -> [dp_size * local_tokens, hidden].

    Both groups are explicit in benchmark inputs. None denotes WORLD for TP and
    a singleton DP axis, making the default a conventional all-reduce.
    FP32 accumulation precedes the final cast; the input is never modified.
    """
    tp_group = tp_group if tp_group is not None else dist.group.WORLD
    reduced = partial.float().clone()
    if dist.get_world_size(tp_group) > 1:
        dist.all_reduce(reduced, group=tp_group)
    if dp_group is not None and dist.get_world_size(dp_group) > 1:
        shards = [torch.empty_like(reduced) for _ in range(dist.get_world_size(dp_group))]
        dist.all_gather(shards, reduced.contiguous(), group=dp_group)
        reduced = torch.cat(shards, dim=0)
    return reduced.to(partial.dtype)
