from typing import Optional

import torch
import torch.distributed as dist


def _all_gather_dim0(shard: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    full = torch.empty(
        (shard.shape[0] * world_size, *shard.shape[1:]),
        dtype=shard.dtype,
        device=shard.device,
    )
    dist.all_gather_into_tensor(full, shard.contiguous(), group=group)
    return full


def _tp_rows(weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    rows = weight.shape[0] // tp_size
    return weight[tp_rank * rows:(tp_rank + 1) * rows]


@torch.no_grad()
def solution(
    qkv_shard: torch.Tensor,
    gate_up_shard: torch.Tensor,
    down_shard: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    tp_rank = dist.get_rank(group)
    tp_size = dist.get_world_size(group)

    # FSDP -> full weights.
    w_qkv = _all_gather_dim0(qkv_shard, group)
    w_gate_up = _all_gather_dim0(gate_up_shard, group)
    w_down = _all_gather_dim0(down_shard, group)

    # Column-parallel QKV: split Q, K, V by heads independently, then re-fuse.
    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    w_q, w_k, w_v = torch.split(w_qkv, [q_size, kv_size, kv_size], dim=0)
    qkv_tp = torch.cat(
        [_tp_rows(w, tp_rank, tp_size) for w in (w_q, w_k, w_v)], dim=0
    )

    # Column-parallel gate/up: split each half, then re-fuse.
    w_gate, w_up = w_gate_up.chunk(2, dim=0)
    gate_up_tp = torch.cat(
        [_tp_rows(w, tp_rank, tp_size) for w in (w_gate, w_up)], dim=0
    )

    # Row-parallel down: split along the input dim.
    cols = w_down.shape[1] // tp_size
    down_tp = w_down[:, tp_rank * cols:(tp_rank + 1) * cols]

    return torch.cat([qkv_tp.flatten(), gate_up_tp.flatten(), down_tp.flatten()])
