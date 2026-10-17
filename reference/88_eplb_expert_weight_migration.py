from typing import Dict, List, Optional

import torch
import torch.distributed as dist


def _first_slot(indices: List[int]) -> Dict[int, int]:
    first: Dict[int, int] = {}
    for slot, expert in enumerate(indices):
        if expert >= 0 and expert not in first:
            first[expert] = slot
    return first


@torch.no_grad()
def solution(
    expert_weights: torch.Tensor,
    old_global_expert_indices: torch.Tensor,
    new_global_expert_indices: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    group = group or dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    num_local_experts = expert_weights.shape[0]

    old_indices = old_global_expert_indices.tolist()
    new_indices = new_global_expert_indices.tolist()
    old_first = _first_slot(old_indices)

    local_lo = rank * num_local_experts
    local_hi = local_lo + num_local_experts
    old_local = {
        old_indices[slot]: slot - local_lo
        for slot in range(local_hi - 1, local_lo - 1, -1)
        if old_indices[slot] >= 0
    }

    # Staging buffer: every destination slot is written from the untouched input.
    out = torch.zeros_like(expert_weights)
    p2p_ops: List[dist.P2POp] = []

    # Receives: one per (source rank, expert) needed here and not already held locally.
    recv_slot: Dict[int, int] = {}
    pending_copies: List[tuple] = []
    for dst_local in range(num_local_experts):
        expert = new_indices[local_lo + dst_local]
        if expert < 0:
            continue
        if expert in old_local:
            out[dst_local].copy_(expert_weights[old_local[expert]])
        elif expert in recv_slot:
            pending_copies.append((dst_local, recv_slot[expert]))
        else:
            recv_slot[expert] = dst_local
    for expert in sorted(recv_slot):
        src = old_first[expert] // num_local_experts
        p2p_ops.append(
            dist.P2POp(
                dist.irecv,
                out[recv_slot[expert]],
                dist.get_global_rank(group, src),
                group=group,
            )
        )

    # Sends: each local expert whose canonical old copy lives here goes to every rank
    # that needs it in the new layout but does not already hold it.
    for dst in range(world_size):
        if dst == rank:
            continue
        dst_new = set(new_indices[dst * num_local_experts:(dst + 1) * num_local_experts])
        dst_old = set(old_indices[dst * num_local_experts:(dst + 1) * num_local_experts])
        for expert in sorted(dst_new - dst_old):
            if expert < 0 or old_first[expert] // num_local_experts != rank:
                continue
            p2p_ops.append(
                dist.P2POp(
                    dist.isend,
                    expert_weights[old_first[expert] - local_lo],
                    dist.get_global_rank(group, dst),
                    group=group,
                )
            )

    if p2p_ops:
        for req in dist.batch_isend_irecv(p2p_ops):
            req.wait()

    for dst_local, src_local in pending_copies:
        out[dst_local].copy_(out[src_local])
    return out
