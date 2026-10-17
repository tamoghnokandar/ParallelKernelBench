import os
from typing import Dict, Optional

import torch
import torch.distributed as dist

# Per-process NIXL state, keyed by global rank. Agent creation and peer metadata exchange
# happen once; memory is re-registered only when the input / staging buffers move.
_NIXL_STATE: Dict[int, dict] = {}


def _import_nixl():
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError:
        try:
            from nixl_cu12._api import nixl_agent, nixl_agent_config
        except ImportError as exc:
            raise ImportError(
                "Problem 92 reference requires NIXL. Install it with `uv sync --extra nixl` "
                "(or `pip install nixl-cu12`)."
            ) from exc
    return nixl_agent, nixl_agent_config


def _get_state(global_rank: int) -> dict:
    state = _NIXL_STATE.get(global_rank)
    if state is None:
        os.environ.setdefault("UCX_TLS", "cuda_ipc,cuda_copy,sm,self")
        nixl_agent, nixl_agent_config = _import_nixl()
        agent = nixl_agent(f"pkb_rank{global_rank}", nixl_agent_config(backends=[]))
        # Single-node pull: no peer-failure reporting, which lets UCX use shared memory for its
        # control traffic (peer error handling needs a network transport such as tcp or IB).
        params = agent.get_plugin_params("UCX")
        params["ucx_error_handling_mode"] = "none"
        agent.create_backend("UCX", params)
        state = {
            "agent": agent,
            "key": None,
            "reg_descs": [],
            "tensors": None,
            "staging": None,
            "peers": {},
        }
        _NIXL_STATE[global_rank] = state
    return state


def _dev_id(t: torch.Tensor) -> int:
    return max(t.get_device(), 0)


def _get_staging(state: dict, shape: tuple, like: torch.Tensor) -> torch.Tensor:
    staging = state["staging"]
    if (
        staging is None
        or tuple(staging.shape) != shape
        or staging.dtype != like.dtype
        or staging.device != like.device
    ):
        staging = torch.empty(shape, dtype=like.dtype, device=like.device)
    return staging


def _register_and_exchange(
    state: dict,
    kv_pages: torch.Tensor,
    staging: torch.Tensor,
    key: tuple,
    group: dist.ProcessGroup,
) -> None:
    agent = state["agent"]
    if state["key"] != key:
        for descs in state["reg_descs"]:
            agent.deregister_memory(descs)
        state["reg_descs"] = [agent.register_memory(kv_pages), agent.register_memory(staging)]
        # Hold references so the registered allocations cannot be freed and reused.
        state["tensors"] = (kv_pages, staging)
        state["staging"] = staging
        state["key"] = key

    # Remote keys travel in the agent metadata, so peers reload it after any registration change.
    world_size = dist.get_world_size(group)
    info = (agent.get_agent_metadata(), kv_pages.data_ptr(), _dev_id(kv_pages))
    infos = [None] * world_size
    dist.all_gather_object(infos, info, group=group)

    rank = dist.get_rank(group)
    peers = {}
    for peer in range(world_size):
        if peer == rank:
            continue
        name = f"pkb_rank{dist.get_global_rank(group, peer)}"
        if name in state["peers"]:
            agent.remove_remote_agent(name)
        agent.add_remote_agent(infos[peer][0])
        peers[peer] = {"name": name, "base": infos[peer][1], "dev": infos[peer][2]}
    peers[rank] = {"name": agent.name, "base": kv_pages.data_ptr(), "dev": _dev_id(kv_pages)}
    state["peers"] = {p["name"]: p for p in peers.values()}
    state["by_rank"] = peers


@torch.no_grad()
def solution(
    kv_pages: torch.Tensor,
    src_block_table: torch.Tensor,
    dst_block_table: torch.Tensor,
    num_dst_pages: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    import numpy as np

    group = group or dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    state = _get_state(dist.get_rank())
    agent = state["agent"]

    kv_pages = kv_pages.contiguous()
    _, page, local_heads, head_dim = kv_pages.shape
    page_bytes = page * local_heads * head_dim * kv_pages.element_size()

    staging = _get_staging(
        state, (num_dst_pages, world_size, page, local_heads, head_dim), kv_pages
    )
    staging.zero_()
    key = (kv_pages.data_ptr(), kv_pages.nbytes, staging.data_ptr(), staging.nbytes)

    # Sources and the zeroed staging buffer must be complete before any peer READs.
    if kv_pages.is_cuda:
        torch.cuda.current_stream().synchronize()

    # Metadata exchange (block tables + "registration changed" flag); doubles as the
    # pre-transfer barrier.
    num_logical = src_block_table.shape[0]
    meta = torch.empty(num_logical + 1, dtype=torch.int64, device=kv_pages.device)
    meta[0] = int(state["key"] != key)
    meta[1:] = src_block_table.to(torch.int64)
    all_meta = torch.empty(world_size * (num_logical + 1), dtype=torch.int64, device=kv_pages.device)
    dist.all_gather_into_tensor(all_meta, meta, group=group)
    all_meta = all_meta.view(world_size, num_logical + 1).cpu()
    if bool(all_meta[:, 0].any()):
        _register_and_exchange(state, kv_pages, staging, key, group)
    src_tables = all_meta[:, 1:]
    peers = state["by_rank"]

    dst = dst_block_table.to(torch.int64).cpu()
    mine = (dst[:, 0] == rank).nonzero().flatten()
    dst_slots = dst[mine, 1]

    # Local head slice: plain device copy.
    if mine.numel() > 0:
        local_pages = src_tables[rank, mine].to(kv_pages.device)
        staging[dst_slots.to(kv_pages.device), rank] = kv_pages[local_pages]

    # Remote head slices: one-sided READ from each peer's registered page pool.
    handles = []
    staging_base = staging.data_ptr()
    local_dev = _dev_id(staging)
    mem_type = "VRAM" if staging.is_cuda else "DRAM"
    for src in range(world_size):
        if src == rank or mine.numel() == 0:
            continue
        n = mine.numel()
        remote = np.empty((n, 3), dtype=np.uint64)
        remote[:, 0] = (peers[src]["base"] + src_tables[src, mine] * page_bytes).numpy()
        remote[:, 1] = page_bytes
        remote[:, 2] = peers[src]["dev"]
        local = np.empty((n, 3), dtype=np.uint64)
        local[:, 0] = (staging_base + (dst_slots * world_size + src) * page_bytes).numpy()
        local[:, 1] = page_bytes
        local[:, 2] = local_dev

        handle = agent.initialize_xfer(
            "READ",
            agent.get_xfer_descs(local, mem_type),
            agent.get_xfer_descs(remote, mem_type),
            peers[src]["name"],
        )
        if agent.transfer(handle) == "ERR":
            raise RuntimeError(f"NIXL READ from rank {src} failed to post")
        handles.append((src, handle))

    try:
        for src, handle in handles:
            while True:
                status = agent.check_xfer_state(handle)
                if status == "DONE":
                    break
                if status == "ERR":
                    raise RuntimeError(f"NIXL READ from rank {src} failed")
    finally:
        for _, handle in handles:
            agent.release_xfer_handle(handle)

    # No rank may reuse its source pages until every peer has finished reading them.
    dist.barrier(group=group)

    num_heads = world_size * local_heads
    return staging.permute(0, 2, 1, 3, 4).reshape(num_dst_pages, page, num_heads, head_dim)
