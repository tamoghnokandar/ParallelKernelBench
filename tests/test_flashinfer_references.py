"""CPU/Gloo semantic checks; CUDA/NCCL runs use the normal PKB worker.

Run: python -m pytest tests/test_flashinfer_references.py -q
"""
import datetime
import importlib.util
import io
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.input_output_tensors import create_input_tensor
from utils.problem_id import resolve_problem


ROOT = Path(__file__).resolve().parents[1]


def _load(problem):
    stem, _ = resolve_problem(ROOT, str(problem), "reference")
    spec = importlib.util.spec_from_file_location(stem, ROOT / "reference" / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gather(x, size):
    buffers = [torch.empty_like(x) for _ in range(size)]
    dist.all_gather(buffers, x.contiguous())
    return buffers


def _norm(x, gamma):
    xf = x.float()
    return (xf / torch.sqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * gamma.float()).to(x.dtype)


def _roundtrip(x, group_size):
    groups = x.float().reshape(-1, group_size)
    scales = groups.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
    return ((groups / scales).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scales).flatten()


def _check(problem, args, out, rank, size):
    if problem == 88:
        values, lses = _gather(args[0], size), _gather(args[1], size)
        # Independent high-precision merge of source contributions to this destination.
        lse = torch.stack([x[:, :, rank] for x in lses]).double()
        if not args[2]:
            lse = lse * torch.log(torch.tensor(2.0, dtype=torch.float64))
        empty = torch.isneginf(lse).all(0)
        lse[:, empty] = 0
        weights = torch.softmax(lse, dim=0)
        expected = (torch.stack([x[:, :, rank] for x in values]).double() * weights[..., None]).sum(0)
        expected[empty] = 0
        torch.testing.assert_close(out, expected.to(out.dtype), atol=1e-2, rtol=1e-2)
    elif problem == 89:
        inputs = _gather(args[0], size)
        tp_size = dist.get_world_size(args[1])
        expected = torch.cat([
            torch.stack(inputs[start:start + tp_size]).float().sum(0)
            for start in range(0, size, tp_size)
        ])
        torch.testing.assert_close(out, expected.to(out.dtype), atol=1e-2, rtol=1e-2)
    elif problem == 90:
        rows, mappings = _gather(args[0], size), _gather(args[1], size)
        weights, residual, gamma = args[2:5]
        # Explicit token/route iteration avoids relying on the reference's gather layout.
        combined = torch.zeros_like(residual, dtype=torch.float64)
        for source in range(size):
            for token in range(residual.shape[0]):
                for route in range(weights.shape[1]):
                    index = int(mappings[source][token, route])
                    if index >= 0:
                        combined[token] += rows[source][index].double() * weights[token, route].double()
        expected = (combined.float() + residual.float()).to(residual.dtype)
        torch.testing.assert_close(out[0], expected, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[1], _norm(expected, gamma), atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out[0][0], residual[0])
    elif problem == 91:
        inputs = _gather(args[0], size)
        residual = (torch.stack(inputs).float().sum(0) + args[1].float()).to(args[0].dtype)
        normalized = _norm(residual, args[2]).float()
        scale = normalized.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448.0
        quantized = (normalized / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        torch.testing.assert_close(out[0], residual)
        torch.testing.assert_close(out[1].float(), quantized.float())
        torch.testing.assert_close(out[2], scale)
        # Verify that native FP8 outputs survive PKB's tensor artifact format.
        buffer = io.BytesIO()
        torch.save({"output_1": out[1]}, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)["output_1"]
        assert torch.equal(restored.float(), out[1].float())
    elif problem == 92:
        inputs = _gather(args[0], size)
        reconstructed = [_roundtrip(x, args[1]) for x in inputs]
        summed = torch.zeros_like(reconstructed[0])
        for x in reconstructed:
            summed += x
        expected = _roundtrip(summed, args[1]).to(args[0].dtype)
        stripe = summed.numel() // size
        expected[rank * stripe:(rank + 1) * stripe] = summed[rank * stripe:(rank + 1) * stripe].to(expected.dtype)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)


def _attention_from_logits(rank, size):
    """Check task 88 against full softmax attention, not just another LSE merge."""
    torch.manual_seed(100 + rank)
    scores = torch.randn(2, 3, size, 5)
    values = torch.randn(2, 3, size, 5, 8)
    partial = (scores.softmax(-1)[..., None] * values).sum(-2)
    lse = scores.logsumexp(-1)
    out = _load(88).solution(partial, lse)
    all_scores, all_values = _gather(scores, size), _gather(values, size)
    scores_full = torch.cat([x[:, :, rank] for x in all_scores], dim=-1)
    values_full = torch.cat([x[:, :, rank] for x in all_values], dim=-2)
    expected = (scores_full.softmax(-1)[..., None] * values_full).sum(-2)
    torch.testing.assert_close(out, expected)


def _worker(rank, size, rendezvous, dtype_name):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=size,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        for trial in range(2):
            for problem in range(88, 93):
                args = create_input_tensor(rank, size, problem, (32, 64), getattr(torch, dtype_name), trial, "cpu")
                before = [x.clone() if isinstance(x, torch.Tensor) else None for x in args]
                out = _load(problem).solution(*args)
                _check(problem, args, out, rank, size)
                for original, value in zip(before, args):
                    if original is not None:
                        assert torch.equal(original, value), f"Problem {problem} mutated an input"
        _attention_from_logits(rank, size)
        # Empty expert storage and padded routes must still return the residual.
        empty_args = (
            torch.empty(0, 8), torch.full((2, 2), -1), torch.ones(2, 2) / 2,
            torch.ones(2, 8), torch.ones(8),
        )
        out = _load(90).solution(*empty_args)
        torch.testing.assert_close(out[0], empty_args[3])
        with pytest.raises(ValueError, match="divisible"):
            _load(92).solution(torch.ones(7), scale_group=4)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [1, 2, 4])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_distributed_references(tmp_path, size, dtype_name):
    rendezvous = (tmp_path / "gloo_init").as_uri()
    mp.spawn(_worker, args=(size, rendezvous, dtype_name), nprocs=size, join=True)
