# FlashInfer-inspired distributed problems

Problems 88–92 are standalone PyTorch/distributed semantic adaptations, not
copies of FlashInfer's CUDA/Triton kernels or drop-in API replacements. They
require no FlashInfer installation, NVSHMEM, or MNNVL fabric memory. Optimized
solutions may choose those transports when the target hardware supports them.
All inputs are read-only; input allocation and process-group creation belong to
the input builder, outside timed solution calls. Temporary tensor allocation within
each reference call is included in its timing, as in other PKB references.

| ID | Problem | Parallelism / reference communication | Outputs |
| --- | --- | --- | --- |
| 88 | `decode_cp_attention_merge` | CP: two all-to-all exchanges, then LSE-weighted reduction | `[B,H,D]` merged attention |
| 89 | `mixed_tp_dp_allreduce_allgather` | TP sum, then DP gather of token rows | `[DP*T,H]` activation |
| 90 | `moe_finalize_allreduce_residual_rmsnorm` | Weighted expert unpermutation, TP sum, residual/RMSNorm | residual and normalized output |
| 91 | `allreduce_residual_rmsnorm_fp8` | TP sum, residual/RMSNorm, per-token E4M3 output | residual, FP8 values, FP32 scales |
| 92 | `fp8_two_shot_activation_allreduce` | FP8 byte/scales all-to-all, owner reduction, FP8 byte/scales all-gather | reconstructed activation sum |

## Contracts and differences from upstream

**88:** Input shapes are `[B,H,P,D]` and `[B,H,P]`. Each source computes
contributions for all destinations; output rank `r` merges slot `r` from all
sources. LSEs may use natural logs or base 2, selected by a scalar flag. `-inf`
means an empty KV shard; an entirely empty row returns zero. Inputs must not
contain NaN or positive-infinite LSEs. The reference exchanges contributions
through NCCL and merges in FP32. FlashInfer's specific FIFO transport requires
MNNVL fabric memory; that restriction is not part of the benchmark semantics.
The input builder alternates LSE bases and includes empty shards and LSEs near
1000 to exercise stable merging. This task starts after local attention;
query projection and paged-KV attention are outside its scope.

**89:** Rank layout is `rank = dp_rank * tp_size + tp_rank`. Contiguous TP groups
sum partial activations; strided DP groups concatenate the resulting token rows
in DP-rank order. FP32 intermediates are cast to the input dtype at the end.
Inputs contain distinct contributions on each rank. On eight GPUs, trials
alternate TP×DP layouts 2×4 and 4×2. Prime world sizes use TP=world, DP=1;
one rank is the identity. This adapts FlashInfer's logical mixed collective,
without requiring its CUDA virtual-memory/NVSHMEM fused transport.

**90:** Expert outputs have shape `[permuted_rows,H]`. The map `[T,K]` identifies
each token/route's row, with `-1` denoting absence. Other indices must be in
range. Router weights `[T,K]`, residual `[T,H]`, and gamma `[H]` are replicated;
expert outputs are rank-local partial contributions. Local combination and TP
sum use FP32. The residual is added **once after** the sum, rounded to residual
dtype, then normalized in FP32 and cast back. Return both the residual and
normalized tensors. There is no shared expert. Trials use top-k 2/4, random
permutations, missing routes, and a completely absent token. An empty expert
buffer is valid when all its routes are absent.

**91:** Sum partial activations in FP32, add the replicated residual, and round
to input dtype. Compute RMSNorm in FP32 and round the normalized activation to
input dtype before quantization. For each token,
`scale = max(amax(abs(normalized)), 1e-12) / 448`; output is saturated E4M3
`normalized / scale`. Return `(residual_out, fp8_out, scale)`; reconstruction is
`fp8_out.float() * scale`. This deliberately fixes rounding boundaries for an
unambiguous benchmark. Unlike task 92, communication is **not compressed**.
The input builder includes an all-zero token.

**92:** Flatten inputs and divide into contiguous owner stripes. The total
element count must be divisible by `world_size * scale_group`. Quantize each
input group to E4M3 with FP32 scales, exchange destination stripes as uint8
bytes plus scales, and accumulate dequantized contributions in source-rank
order using FP32. Owners quantize their reduced stripes again for gathering.
As in FlashInfer's implementation, each owner keeps its own stripe before the
second quantization; other ranks reconstruct that stripe after the second
quantization. Outputs therefore differ slightly by rank. This is a **sum**,
not a gradient average; even at world size 1 the input quantization occurs.
The reference sends quantized bytes rather than reconstructing BF16 before
communication. Generated inputs use groups of 256, zero groups, and different
rank-local magnitudes. FlashInfer's implementation uses tiled ownership and
symmetric-memory peer access; this reference fixes contiguous ownership and
uses explicit collectives. Candidates must preserve this reference's ownership
and rounding semantics rather than returning an exact unquantized sum.

## Sources

- 88: [DCP exchange](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/dcp_alltoall.py)
  and [LSE correction reference](https://github.com/flashinfer-ai/flashinfer/issues/4575).
- 89: [MixedCommHandler / run_mixed_comm](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/mixed_comm.py).
- 90–91: [allreduce_fusion patterns](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/allreduce.py)
  and [TRT-LLM fusion bindings](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/trtllm_ar.py).
- 92: [quantized_all_reduce](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/comm/quantized_allreduce.py).

## Validation and execution

CPU/Gloo tests compare against full-input mathematical oracles, including a
full-softmax attention oracle for task 88. They cover one, two, and four ranks,
FP32/BF16, both LSE bases, both MoE top-k values, a 2×2 TP/DP grid,
input immutability, empty experts, and FP8 artifact serialization:

```bash
python -m pytest tests/test_flashinfer_references.py -q
```

Run each reference with the usual GPU harness, for example:

```bash
python run_local.py --nproc-per-node 4 --mode dryrun --problem 88 \
  --dtype bfloat16 --solution reference --download --measure-perf
```

Repeat with IDs 89–92. Kernel generation discovers the numbered files
automatically. CPU tests establish semantic correctness, not CUDA/NCCL
correctness, overlap, speedup, or performance on the target GPU topology.
