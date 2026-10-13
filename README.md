# warp_decode

An **unofficial re-implementation** of Cursor's *warp decode* MoE inference
technique ([blog](https://cursor.com/blog/warp-decode)), targeting **BF16
precision on NVIDIA Hopper GPUs (H20 / H100)**.

Cursor's original work targets Blackwell and MXFP8; the public blog describes
the high-level idea (organize the kernel around outputs instead of experts,
give each warp one output neuron, avoid all staging) but not the concrete
kernel shape. This repository is an independent implementation that arrives at
similar conclusions on Hopper by working the design out from a roofline
argument, and adds a few Hopper-specific optimizations that were needed to
actually turn the idea into a measured win.

The two kernels in this repository are:

| Kernel                          | Fused operation                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------- |
| `moe_gate_up_3d_batched`        | gate + up projections, `silu(gate) * up` activation, all in one launch          |
| `moe_down_3d_batched`           | down projection with topk-weighted sum-reduce, in one launch                    |

Both take unsorted `topk_ids` directly, so **no `moe_align_block_size` /
scatter / permute is needed**. Router scoring and top-k selection are excluded.

---

## Performance on Hopper

### Setup

- **GPU:** NVIDIA H20 (Hopper, sm_90), with dynamic clocks.
- **Software:** PyTorch 2.13.0+cu130, CUDA toolkit 13.0.88, Triton 3.7.1,
  SGLang 0.5.21, sglang-kernel 0.4.7.
- **Shape:** Qwen3-235B-A22B-style MoE FFN under TP=8:
  `hidden=4096`, `inter=192` (1536 before TP split), `num_experts=128`, `topk=8`.
- **Reference:** SGLang's standard Triton configuration lookup and
  `fused_experts` end-to-end path. Configurations are selected using the original
  batch size `M=B`; gate/up and down use their respective standard configurations.
- **Routing:** eight distinct experts per token, with normalized FP32 routing weights.
- **Timing:** CUDA Events over 16 calls per graph and 200 graph replays per sample.
  Latencies are medians of 14 samples from two runs. Weights and routing are fixed;
  repeated replay can benefit from caching.
- **Precision:** BF16 activations and weights, FP32 accumulators.

For this environment, SGLang falls back to its bundled Triton 3.2.0 H20
configuration. There is no dedicated down configuration for this shape, so
SGLang uses its standard gate/up configuration fallback.

### Results

Times are per call in milliseconds. `speedup = sglang_ms / warp_decode_ms`.
SGLang stage timings include expert alignment.
Speedups above 1 favor warp_decode; `tokens = batch × topk`. Both implementations receive
precomputed top-k IDs and weights, so router computation is excluded. The
end-to-end benchmark measures gate/up, activation, down, and expert combination
in one call. Stage timings are measured separately and should not be added to
estimate end-to-end latency.

**Gate/up + SiLU activation:**

| batch | tokens | warp_decode (ms) | SGLang Triton (ms) | speedup |
| ----: | -----: | ---------------: | -----------------: | ------: |
| 1 | 8 | 0.0059 | 0.0169 | 2.84x |
| 2 | 16 | 0.0121 | 0.0277 | 2.28x |
| 4 | 32 | 0.0315 | 0.0369 | 1.17x |
| 8 | 64 | 0.0504 | 0.0573 | 1.14x |
| 16 | 128 | 0.0933 | 0.0896 | 0.96x |
| 32 | 256 | 0.1551 | 0.1198 | 0.77x |
| 64 | 512 | 0.2829 | 0.1258 | 0.44x |

**Down + top-k weighted sum (with alignment):**

| batch | tokens | warp_decode (ms) | SGLang Triton (ms) | speedup |
| ----: | -----: | ---------------: | -----------------: | ------: |
| 1 | 8 | 0.0050 | 0.0097 | 1.94x |
| 2 | 16 | 0.0079 | 0.0165 | 2.08x |
| 4 | 32 | 0.0151 | 0.0210 | 1.39x |
| 8 | 64 | 0.0334 | 0.0373 | 1.12x |
| 16 | 128 | 0.0680 | 0.0555 | 0.82x |
| 32 | 256 | 0.1372 | 0.0711 | 0.52x |
| 64 | 512 | 0.2689 | 0.0767 | 0.29x |

**End-to-end MoE FFN:**

| batch | tokens | warp_decode (ms) | SGLang Triton (ms) | speedup |
| ----: | -----: | ---------------: | -----------------: | ------: |
| 1 | 8 | 0.0136 | 0.0259 | 1.90x |
| 2 | 16 | 0.0267 | 0.0436 | 1.63x |
| 4 | 32 | 0.0501 | 0.0580 | 1.16x |
| 8 | 64 | 0.0848 | 0.0903 | 1.06x |
| 16 | 128 | 0.1630 | 0.1441 | 0.88x |
| 32 | 256 | 0.2945 | 0.1870 | 0.63x |
| 64 | 512 | 0.5516 | 0.2037 | 0.37x |

### Interpretation

For this shape on H20, warp_decode is faster at small batch sizes: the
end-to-end speedup is 1.90x at batch 1 and 1.06x at batch 8. The standard
SGLang reference is faster at batches 16, 32, and 64.

These measurements support using warp_decode for small-batch decode. The
crossover depends on the shape, routing distribution, GPU, and SGLang
configuration; it should be measured for the intended workload.

---

## Design & optimizations

The blog gives one useful headline — *"assign each warp to one output value,
stream weights, avoid staging"* — but leaves out the details that actually
determine performance on Hopper. Direct translation of the headline gives a
kernel that is correct but far from optimal (we measured this — see below).
Two structural changes were needed for each of the two kernels.

### `moe_gate_up_3d_batched`

**Shape.** `X:[B, hidden]` × `W:[E, 2*inter, hidden]` → `[B*topk, inter]`,
where the output is `silu(gate) * up` fused.

**Kernel structure:**
- 2D grid: `(B*topk, inter / WARPS_PER_CTA)`, `blockDim = 256` = 8 warps
- Each CTA handles one `(token, topk_slot)` pair.
- The 8 warps in a CTA together cover 8 adjacent output neurons.
- All 8 warps in a CTA read the **same** `x[token]` row directly from
  global memory through the cache hierarchy, without shared-memory staging.
- Each warp streams two weight rows (`gate` and `up` for its neuron) directly
  from HBM using vectorized `uint4` loads (`VEC = 8` bf16 per lane per iter),
  and accumulates the two dot products in registers.
- Final `silu(g) * u` and the bf16 write-out happen at the end in a single
  lane.

**Why this specific layout:**
- The reduction axis is `hidden = 4096`, long enough that a single warp gives
  a sufficiently deep dot-product pipeline.
- 8 warps per CTA provide warp-level parallelism to hide weight-loading latency.
- Each warp performs its own reduction, without a CTA barrier.

### `moe_down_3d_batched`

**Shape.** `buffer:[B, topk, inter]` × `W:[E, hidden, inter]` → `[B, hidden]`,
with per-slot `topk_scale` folded in and a sum-reduce over `topk`.

**This kernel is where the naive translation of Cursor's headline hurts.**
The obvious mirror of the gate_up design — 1 warp per output `h`, stage
`buffer[token, :, :]` in shared memory, sum across topk in a single warp —
was significantly slower than the sglang triton reference for `B > 4` in
earlier measurements. The reasons are structural:

1. The reduction axis is `inter = 192`, only ~3 lane-loop iterations. A
   single dot product is not enough to hide HBM latency by itself.
2. The small input buffer is reused through the cache hierarchy. Staging it
   in smem adds `__syncthreads` and extra staging work.

The design that actually works on Hopper is very different:

**Kernel structure (v2):**
- **1D grid** over `(token, h-group)`. Each warp owns adjacent output dims:
  2 for `inter=192, B=1–2`, and 8 otherwise.
- No shared memory at all. The 3 KB per-token buffer fits comfortably in the
  L1/L2 hierarchy on its own; explicit staging just costs.
- For each of the `topk` experts, the warp reads one `bf16x2` element of the
  buffer and reuses it across its output weight rows. BF16 operands are widened
  before FP32 `fmaf` updates, without rounding the products to BF16.
- Independent FP32 accumulators (one per output `h`) live in registers per
  lane; the topk-weighted sum folds `topk_scale` in FP32 before the final
  BF16 conversion.
- The output loop is `#pragma unroll`-ed. The `inter=192, B < 16` path
  also fully unrolls the reduction.

**Development measurements** (Qwen3 TP=8, batch 16):

| version | down time |
| ------- | --------: |
| v1: 1 warp / output, smem-staged buffer | 0.143 ms |
| v2: `OUTS_PER_WARP = 2`, no smem | 0.097 ms |
| v2: `OUTS_PER_WARP = 8`, fully unrolled | 0.066 ms |

These earlier measurements illustrate the effect of the kernel design changes.
The performance tables above use the current standard SGLang reference.

The earlier `OUTS_PER_WARP` sweep on H20 favored 8: 4 leaves
input-side arithmetic intensity on the table, 16 pushes register pressure
past the point where occupancy falls off and the B=1 CTA count halves.

### Numerical accuracy

Both kernels use BF16 activations and weights with FP32 accumulators. The
gate/up kernel converts operands to FP32 before multiplication and applies
SiLU in FP32 before writing BF16 output. The down kernel also widens operands
before FP32 multiplication and accumulation. It applies `topk_scale` and sums
across experts in FP32 before the final BF16 conversion.

The measured batches passed independent FP32-reference and CUDA Graph checks.
Maximum warp_decode NRMSE was 0.1679% for down and 0.1694% for the complete FFN.

Both kernels support PyTorch's current CUDA stream and CUDA Graph capture;
warp_decode does not currently use Programmatic Dependent Launch (PDL).

---

## Build & test

### Requirements

- Linux, a CUDA toolkit compatible with the installed PyTorch build, and a
  Hopper GPU (sm_90). The benchmark above was tested on H20 with CUDA 13.0.
- PyTorch (with matching CUDA build) — the kernels are JIT-compiled via
  `torch.utils.cpp_extension.load`, no separate build step is needed.
- For benchmarks: `sglang` and `sglang-kernel`.

### Install & first run

```bash
# From the repo root — the kernels are JIT-compiled on the first import.
python tests/test_gate_up_correctness.py
python tests/test_down_correctness.py

# With SGLang and sglang-kernel installed:
python tests/test_benchmark_reference.py
```

The initial import triggers an `nvcc` build under `build/`. Subsequent imports
reuse the cached extension.

Set `WARP_DECODE_VERBOSE=1` to see the `nvcc` invocation on rebuild:

```bash
WARP_DECODE_VERBOSE=1 python tests/test_gate_up_correctness.py
```

### Benchmark

```bash
# Quick comparison (one call per CUDA Graph).
python benchmark/bench.py --mode all --batches 1,2,4,8,16,32,64 --warmup 25 --iters 200

# One mode only, custom batch set
python benchmark/bench.py --mode e2e --batches 1,8,16,32,64

# Custom shape (e.g. TP=4)
python benchmark/bench.py --tp-size 4 --batches 1..32
```

The script prints the SGLang version and selected configurations. SGLang logs
any configuration fallback. The isolated down benchmark includes routing and
uses initialized activations for both implementations. For a down configuration
that requires sorted TMA activations, use `--mode e2e` to run SGLang's standard
end-to-end path.

### Python API

```python
from warp_decode import warp_decode as wd

# gate_up + silu_and_mul (fused)
buf = wd.moe_gate_up_3d_batched(
    x,               # [B, hidden]         bf16
    gate_up_w,       # [E, 2*inter, hidden] bf16
    topk_ids,        # [B, topk]           int32
)                    # -> [B*topk, inter]  bf16

# down + topk-weighted sum-reduce (fused)
out = wd.moe_down_3d_batched(
    buf,             # [B*topk, inter]     bf16
    down_w,          # [E, hidden, inter]  bf16
    topk_ids,        # [B, topk]           int32
    topk_scale,      # [B, topk]           fp32
    B,               # int
)                    # -> [B, hidden]      bf16
```

### Shape constraints

| kernel   | constraint                                                     |
| -------- | -------------------------------------------------------------- |
| gate_up  | `hidden % 256 == 0` (32 lanes × VEC=8), `inter % 8 == 0`       |
| down     | `inter % 64 == 0` (32 lanes × VEC=2), `hidden % 8 == 0`        |

Both kernels assume the standard weight layouts used by sglang's fused_moe.

---

## Future work

**Consumer GPUs with multi-precision support (sm_120 and similar).**
Local / on-device deployment is exactly the setting where small-batch,
latency-bound decode kernels are most valuable — batch sizes are often 1–4,
and every millisecond of TTFT/ITL is user-visible. Extending warp_decode to
GeForce-class Blackwell (sm_120) with the full precision matrix (BF16, FP16,
FP8, MXFP8, INT8, INT4) would let a single-GPU workstation run frontier MoEs
at interactive latency. Concrete next steps:
- On-the-fly dequant inside the warp loop — the parallelism layout does not
  change, only the innermost read+convert.
- Group / per-channel / MX scale handling that stays in registers to avoid
  extra smem traffic.
- Autotune the same shape knobs (`WARPS_PER_CTA`, `OUTS_PER_WARP`) for
  smaller SM counts and different L2/HBM ratios.

**Same idea on non-NVIDIA accelerators.**
The roofline argument that motivates warp_decode is architecture-agnostic:
for `B * topk / E_local < ridge_point`, any expert-centric grouped GEMM is
wasting work, and an output-centric kernel that avoids all routing/staging
should win. Ports worth trying:
- **AMD (CDNA3 / RDNA):** HIP is close enough to CUDA that the two kernels
  should translate almost mechanically; the CDNA3 memory hierarchy has a
  lower ridge point, which extends the useful batch range further.
- **Ascend NPU (Cube / Vector cores):** the equivalent of the "one warp per
  output" mapping is one Vector core per output group, feeding the Cube
  core with weight tiles. The same structural argument — skip routing,
  fold the topk-weighted sum inside the compute unit — should transfer.
- **Other tile-based accelerators (Trainium, TPU, etc.):** any device where
  small-batch decode is memory-bound is a candidate.


---

## Acknowledgements

- [Cursor's warp decode blog](https://cursor.com/blog/warp-decode) —
  the original idea.
- [sglang](https://github.com/sgl-project/sglang) — the reference
  implementation used for correctness and performance comparisons.
