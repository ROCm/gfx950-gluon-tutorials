# inter_wave/a8w8 — 8-wave warp-pipeline BF8 (a8w8) GEMM (gfx950)

<p align="center">
  <img src="images/maturity_radar.png" alt="8-wave a8w8 optimization maturity" width="300">
</p>

**Optimization maturity (rough).** Axes — codegen, global latency, LDS latency, LDS bank conflict, scheduling, L2 locality — are defined in the [`v0_naive` README](../../intra_wave/a16w16/v0_naive/README.md); the polygon vs the dashed "optimal" envelope shows how mature this kernel is.


An **8-wave** (8 warps/CTA → **2 waves/SIMD**) BF8/e5m2 GEMM for gfx950 / MI350X /
MI355X. It is the **8-bit sibling** of [`inter_wave/a16w16`](../a16w16/README.md): the same
`sliceMN` warp-pipeline solution with the 16-bit numerics swapped for 8-bit. Read the
[`inter_wave/a16w16` README](../a16w16/README.md) first for the 8-wave design and loop
structure — this document covers the 8-bit deltas and the measured performance.

## 1. What changes from 16-bit to 8-bit

The 8-wave skeleton — 8 warps `[2,4]`, the 2×2 `[128×128]` quadrant slicing with four
separate double-buffered LDS allocations, the `warp_pipeline_stage` ping-pong schedule,
no-AGPR, and the spill-free store-side pointer-walk epilogue — is **identical** to
[`inter_wave/a16w16`](../a16w16/README.md). Only the numerics move from 16-bit to 8-bit,
exactly as they do between the 4-wave `a16w16/v8_sliceMN` and `a8w8` kernels:

| | `inter_wave/a16w16` (16-bit) | **`inter_wave/a8w8` (8-bit)** |
|---|---|---|
| Operand dtype | fp16 / bf16 (2 B) | **BF8 / float8_e5m2 (1 B)** |
| `BLOCK_K` | 64 | **128** (same 128 B / row into LDS) |
| MFMA | `mfma` `[16,16,32]` | **`mfma_scaled(…,"e5m2",…,"e5m2")` `[16,16,128]`** |
| dot-operand `k_width` | 8 | **32** |
| Per-block scale | — | none (`scale=None`; plain BF8 GEMM) |
| Output C | fp16 / bf16 | **fp16** |

The MFMA is `mfma_scaled(a, None, "e5m2", b, None, "e5m2", acc)` with
`instr_shape=[16,16,128]` and dot-operand `k_width=32`. Scales are `None` — this is a plain
BF8 GEMM. On gfx950 it lowers to `v_mfma_scale_f32_16x16x128_f8f6f4 … cbsz:1 blgp:1` (the
BF8 format select). Two 8-wave-specific details carry over unchanged from the fp16 kernel:
the global-load layouts are the 4-wave a8w8 layouts with **one register base promoted to a
third warp base** (the extra warp dim, since `warpsPerCTA` goes `[2,2]→[2,4]`), and the
padded shared layouts are warp-independent so they are reused **verbatim** from the 4-wave
a8w8 kernel. B is pre-transposed to `(N, K)` and fed as a logical `(K, N)` operand so K is
contiguous for the async copy (no `permute` on the LDS load — the same as fp16).

The loop is unrolled 2× → **8 mfma regions + 8 mem regions**, each wrapped in
`warp_pipeline_stage` with `cdna4_async.wait_group(5)` placed **before** the mfma region.
Because the four LDS allocations are separate, the membar disambiguates the load-read (LR)
from the async-copy write (AC) by allocation — the loads stay plain (non-relaxed) and carry
no extra `s_barrier`. The **hot loop is spill-free** (~99.7% MFMA); the store-side
pointer-walk + de-interleaved epilogue keep the four live `[128×128]` accumulators inside
the 256-VGPR budget, leaving only 13 residual spills in the `convert_layout` + store
epilogue (which carries no MFMA).

## 2. Performance

MI355X, gfx950, 4096×4096, BF8, rocprof cold-rotating (last-100 average of 1000 dispatches;
`--rotating-buffer-size 2048` for K ≥ 16384), Triton `gfx950-tutorial-v1.1`. The 8-wave kernel
(`scripts/collect_perf.py`, **no-AGPR**) vs the 4-wave `intra_wave/a8w8` reference
(`scripts/run_perf_table.py --configs llir+force-agpr+amdgcnas --rocprof`):

| K | 8-wave TFLOPS | 8-wave MFMA eff | 4-wave TFLOPS | 4-wave MFMA eff |
|---|---|---|---|---|
| 8192  | 2901 | 99.7% | **3043** | 99.5% |
| 16384 | 3076 | 99.8% | **3233** | 99.5% |
| 32768 | 3054 | 99.0% | **3059** | 97.6% |

(4-wave = `llir+force-agpr+amdgcnas`; MFMA eff is per-SIMD loop-only, a single-dispatch ATT
reading — treat the last digit as noise.)

Both loops are genuinely MFMA-bound (~99.5–99.8% at K ≤ 16384), but the tuned 4-wave
`llir+force-agpr+amdgcnas` leads on TFLOPS at every K (**~+5%** at K ≤ 16384, near-parity at
K=32768): its AGPR-form accumulators and LLIR schedule extract slightly more throughput than
the 8-wave's no-AGPR VGPR budget. The 8-wave's **hot loop is spill-free** (256 VGPR, 8 residual
spills all in the `convert_layout` + store epilogue); the 4-wave runs at 488 VGPR / 0 spills.

## 3. Running

```bash
# correctness + do_bench TFLOPS (from this kernel dir)
python bench.py --K 8192

# rocprof cold-rotating TFLOPS + MFMA efficiency (ATT) + VGPR/spill (from the repo root)
python scripts/collect_perf.py --kernel a8w8 --K 8192

# large K needs a bigger rotating buffer to stay cold
python scripts/collect_perf.py --kernel a8w8 --K 32768 --rotating-buffer-size 2048
```

Inputs are BF8 (`float8_e5m2`); the output is fp16. Drop `--K` to sweep all sizes. Clear
`~/.triton/cache` (or use `TRITON_ALWAYS_COMPILE=1`) after editing the kernel.

## 4. Files

- `matmul_kernel.py` — the kernel; exposes `a8w8_kernel` (the jit kernel),
  `matmul_kernel_only` / `matmul` (launch wrappers), `MIN_K`, `KERNEL_NAME`.
- `get_pids` (XCD-aware PID remap + `GROUP_SIZE_M` swizzle) is imported from the shared
  [`kernels/gemm/utils/common.py`](../../utils/common.py).
- `bench.py` — correctness (vs dequantized `torch.matmul`) + do_bench TFLOPS + `--rocprof`
  rotating-tensor mode.
- Perf is collected with the shared [`scripts/collect_perf.py`](../../../../scripts/collect_perf.py)
  (`--kernel a8w8`).
