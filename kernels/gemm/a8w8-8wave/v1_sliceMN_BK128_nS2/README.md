# v1_sliceMN_BK128_nS2 — 8-wave warp-pipeline BF8, M/N-sliced (BLOCK_K=128, 2-buffer)

## 1. Directory Structure

```
v1_sliceMN_BK128_nS2/
├── matmul_kernel.py    # The kernel implementation
└── README.md           # This file
```

## 2. What this is

The **8-bit port** of [`a16w16-8wave/v1_sliceMN_BK64_nS2`](../../a16w16-8wave/v1_sliceMN_BK64_nS2/README.md).
The 256×256 output tile is split into a **2×2 grid of `[128×128]` quadrants**, each
operand half-tile in its **own** double-buffered LDS allocation
(`smemA_top/bot`, `smemB_left/right`), and the hot loop is scheduled at the wave level
with `warp_pipeline_stage` running **8 warps → 2 waves/SIMD** in ping-pong. Everything
structural is copied from the fp16 kernel; only the numerics are 8-bit.

```
acc_tl += DOT(A_top, B_left)     acc_tr += DOT(A_top, B_right)
acc_bl += DOT(A_bot, B_left)     acc_br += DOT(A_bot, B_right)
```

The loop is unrolled 2× → **8 mfma regions + 8 mem regions**, each wrapped in
`warp_pipeline_stage` with `cdna4_async.wait_group(5)` placed **before** the mfma region.
Because `smemA_top`, `smemA_bot`, `smemB_left`, `smemB_right` are four *separate*
allocations, the membar disambiguates the load-read (LR) from the async-copy write (AC)
by allocation — the loads stay plain (non-relaxed) and carry no extra `s_barrier`.

## 3. The 8-bit deltas

Relative to the fp16 kernel, the changes are exactly the ones that separate the 4-wave
`a16w16/v8_sliceMN` from `a8w8`:

- **BF8 operands, `BLOCK_K=128`.** Each `[BLOCK_M//2, BLOCK_K] = [128,128]` A half-tile
  and `[BLOCK_K, BLOCK_N//2] = [128,128]` B half-tile is 128 bytes/row — the same LDS
  footprint per K-step as the fp16 `[128,64]` tiles, so the shared layouts and buffering
  are unchanged.
- **`mfma_scaled(a, None, "e5m2", b, None, "e5m2", acc)`** with `instr_shape=[16,16,128]`
  and dot-operand `k_width=32`. Scales are `None` — this is a plain BF8 GEMM. On gfx950
  this lowers to `v_mfma_scale_f32_16x16x128_f8f6f4 … cbsz:1 blgp:1` (the BF8 format
  select).
- **Output fp16.** The store downcasts f32→fp16 before `convert_layout`.

The 8-wave global-load layouts are the 4-wave a8w8 layouts with one register base
promoted to a **third warp base** (`warpsPerCTA=[2,4]` = 8 warps tiling M for A, N for B).
The padded shared layouts are reused verbatim. B is pre-transposed to `(N, K)` so K is
contiguous — no `permute` on the LDS load.

## 4. Epilogue: spill-free store

At 8 waves there are **2 waves/SIMD**, so the per-wave budget is 256 VGPR and the four
live f32 `[128×128]` accumulators alone are 128 VGPR. The same two tricks the fp16 kernel
uses keep the **hot loop spill-free**:

1. **Store-side pointer-walk** — one shared within-quadrant offset tensor + four **scalar**
   base pointers, instead of four full `[128×128]` offset tensors.
2. **De-interleaved epilogue** — finish all four final MFMAs *before* the convert+store
   phase, so the dot operands die first and only the four accumulators stay live during
   the stores.

The loop runs at ~99.7% MFMA; the 13 residual spills are all in the epilogue.

## 5. Performance

MI355X, gfx950, 4096×4096, BF8, no-AGPR, rocprof cold-rotating:

| Metric | Value |
|---|---|
| Correctness vs torch | ✅ PASS (K 512…32768) |
| rocprof TFLOPS (cold) | **2841** @ K=8192, **3097** @ K=32768 |
| MFMA efficiency (per-SIMD), loop-only | **~99.7%** |
| VGPRs / spills | 256 / 13 (loop: **0**) |
| vs 4-wave a8w8 | +9% over base, +3.5% over best (llir+amdgcnas) |

```bash
# correctness + do_bench TFLOPS
python ../bench.py --version 1 --K 8192

# rocprof cold-rotating TFLOPS + MFMA efficiency (ATT) + VGPR/spill
python ../collect_perf.py --version 1 --K 8192
```
