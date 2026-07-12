# v0_sliceMN — 8-wave warp-pipeline MXFP4, M/N-sliced (BLOCK_K=256, 2-buffer)

> [!NOTE]
> This is the **baseline**. It N-slices the B scale into `[128,8]` halves, which at 8 warps
> forces the B-scale MFMA operand through per-byte `ds_read_u8` + `v_perm` reassembly (118
> `v_perm` in the loop). [`v1_combineBsc`](../v1_combineBsc/README.md)
> loads the B scale as one combined `[256,8]` so it transpose-reads with no `v_perm`, for
> **+16–22% TFLOPS**. See the [family README §2](../README.md#2-the-b-scale-bottleneck-and-how-v1-fixes-it).

## 1. Directory Structure

```
v0_sliceMN/
├── matmul_kernel.py    # The kernel implementation
└── README.md           # This file
```

## 2. What this is

The **MXFP4 (4-bit) port** of
[`inter_wave/a16w16`](../../a16w16/README.md),
carrying the scale pipeline of the 4-wave
[`a4w4/v1_sliceMN`](../../../intra_wave/a4w4/v1_sliceMN/README.md). The 256×256 output tile is split
into a **2×2 grid of `[128×128]` quadrants**, each operand half-tile in its **own**
double-buffered LDS allocation, and the hot loop is scheduled at the wave level with
`warp_pipeline_stage` running **8 warps → 2 waves/SIMD** in ping-pong.

```
acc_tl += DOT(A_top, B_left)     acc_tr += DOT(A_top, B_right)
acc_bl += DOT(A_bot, B_left)     acc_br += DOT(A_bot, B_right)
```

The loop is unrolled 2× → **8 mfma regions + 8 mem regions**, each wrapped in
`warp_pipeline_stage` with `cdna4_async.wait_group(5)` before the mfma region. Every
`AC X` issues `AC X` **and** `AC X_sc` in one commit group, and every `LR X` issues
`LR X` **and** `LR X_sc`, so each quadrant's e8m0 scale is in registers right before its
`mfma_scaled`.

## 3. The 4-bit deltas

Relative to the fp16 8-wave kernel, the numerics changes match those between the 4-wave
`a16w16/v8_sliceMN` and `a4w4/v1_sliceMN`:

- **Packed FP4 operands, `BLOCK_K=256`.** A/B tiles are `[128,128]` uint8 (two e2m1
  nibbles per byte = 256 logical K), the same 128 B/row LDS footprint as the fp16 tiles.
- **`mfma_scaled(a, a_sc, "e2m1", b, b_sc, "e2m1", acc)`**, `instr_shape=[16,16,128]`,
  `k_width=16`, with e8m0 scale operands. Lowers to
  `v_mfma_scale_f32_16x16x128_f8f6f4 … cbsz:4 blgp:4` (FP4, ~2× the BF8 rate).
- **B `.permute([1,0])` on the LDS load** — B is stored `(N, K//2)`, so the tile is read
  back transposed to feed the MFMA as a logical `(K, N)` operand.
- **Output bf16.**

The 8-wave global-load layouts are the 4-wave a4w4 layouts with one register base promoted
to a **third warp base** (`warpsPerCTA=[2,4]`). The padded tile shared layouts and the
identity scale shared layout are warp-independent and reused verbatim.

### 3.1 Scale global-load must stay dword-granular

The scale half-tile is `[128, 8]` uint8 = 1024 bytes. `buffer_load_to_shared` (the LDS
DMA) requires **4 bytes/thread** — but an 8-warp CTA has 512 threads while 1024 bytes is
only 256 dword-threads. So `blocked_scales_half` keeps `sizePerThread=[4,1]` and sets
`warpsPerCTA=[2,4]`, which **over-covers M by 2×**: 256 threads issue the dword loads, 256
are masked. A layout that instead halves to 2 bytes/thread fails to lower
(`LLVM Translation failed for unrealized_conversion_cast`). Only `warpsPerCTA` changes from
the 4-wave `[1,4]`.

## 4. Load-side pointer-walk — required to fit 256 VGPR

At 8 waves the per-wave budget is **256 VGPR** (half the 4-wave's 512), and the four live
f32 `[128×128]` accumulators alone are 128 VGPR. The MXFP4 loop also needs the packed-FP4
dot operands, the scale operands, **and** their global offset tensors. Keeping a separate
`[128×128]` offset tensor per quadrant × K-buffer (8 tile + 8 scale tensors) overflowed the
budget and spilled the accumulators **inside the loop** (30 spills, ~40% MFMA eff).

The fix is a **load-side pointer-walk**: all four A/B quadrants and both K-buffers share
**one** within-tile offset tensor each (`a_tile_offsets`, `b_tile_offsets`, `a_sc_offsets`,
`b_sc_offsets`); the top/bot, left/right, and even/odd(`_next`) variants are reached by
adding **scalar** (uniform) deltas to the base pointer — `a_base + a_half_m`,
`a_base + a_k2`, etc. — instead of holding a distinct offset tensor per variant. This
collapses 8 tile + 8 scale offset tensors to **2 + 2**, which:

- takes the **hot loop to 0 spills** (all residual spills move to the epilogue), and
- lifts K=8192 from ~2841 → **3652 TFLOPS** (+28%).

This is the same idea as the fp16 kernel's *store-side* pointer-walk, applied to the load
side because MXFP4's extra scale operands make the loop, not just the epilogue,
register-bound.

## 5. Performance

MI355X, gfx950, 4096×4096, MXFP4, no-AGPR, Triton `gfx950-tutorial-v1.0`, rocprof cold-rotating:

| Metric | Value |
|---|---|
| Correctness vs torch | ✅ PASS (K 1024…65536) |
| rocprof TFLOPS (cold) | 3525 @ K=8192, **4064** @ K=32768 |
| MFMA efficiency (per-SIMD), loop-only | ~57% |
| VGPRs / spills | 256 / 23 (loop: **0**) |
| vs 4-wave a4w4 base | matches at large K (8-wave 4064 @ K=32768 ≈ 4-wave base 4101 @ K=8192) |

The loop is spill-free and reaches ~57% loop MFMA efficiency — comparable to the 4-wave
base's ~61%. Unlike a16w16/a8w8, the 8-wave does **not** beat the tuned 4-wave for MXFP4:
the loop is LDS/scale-throughput-bound rather than latency-bound, so the 8-wave's
ping-pong latency-hiding has little to hide, while its halved VGPR budget is a real cost.
See the [family README](../README.md#3-performance) for the full comparison.

```bash
# correctness + do_bench TFLOPS
python ../bench.py --version 0 --K 8192

# rocprof cold-rotating TFLOPS + VGPR/spill
python ../collect_perf.py --version 0 --K 8192
```
