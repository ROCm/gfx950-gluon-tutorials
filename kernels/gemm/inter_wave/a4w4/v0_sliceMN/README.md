# v0_sliceMN — 8-wave warp-pipeline MXFP4, M/N-sliced

<p align="center">
  <img src="images/maturity_radar.png" alt="8-wave a4w4 v0_sliceMN optimization maturity" width="300">
</p>

**Optimization maturity (rough).** Axes — codegen, global latency, LDS latency, LDS bank conflict, scheduling, L2 locality — are defined in the [`v0_naive` README](../../../intra_wave/a16w16/v0_naive/README.md); the polygon vs the dashed "optimal" envelope shows how mature this kernel is.


## 1. What changed from the 4-wave kernel

`v0_sliceMN` is the **MXFP4 (4-bit) port** of the 8-wave
[`inter_wave/a16w16`](../../a16w16/README.md), carrying the scale pipeline of the 4-wave
[`a4w4/v1_sliceMN`](../../../intra_wave/a4w4/v1_sliceMN/README.md). For the general mechanics of
turning a 4-wave kernel into an 8-wave one — restaging each region's `mfma` and memory into
separate `warp_pipeline_stage` clusters, the `warpsPerCTA = [2,2] → [2,4]` warp-grid change, and
where `async_wait` lands — see
[`inter_wave/a16w16` §2](../../a16w16/README.md#2-what-changes-from-the-4-wave-kernel). This README
covers only what is different for MXFP4.

The 256×256 output tile is split into a **2×2 grid of `[128×128]` quadrants**, each operand
half-tile in its own double-buffered LDS allocation, and the hot loop runs **8 warps → 2
waves/SIMD** in ping-pong:

```
acc_tl += DOT(A_top, B_left)     acc_tr += DOT(A_top, B_right)
acc_bl += DOT(A_bot, B_left)     acc_br += DOT(A_bot, B_right)
```

The MXFP4-specific twist in the loop is the **scale pairing**: every `AC X` async-copies `X`
**and** `X_sc` in one commit group, and every `LR X` local-reads `X` **and** `X_sc`, so each
quadrant's e8m0 scale is in registers right before its `mfma_scaled` (`instr_shape=[16,16,128]`,
`k_width=16`, lowering to `v_mfma_scale_f32_16x16x128_f8f6f4 … cbsz:4 blgp:4`, ~2× the BF8 rate).

## 2. Revisit of layout changes

The 4→8-wave change is the same `warpsPerCTA = [2,2] → [2,4]` warp-grid promotion described in
[`inter_wave/a16w16` §2](../../a16w16/README.md#2-what-changes-from-the-4-wave-kernel): the
global-load layouts gain one extra warp base, while the shared / dot-operand / MFMA layouts are
warp-count-independent and reused verbatim. For a16w16 that is the whole story. For MXFP4 it is
not — the **B scale** is where the extra warp dimension bites, and it is what makes this baseline
generate `ds_read_u8` + `v_perm` inside the loop.

Every tile carries an e8m0 scale that streams **straight into LDS** (`buffer_load_to_shared`, no
`ds_write`) and is read back with the MFMA scale layout just before its DOT.

**The A scale is fine; the B scale is the problem.** The MFMA scale operand is delivered by a
`ds_read_b64_tr_b8` hardware-transpose read, which needs **8 bytes/thread**. The scale operand
layout is derived from the dot-operand layout, so it inherits the warp tiling: the A scale is
tiled by `WARPS_M=2` (unchanged from 4-wave), but the B scale is tiled by `WARPS_N=4`. A `[128,8]`
B-scale half therefore gives each thread only **4 bytes** — below the 64-bit transpose-read width
— so it degrades to **per-byte `ds_read_u8` + `v_perm`** byte-shuffle reassembly. In this kernel
that is **118 `v_perm` + 32 `ds_read_u8`** in the loop, purely for the B scale — VALU work that
stalls between MFMAs and inflates register pressure.

This kernel (v0) N-slices the B scale into `b_sc_left` / `b_sc_right` `[128,8]` halves like
everything else, and pays that cost. [`v1_combineBsc`](../v1_combineBsc/README.md) instead loads
the full `[256,8]` B scale as one combined buffer so it transpose-reads with **no `v_perm`**; see
the [family README §3](../README.md#3-the-b-scale-bottleneck-and-how-v1-fixes-it) for that fix.

### 2.1 Scale global-load must stay dword-granular

The scale half-tile is `[128, 8]` uint8 = 1024 bytes. `buffer_load_to_shared` (the LDS DMA)
requires **4 bytes/thread** — but an 8-warp CTA has 512 threads while 1024 bytes is only 256
dword-threads. So `blocked_scales_half` keeps `sizePerThread=[4,1]` and sets `warpsPerCTA=[2,4]`,
which **over-covers M by 2×**: 256 threads issue the dword loads, 256 are masked. A layout that
instead halves to 2 bytes/thread fails to lower (`LLVM Translation failed for
unrealized_conversion_cast`). Only `warpsPerCTA` changes from the 4-wave `[1,4]`.

## 3. Performance

MI355X, gfx950, 4096×4096, MXFP4, Triton `gfx950-tutorial-v1.1`, rocprof cold-rotating
(`--rotating-buffer-size 2048` for K ≥ 16384). This **8-wave, no-AGPR** kernel
(`scripts/collect_perf.py`) vs the 4-wave
[`intra_wave/a4w4/v1`](../../../intra_wave/a4w4/v1_sliceMN/README.md) reference
(`scripts/run_perf_table.py --configs llir+force-agpr+amdgcnas --rocprof`):

| K | this kernel TFLOPS | this kernel MFMA eff | `intra_wave/a4w4 v1` TFLOPS | `intra_wave/a4w4 v1` MFMA eff |
|---|---|---|---|---|
| 8192  | 3673 | ~65% | **4549** | **93.3%** |
| 16384 | 4140 | ~65% | **4948** | **93.0%** |
| 32768 | 4237 | ~66% | **5176** | **92.1%** |

VGPRs / spills: this kernel **256 / 29** (hot loop **0**); `intra_wave/a4w4 v1` **492 / 0**. ATT
traces for this kernel at K=8192 / 16384 / 32768 are under `/data/att_inter_a4w4_v0_sliceMN_K*`.

The hot loop is spill-free but reaches only **~65% loop MFMA efficiency**, and this baseline
**trails the tuned 4-wave `intra_wave/a4w4/v1`** (~93% MFMA) on TFLOPS at every K. Two reasons:
the B-scale byte-shuffle (118 `v_perm` + 32 `ds_read_u8`, §2) keeps the loop
LDS/scale-throughput-bound rather than latency-bound — so the 8-wave's ping-pong latency-hiding
has little to hide — and the halved 8-wave VGPR budget (256 vs 492) is a real cost. Eliminating the
B-scale `v_perm` in [`v1_combineBsc`](../v1_combineBsc/README.md) closes most of the gap; see the
[family README §4](../README.md#4-performance) for the full v0 → v1 → v2 comparison.

```bash
# correctness + do_bench TFLOPS (from this v0_sliceMN dir)
python ../bench.py --version 0 --K 8192

# rocprof cold-rotating TFLOPS + MFMA eff + VGPR/spill (from the repo root)
python scripts/collect_perf.py --kernel a4w4 --version 0 --K 8192
```
