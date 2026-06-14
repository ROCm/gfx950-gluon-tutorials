# MXFP4 GEMM Kernel — v1_sliceMN

`v1_sliceMN` is a refined MXFP4 kernel that makes **two independent changes on
top of [v0_sliceN](../v0_sliceN/README.md)**. Read v0's README first — it
introduces all the MXFP4 fundamentals (e8m0 scales, `mfma_scaled`, the scale
layout, `ds_read_tr`) that carry over unchanged here. This document only
covers the two deltas and the final performance.

## 1. Two changes over v0

### 1.1 Simpler scale pipeline: drop the LDS round-trip

v0 moves each scale tile through **three** steps:

```
buffer_load (global → registers) → local_store (ds_write) → local_load (ds_read)
```

The register-staged `buffer_load` lands the scales in VGPRs, then `local_store`
writes them to LDS, then `local_load` reads them back in the layout the MFMA
wants. This costs a register round-trip, a `ds_write` per scale tile, and the
four-deep register double-buffering needed to keep the scales alive across the
pipeline (v0 README §3.4–3.6, including the surprise that `ds_write` can take
~400 cycles).

v1 collapses this to **two** steps:

```
buffer_load_to_lds (global → LDS directly) → local_load (ds_read)
```

`gl.amd.cdna4.async_copy.buffer_load_to_shared` streams each scale tile
straight into LDS with no VGPR staging and **no `ds_write`**. Each half-tile
scale is 128×8 uint8 = 1024 bytes = 4 bytes/thread, so it lowers to a single
`buffer_load_dword ... lds` that rides right behind its input tile's
`buffer_load_dwordx4` in the same commit group. The scale then waits in LDS
until the `smem.load(scale_layout)` (`ds_read_tr`) just before its DOT.

Beyond being simpler, removing `ds_write` matters for scheduling: the **LLIR
scheduler no longer has to reason about `ds_write` in the hot loop**, and tiles
and scales share commit groups so the waitcnt schedule needs only one count.

### 1.2 Symmetric tiling: slice both M and N

v0 slices the output tile along **N only** (`B` split into `B_left` / `B_right`,
`A` kept whole). v1 slices along **both M and N** — `A` into `A_t` / `A_b`,
`B` into `B_l` / `B_r` — producing a 2×2 grid of 128×128 accumulator quadrants
(`C_tl`, `C_tr`, `C_bl`, `C_br`).

This is the M+N slicing pattern from
[a16w16/v8_sliceMN](../../a16w16/v8_sliceMN/README.md). It gives a more
symmetric kernel with a more balanced `buffer_load` distribution across the
four quadrants, which — as the a16w16 series shows — is more effective at large
K (less per-quadrant register pressure, cleaner natural-pipeline epilogue).

## 2. Pipeline

![v1 pipeline design](../images/mxfp4_v1_pipeline_design.png)

Each K-step runs four regions (one DOT + one LR + one AC each), and the loop is
unrolled by 2 so LDS buffer indices alternate without runtime computation. With
the §1.1 change, every `AC X` issues `AC X` **and** `AC X_sc` in one commit
group, and every `LR X` issues `LR X` **and** `LR X_sc` — so the scale for each
quadrant is always in registers right before its DOT.

## 3. Performance

Measured on MI355, 4096×4096×K, rocprof timing (1000 dispatches, last-100
average), one config per invocation:

| Config (K=32768) | v0_sliceN | v1_sliceMN | v1 MFMA Eff. |
|------------------|-----------|------------|--------------|
| base | 4258 | 4589 | 60.4% |
| llirSched | 4780 | 4892 | 70.6% |
| llirSched + amdgcnas | 5265 | 5387 | 94.5% |
| llirSched + amdgcnas + nobar | 5311 | **5419** | 96.0% |

`nobar` adds `TRITON_ENABLE_AMDGCN_AS_REMOVE_BARRIER=1`, which keeps only the
first `s_barrier` per loop iteration. The four waves progress uniformly in this
kernel, so the remaining per-region barriers are redundant — verified correct
for both versions; do not assume for other kernels.

At K=65536: llirSched + amdgcnas 5390 (v0) / 5541 (v1); + nobar 5457 / **5583**
TFLOPS (94.8% MFMA eff).

## 4. How to Run

From the `a4w4` directory:

```bash
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
python bench.py --version 1

# Full table (v0 vs v1, all configs):
python ../../../scripts/run_perf_table.py --kernel a4w4 --versions 0 1 \
  --configs base llir llir+amdgcnas llir+amdgcnas+nobar --K 32768 --rocprof
```
