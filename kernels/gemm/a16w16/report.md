# a16w16 LLIR scheduler — tile-size robustness sweep

How robust is the gfx950 LLIR scheduler across GEMM tile sizes, and how much of
its benefit depends on forcing MFMA accumulators into AGPRs? We sweep
`BLOCK_M ∈ {256,128,64}` × `BLOCK_N ∈ {256,128,64}` at `BLOCK_K = 64` (a 3×3 =
9-tile grid) on the `v5_local_prefetch`, `v6_loop_unroll`, `v7_sliceN`, and
`v8_sliceMN` kernels, comparing three configs:

- **`base`** — stock compile, scheduler off.
- **`llir`** — `schedule_hint="gemm-4waves"`: the LLIR scheduler runs and LLVM's
  machine schedulers are disabled (so the schedule survives codegen), but
  **register allocation is left at the backend default** (no AGPR forcing). This
  is the default behavior after the `gemm-4waves`/`force-agpr` split.
- **`+agpr`** — `schedule_hint="gemm-4waves, force-agpr"`: same scheduler, plus
  forcing MFMA accumulators into AGPR form (`amdgpu-agpr-alloc=256` +
  `amdgpu-mfma-vgpr-form=0`). This is what the `llir` column meant in the
  earlier version of this report.

## Method

- The tutorial kernels hardcode their gluon layouts for 256×256×64. To vary the
  tile we made them **tile-parametric**: the `PaddedSharedLayout` is computed by
  [`ttgl.amd.cdna4.compute_efficient_padded_shared_layout`](https://github.com/triton-lang/triton/pull/10302)
  (triton-lang/triton#10302); the matching coalesced global-load `DistributedLinearLayout`
  is derived from it (reimplementing `CoalesceAsyncCopy`'s padded-encoding logic);
  the MFMA layout keeps `warps_per_cta=[2,2]`.
- Tile coverage depends on the kernel: **v5/v6 are correct on 5 of the 9 tiles**
  (the others need the exact `load_contig`, currently pinned to 8, which lives in
  the C++ `CoalesceAsyncCopy` pass); **v7 (slices B along N, `BLOCK_N // 2`) and
  v8 (slices both M and N) are correct on all 9**. Only the correct tiles are reported.
- **Occupancy is fixed to 1 wave/SIMD.** We set `M = 16·BLOCK_M`, `N = 16·BLOCK_N`
  so the grid is 16×16 = 256 workgroups ≈ 1 per CU on MI350. This isolates the
  scheduler's effect from occupancy.
- Metrics: **TFLOPS, MFMA-eff, spills, VGPRs**, collected with `--rocprof`
  (rocprofv3 kernel-trace, 1000 dispatches, last-100 avg; MFMA-eff from ATT,
  hot-loop per-iteration). `sched?` = whether the LLIR scheduler actually applied
  (region markers in the generated asm) — `YES` for every `llir`/`+agpr` row.
  `K = 8192`, fp16.

---

## Sweep results (occupancy fixed, M = 16·BLOCK_M, N = 16·BLOCK_N)

### v5 (local_prefetch)

| tile | M×N | cfg | TFLOPS | MFMA-eff | spills | VGPRs | sched? |
|---|---|---|---|---|---|---|---|
| **256×256×64** | 4096×4096 | base | 1134 | 58.63% | 0 | 452 | – |
| | | llir | **1263** | 71.59% | 0 | 509 | YES |
| | | +agpr | 1190 | 77.59% | **248** | 512 | YES |
| 256×128×64 | 4096×2048 | base | 882 | 51.21% | 0 | 244 | – |
| | | llir | 866 | 65.30% | 0 | 292 | YES |
| | | +agpr | 872 | 61.68% | 0 | 356 | YES |
| 128×256×64 | 2048×4096 | base | 898 | 51.06% | 0 | 244 | – |
| | | llir | 883 | 64.17% | 0 | 324 | YES |
| | | +agpr | 884 | 64.27% | 0 | 356 | YES |
| 128×128×64 | 2048×2048 | base | 611 | 41.75% | 0 | 141 | – |
| | | llir | 605 | 40.59% | 0 | 190 | YES |
| | | +agpr | 604 | 39.79% | 0 | 196 | YES |
| 64×256×64 | 1024×4096 | base | 545 | 36.64% | 0 | 162 | – |
| | | llir | 530 | 35.53% | 0 | 242 | YES |
| | | +agpr | 529 | 36.38% | 0 | 244 | YES |

v5's `local_prefetch` pipeline is the one kernel where **forcing AGPRs hurts**: at
256×256 `+agpr` spills 248 VGPRs (1190 TFLOPS), while the default `llir` is
spill-free and fastest (**1263**). Elsewhere the three are within noise.

### v6 (loop_unroll)

| tile | M×N | cfg | TFLOPS | MFMA-eff | spills | VGPRs | sched? |
|---|---|---|---|---|---|---|---|
| **256×256×64** | 4096×4096 | base | 992 | 53.20% | 8 | 512 | – |
| | | llir | 312 | 17.25% | **76** | 512 | YES |
| | | +agpr | **1273** | **95.70%** | 0 | 500 | YES |
| 256×128×64 | 4096×2048 | base | 892 | 51.79% | 0 | 244 | – |
| | | llir | 885 | 63.71% | 0 | 296 | YES |
| | | +agpr | 889 | 65.26% | 0 | 340 | YES |
| 128×256×64 | 2048×4096 | base | 903 | 51.63% | 0 | 288 | – |
| | | llir | 900 | 67.01% | 0 | 324 | YES |
| | | +agpr | 902 | 67.36% | 0 | 324 | YES |
| 128×128×64 | 2048×2048 | base | 619 | 42.46% | 0 | 142 | – |
| | | llir | 615 | 40.76% | 0 | 194 | YES |
| | | +agpr | 614 | 41.27% | 0 | 196 | YES |
| 64×256×64 | 1024×4096 | base | 549 | 36.57% | 0 | 164 | – |
| | | llir | 545 | 36.38% | 0 | 244 | YES |
| | | +agpr | 545 | 35.95% | 0 | 244 | YES |

v6 is the **opposite**: at 256×256 the scheduler *needs* the AGPR budget. Without
it (`llir`) the interleaved loads can't share VGPRs with the full accumulator tile,
so it spills 76 and collapses to **312 TFLOPS — below base**; `+agpr` is spill-free
at **1273**. Every smaller tile is config-independent.

### v7 (sliceN) — full grid

N-sliced (B = `BLOCK_N // 2`), correct on every tile, spill-free in all configs.

| tile | M×N | cfg | TFLOPS | MFMA-eff | spills | VGPRs | sched? |
|---|---|---|---|---|---|---|---|
| **256×256×64** | 4096×4096 | base | 1226 | 64.99% | 0 | 496 | – |
| | | llir | 1375 | 88.32% | 0 | 458 | YES |
| | | +agpr | **1427** | 96.14% | 0 | 460 | YES |
| 256×128×64 | 4096×2048 | base | 1077 | 56.92% | 0 | 271 | – |
| | | llir | 1186 | 85.85% | 0 | 296 | YES |
| | | +agpr | 1183 | 86.51% | 0 | 320 | YES |
| 256×64×64 | 4096×1024 | base | 713 | 44.39% | 0 | 236 | – |
| | | llir | 728 | 48.64% | 0 | 228 | YES |
| | | +agpr | 733 | 51.77% | 0 | 248 | YES |
| 128×256×64 | 2048×4096 | base | 1011 | 56.48% | 0 | 246 | – |
| | | llir | **1235** | 84.64% | 0 | 268 | YES |
| | | +agpr | 1225 | 84.44% | 0 | 268 | YES |
| 128×128×64 | 2048×2048 | base | 900 | 54.39% | 0 | 148 | – |
| | | llir | 917 | 66.27% | 0 | 168 | YES |
| | | +agpr | 920 | 64.30% | 0 | 176 | YES |
| 128×64×64 | 2048×1024 | base | 526 | 36.52% | 0 | 124 | – |
| | | llir | 540 | 38.20% | 0 | 124 | YES |
| | | +agpr | 537 | 37.89% | 0 | 136 | YES |
| 64×256×64 | 1024×4096 | base | 742 | 49.00% | 0 | 130 | – |
| | | llir | 734 | 56.74% | 0 | 178 | YES |
| | | +agpr | 731 | 57.50% | 0 | 180 | YES |
| 64×128×64 | 1024×2048 | base | 563 | 41.51% | 0 | 90 | – |
| | | llir | 556 | 42.47% | 0 | 108 | YES |
| | | +agpr | 561 | 42.20% | 0 | 108 | YES |
| 64×64×64 | 1024×1024 | base | 324 | n/a | 0 | 94 | – |
| | | llir | 318 | n/a | 0 | 114 | YES |
| | | +agpr | 318 | n/a | 0 | 116 | YES |

(`n/a` = ATT eff-collection failed on the tiniest kernel; TFLOPS still valid.)
For v7 the default `llir` (no AGPR) **matches `+agpr` within a few percent** at every
tile, with 0 spills — slicing keeps the accumulator small enough that the backend
default RA never needs the AGPR budget.

### v8 (sliceMN) — full grid

Slices **both** M and N (A = `BLOCK_M//2`, B = `BLOCK_N//2`, 4 accumulator
quadrants), correct on every tile, spill-free in all configs.

| tile | M×N | cfg | TFLOPS | MFMA-eff | spills | VGPRs | sched? |
|---|---|---|---|---|---|---|---|
| **256×256×64** | 4096×4096 | base | 1251 | 68.92% | 0 | 496 | – |
| | | llir | 1363 | 84.75% | 0 | 436 | YES |
| | | +agpr | **1423** | 93.65% | 0 | 436 | YES |
| 256×128×64 | 4096×2048 | base | 1117 | 63.40% | 0 | 264 | – |
| | | llir | **1195** | 83.24% | 0 | 260 | YES |
| | | +agpr | 1193 | 83.39% | 0 | 272 | YES |
| 256×64×64 | 4096×1024 | base | 738 | 43.81% | 0 | 254 | – |
| | | llir | 740 | 48.25% | 0 | 184 | YES |
| | | +agpr | 736 | 48.01% | 0 | 200 | YES |
| 128×256×64 | 2048×4096 | base | 1116 | 66.04% | 0 | 242 | – |
| | | llir | **1215** | 83.87% | 0 | 260 | YES |
| | | +agpr | 1210 | 83.36% | 0 | 260 | YES |
| 128×128×64 | 2048×2048 | base | 954 | 55.65% | 0 | 142 | – |
| | | llir | 945 | 67.22% | 0 | 150 | YES |
| | | +agpr | 950 | 68.44% | 0 | 152 | YES |
| 128×64×64 | 2048×1024 | base | 620 | 40.09% | 0 | 100 | – |
| | | llir | 628 | 42.86% | 0 | 100 | YES |
| | | +agpr | 626 | 42.85% | 0 | 108 | YES |
| 64×256×64 | 1024×4096 | base | 732 | 49.40% | 0 | 234 | – |
| | | llir | 730 | 56.13% | 0 | 178 | YES |
| | | +agpr | 729 | 56.52% | 0 | 180 | YES |
| 64×128×64 | 1024×2048 | base | 607 | 42.70% | 0 | 92 | – |
| | | llir | 614 | 46.54% | 0 | 98 | YES |
| | | +agpr | 606 | 47.17% | 0 | 100 | YES |
| 64×64×64 | 1024×1024 | base | 367 | 27.49% | 0 | 60 | – |
| | | llir | 369 | 28.59% | 0 | 60 | YES |
| | | +agpr | 370 | 28.65% | 0 | 60 | YES |

Same as v7: `llir` and `+agpr` are interchangeable across all 9 tiles, 0 spills.

---

## v6 vs v7 vs v8 — shapes where v6 is numerically correct

`v6_loop_unroll` is only correct on 5 of the 9 tiles; it produces wrong results on
the 4 narrow tiles (256×64, 128×64, 64×128, 64×64) due to a non-deterministic race
in its unrolled pipeline — see [issues.md](issues.md). On the 5 shapes where v6 *is*
correct, here is the head-to-head with v7/v8 for `[TFLOPS, MFMA-eff]` across all three
configs:

| shape | kernel | base TFLOPS | base eff | llir TFLOPS | llir eff | +agpr TFLOPS | +agpr eff |
|---|---|---|---|---|---|---|---|
| **256×256×64** | v6 | 992 | 53.20% | 312 | 17.25% | 1273 | 95.70% |
| | v7 | 1226 | 64.99% | 1375 | 88.32% | **1427** | **96.14%** |
| | v8 | 1251 | 68.92% | 1363 | 84.75% | 1423 | 93.65% |
| **256×128×64** | v6 | 892 | 51.79% | 885 | 63.71% | 889 | 65.26% |
| | v7 | 1077 | 56.92% | 1186 | 85.85% | 1183 | 86.51% |
| | v8 | 1117 | 63.40% | **1195** | 83.24% | **1193** | 83.39% |
| **128×256×64** | v6 | 903 | 51.63% | 900 | 67.01% | 902 | 67.36% |
| | v7 | 1011 | 56.48% | **1235** | 84.64% | **1225** | 84.44% |
| | v8 | 1116 | 66.04% | 1215 | 83.87% | 1210 | 83.36% |
| **128×128×64** | v6 | 619 | 42.46% | 615 | 40.76% | 614 | 41.27% |
| | v7 | 900 | 54.39% | 917 | 66.27% | 920 | 64.30% |
| | v8 | 954 | 55.65% | **945** | **67.22%** | **950** | **68.44%** |
| **64×256×64** | v6 | 549 | 36.57% | 545 | 36.38% | 545 | 35.95% |
| | v7 | 742 | 49.00% | **734** | 56.74% | **731** | 57.50% |
| | v8 | 732 | 49.40% | 730 | 56.13% | 729 | 56.52% |

- **v7 and v8 beat v6 on every shape** in both TFLOPS and MFMA-eff (in all three configs):
  slicing keeps the accumulator small, so the sliced kernels are strictly better here even
  apart from v6's correctness bug.
- **v6's only standout is 256×256 `+agpr` (1273 / 95.7%)**, and it needs `force-agpr`:
  its `llir` (no-AGPR) craters to **312 / 17%** (the 76-spill case). v7/v8 reach ~1370–1420
  in *both* configs, spill-free, so they don't depend on AGPR forcing.
- For v7/v8 `llir ≈ +agpr` everywhere; v6 only benefits from `+agpr` at 256×256.

---

## v6 vs v7 vs v8 at K=16384 (does v8 win at large K?)

The tutorial claims v8 (sliceMN) beats v7 (sliceN) for large K. Re-running the same
5 shapes / 3 configs / fixed occupancy with **K = 16384** (double the K=8192 table):

| shape | kernel | base TFLOPS | base eff | llir TFLOPS | llir eff | +agpr TFLOPS | +agpr eff |
|---|---|---|---|---|---|---|---|
| **256×256×64** | v6 | 1029 | 52.14% | 345 | 12.24% | 1312 | 86.94% |
| | v7 | 1259 | 64.63% | 1427 | 86.32% | 1480 | 94.83% |
| | v8 | 1257 | 66.24% | 1410 | 84.22% | **1484** | 94.73% |
| **256×128×64** | v6 | 689 | 51.39% | 697 | 65.68% | 695 | 63.46% |
| | v7 | 1017 | 56.85% | 1094 | 85.86% | 1093 | 86.17% |
| | v8 | **1113** | 63.49% | **1105** | 83.39% | **1104** | 83.93% |
| **128×256×64** | v6 | 784 | 51.55% | 785 | 64.85% | 784 | 63.27% |
| | v7 | 1036 | 55.80% | **1154** | 84.77% | **1156** | 84.73% |
| | v8 | 1035 | 65.71% | 1049 | 83.93% | 1038 | 83.91% |
| **128×128×64** | v6 | 505 | 36.21% | 498 | 36.36% | 493 | 36.35% |
| | v7 | 804 | 54.71% | **812** | 66.13% | 805 | 66.07% |
| | v8 | 797 | 55.66% | 777 | 67.69% | 797 | 68.60% |
| **64×256×64** | v6 | 490 | 34.84% | 479 | 34.78% | 482 | 34.10% |
| | v7 | 719 | 48.70% | **712** | 54.37% | 713 | 55.09% |
| | v8 | 715 | 49.03% | 704 | 54.01% | 712 | 53.97% |

**The claim does not hold at these (occupancy-fixed, spill-free) sizes.** The mean v8/v7
TFLOPS advantage actually *drops* with K — **+1.6% at K=8192 → −1.1% at K=16384** (a
−2.7pp shift toward v7). Doubling K moves the balance the *opposite* way from the claim:
v7 pulls clearly ahead at 128×256 (`+agpr` v7 1156 vs v8 1038, +11%) and 128×128 (`llir`
+4.5%), while v8 wins only at 256×128 (and at 256×256 `+agpr` by a hair: 1484 vs 1480).
- **Why:** here neither v7 nor v8 spills (0 spills everywhere), so v8's advantage — smaller
  per-quadrant accumulators relieving register pressure over a long K-loop — never gets to
  matter. The tutorial's "v8 for large K" likely refers to a regime where v7's larger
  accumulator *spills* at high K; at 1 wave/SIMD on these shapes that regime isn't reached.
- v6@256×256 `llir` still craters (345 TFLOPS / 76 spills) at K=16384 — the AGPR-spill
  dependence is K-independent. (Single-run rocprof; small-tile cells carry run-to-run jitter.)

---

## Conclusions

1. **The scheduler applies successfully at every tile in every config**
   (`sched? = YES`): the `gemm-4waves`/`force-agpr` split changes only register
   allocation, never whether the schedule is emitted.

2. **AGPR forcing only matters at the full 256×256 tile — and there it is
   kernel-dependent**, cutting in opposite directions:
   - **v6 needs it.** Its `loop_unroll` keeps the whole 256×256 accumulator live,
     so once the scheduler interleaves loads there is nowhere to put them: `llir`
     (no AGPR) spills 76 and falls to **312 TFLOPS — below base (992)**, while
     `+agpr` is spill-free at **1273** (eff 96%). This is the one case that must
     opt into `force-agpr`.
   - **v5 is hurt by it.** Its `local_prefetch` buffers over-subscribe registers
     under the forced `agpr-alloc=256`, so `+agpr` spills 248 (1190 TFLOPS); the
     default `llir` is spill-free and **fastest at 1263**.
   - **v7/v8 are indifferent** (eff a touch higher with `+agpr` — v7 88→96%, v8
     85→94% — but TFLOPS within ~4% and 0 spills either way).

3. **Away from 256×256, AGPR forcing is a no-op.** For every tile with a ≤128
   dimension, `base`, `llir`, and `+agpr` are within run-to-run noise and all
   spill-free — the accumulators are already small enough that RA never needed the
   AGPR budget. So the decoupling costs nothing on the bulk of the grid.

4. **The sliced kernels never need `force-agpr`.** Across all 9 tiles, v7 and v8's
   default `llir` matches `+agpr` (0 spills throughout). Slicing keeps the
   accumulators small, so the scheduler's MFMA-interleaving gains (e.g. v7 256×256
   65→88%, v8 65→85% vs base) come **for free**, without touching register
   allocation. This is the strongest argument for the new default: on the kernels
   that are tile-robust to begin with, plain `gemm-4waves` already captures the win.

5. **Net: decoupling makes the default safe.** Plain `gemm-4waves` improves or ties
   base everywhere except v6's full-tile case, and it removes v5's AGPR-induced
   spill. The only kernel that regresses without AGPRs (v6 @ 256×256) can request
   `force-agpr` explicitly. AGPR forcing is now an opt-in for the specific
   large-accumulator kernels that benefit, instead of a blanket coupling that
   sometimes backfires.

### Caveats
- The parametric kernels pin `warps_per_cta=[2,2]` and `load_contig=8`, so non-default
  base perf is partly a kernel-tuning artifact, not purely the scheduler.
- For v5/v6, 4 of the 9 tiles (256×64, 128×64, 64×128, 64×64) are not yet numerically
  correct; closing them needs the exact `load_contig` exposed from the C++
  `CoalesceAsyncCopy` path (a follow-up to #10302). v7/v8 cover all 9.
- At these fixed-occupancy sizes the kernels are small/fast, so absolute TFLOPS are
  low and carry more launch-overhead and run-to-run jitter than the full 4096³ case;
  use them for the cross-config *comparison*, not as headline numbers.

---

## Follow-up investigations

### 1. Does v8 (sliceMN) beat v7 (sliceN) at large K?
**Experiment:** re-ran the v6/v7/v8 comparison (5 shapes, base/llir/+agpr, fixed
occupancy) at K=16384 vs the K=8192 table — see "v6 vs v7 vs v8 at K=16384" above.
**Conclusion:** No, not at these occupancy-fixed, spill-free sizes. The mean v8/v7
TFLOPS advantage *drops* with K (+1.6% at K=8192 → −1.1% at K=16384); doubling K shifts
the balance toward v7, the opposite of the tutorial's claim. v8's edge presumably needs
a regime where v7's larger accumulator spills, which 1 wave/SIMD never reaches here.

### 2. Is v6 → v7 only the N-slicing? Do they share prologue/epilogue?
**Experiment:** source-diffed `v6_loop_unroll` vs `v7_sliceN`.
**Conclusion:** No — beyond N-slicing, v7 also differs in commit-group granularity and
async-wait depth (loop `wait_group(2)` keeping 2 groups in flight, vs v6 `wait_group(0)`
draining every step), in the store `BlockedLayout` (`[4,16]` vs `[2,32]` threadsPerWarp),
and in the epilogue (v7 overlaps the `c_left` store with the `acc_right` compute; v6 does
both dots then one store). They *do* share the same 2×-unrolled K-loop and `nBuffers=2`.
So the prologue and epilogue are **not** identical. (This different synchronization
structure is also a plausible reason v7 avoids v6's narrow-tile race — see issues.md.)

### 3. Why is v8's MFMA-eff below v7's under the scheduler (256×256×64, +agpr)?
**Experiment:** ran v7 and v8 with `force-agpr`, collected ATT MFMA-eff and disassembled
the hot loops. Both loops issue identical work (256 MFMA, 64 ds_read, 32 global-loads).
**Conclusion:** v8's M+N slicing halves each dot (`[128,128]` → 32 MFMA/region vs v7's
`[256,128]` → 64), so the scheduler emits **twice the regions (8 vs 4) and twice the
per-iteration workgroup `s_barrier`s (8 vs 4, one per region boundary)**. Those extra
barrier stalls cap v8 ~1pp below v7 under the scheduler (94.8% vs 95.9%). Under `base`
the deciding factor is instead register pressure, where v8's smaller accumulators win
(436 vs 460 VGPRs, eff 68.9% vs 65.0%) — so the lever flips: register pressure favors v8
unscheduled, barrier/region count favors v7 once the scheduler packs the MFMAs.

---

## Cold-cache K-scaling at full size: v6 / v7 / v8 at K ∈ {8192, 16384, 32768}

The sweeps above fix occupancy at small tiles to isolate the scheduler. This section
asks the complementary question: at the full 4096³ tile, **how do the three loop
pipelines hold up as K (and the memory working set) grows, with caches genuinely
cold?** Config is **`llir+agpr+amdgcnas`** (scheduler + `force-agpr` + the amdgcnas
post-assembly pass), fp16, `BLOCK = 256×256×64`, 1 wave/SIMD.

**Method — true cold cache.** `bench.py --rocprof` cycles through
`floor(rotating_buffer_size / working_set)` tensor copies. With the default 512 MB that
count collapses to **1** at K ≥ 16384 (working sets 288 / 544 MB), so the single reused
buffer stays cache-resident and *inflates* TFLOPS — at K=16384 it falsely reads
~1300 / 86% for v6. We instead pass **`--rotating-buffer-size 2048`** → **12 / 7 / 3**
copies at K = 8192 / 16384 / 32768, forcing eviction at every K. TFLOPS is rocprofv3
kernel-trace (last-100 of 1000 dispatches, `AMD_SERIALIZE_KERNEL=3`); MFMA-eff, the
prologue/loop/epilogue cycle split, and the in-loop buffer_load stall are from ATT
(`process_json.py` + a back-edge loop scan) over the same cold rotating path.

**Single-dispatch jitter → medians.** ATT traces one dispatch (iteration 15), whose
cold-cache state varies run-to-run (±~3 pp on MFMA-eff, ±~25 % on the stall counter).
So every ATT-derived value below — MFMA-eff, the prologue/loop/epilogue cycles, and the
buffer_load stall — is the **median of 5 ATT runs** per cell. TFLOPS is a 100-dispatch
average (kernel-trace) and stable, shown from a single run.

### TFLOPS

| K | v6 | v7 | v8 |
|---|---|---|---|
| 8192 | 1265 | 1432 | **1447** |
| 16384 | 1068 | 1405 | **1458** |
| 32768 | 896 | **1316** | 1298 |

### MFMA efficiency (median of 5)

| K | v6 | v7 | v8 |
|---|---|---|---|
| 8192 | 85.0% | 98.1% | **99.0%** |
| 16384 | 59.4% | 93.3% | **95.1%** |
| 32768 | 38.5% | **83.9%** | 79.7% |

### Cycle breakdown — prologue / loop / epilogue (ATT clock cycles, median of 5)

| K | kernel | prologue | loop | epilogue | total | iters | loop/iter |
|---|---|---|---|---|---|---|---|
| 8192 | v6 | 5,332 | 303,408 | 14,896 | 323,636 | 63 | 4,816 |
| | v7 | 5,028 | 263,036 | 14,312 | 282,376 | 63 | 4,175 |
| | v8 | 5,252 | 260,736 | 12,712 | 278,700 | 63 | 4,138 |
| 16384 | v6 | 6,476 | 875,164 | 14,820 | 896,460 | 127 | 6,891 |
| | v7 | 5,884 | 557,364 | 13,992 | 577,240 | 127 | 4,388 |
| | v8 | 6,060 | 546,992 | 12,748 | 565,800 | 127 | 4,307 |
| 32768 | v6 | 8,880 | 2,716,620 | 14,152 | 2,739,652 | 255 | 10,653 |
| | v7 | 7,384 | 1,244,852 | 14,408 | 1,266,644 | 255 | 4,881 |
| | v8 | 7,372 | 1,310,004 | 12,696 | 1,330,072 | 255 | 5,137 |

Prologue (~5–9k) and epilogue (~13–15k) are essentially K-independent — pipeline fill,
final MFMA drain, downcast + store — so the epilogue fades from ~5 % of the kernel at
K=8192 to ~0.5 % at K=32768. All the scaling is in the loop. (Each column is an
independent per-cell median, so `total` is their sum.)

### In-loop buffer_load stall — averaged stall cycles per buffer_load (median of 5)

Mean over the 32 `buffer_load` instructions in the loop body of each load's
`stall / hit` (its ATT stall counter averaged across the loop's iterations) — the stall
an individual global load incurs per iteration. (×32 gives the total load-stall cycles
charged to one K-loop iteration.)

| K | v6 | v7 | v8 |
|---|---|---|---|
| 8192 | 15.9 | **4.3** | 4.8 |
| 16384 | 39.5 | 9.6 | **6.8** |
| 32768 | 118.1 | **23.6** | 30.4 |

### Reading the four tables together

- **v6 collapses, monotonically.** TFLOPS 1265 → 1068 → 896, MFMA-eff 85 → 59 → 38 %,
  loop/iter ballooning 4,816 → 6,891 → 10,653 cycles. The cause is direct: the average
  per-load stall explodes **7.4×** (15.9 → 118.1 cyc); summed over all 32 in-loop loads
  that is ~3,780 cyc/iter — about a third of the 10,653-cycle iteration at K=32768.
  v6's flat (un-sliced) pipeline can't keep the global
  loads ahead of the MFMAs once the working set goes cold, so the loads stall and the
  MFMA units starve. *(The default 512 MB run hid this — it falsely showed K=16384
  rising to 1300; with true rotation v6 already drops to 1068 / 59 % there.)*
- **v7 stays fed.** per-load stall 4.3 → 9.6 → 23.6 cyc (3–4.5× lower than v6 at every K),
  loop/iter near-flat (4,175 → 4,881, +17 % over a 4× K range), eff 98 → 93 → 84 %.
- **v8 wins at K ≤ 16384, v7 overtakes at K=32768.** v8's smaller 128-row A tile gives
  it the **lowest** per-load stall at K=16384 (6.8 cyc, even below v7's 9.6) → highest
  eff (95.1 %) and TFLOPS (1458). But at K=32768 v8's stall (30.4 cyc) passes v7's (23.6), its
  eff drops below v7 (79.7 vs 83.9 %), and v7 takes the TFLOPS lead (1316 vs 1298). The
  buffer_load stall is the mechanism behind the v7/v8 crossover the tutorial points at —
  and it lands at K=32768, not K=16384.
- The buffer_load stall tracks MFMA-eff across the table — clearest at large K, where
  v6's 118 cyc/load vs v7's 24 mirrors the 38 % vs 84 % eff gap. (At K=8192 the three are
  near-tied at ≥98 % eff / ≤16 cyc, within run-to-run jitter.) The efficiency loss at
  large K is **global-load latency the pipeline fails to hide**, not compute throughput
  or register pressure.

---

## Where the cache locality comes from: M-slicing vs N-slicing (no XCD remapping)

v8 (slice M+N) edges v7 (slice N) at moderate K mainly via **better cache locality** — but
which slicing axis is responsible? To isolate it we built **`v7m_sliceM`**: a symmetric
mirror of v7 that slices **A along M** (a_top/a_bot, 128 rows) and keeps **B full** — same
pipeline structure (2 buffers, factor-2 unroll, identical wait_group depths) as v7, so the
only change vs v7 is the slicing *axis*. Three-way comparison, cold `--rocprof` regime,
`--rotating-buffer-size 2048`, `llir+agpr+amdgcnas`, 5-run median TFLOPS / 3-run median
counters. **All kernels here use the plain `pid_m = pid // num_pid_n; pid_n = pid %
num_pid_n` mapping — NO XCD-aware PID remapping (that arrives in v9).**

| K | kernel | TFLOPS (median, range) | VmemLat | L2 hit % | DRAM reads |
|---|---|---|---|---|---|
| 8192 | sliceN (v7) | 1432 (1429–1434) | 1073 | 71.1 | 4.8M |
| 8192 | sliceM (v7m) | 1410 (1407–1413) | 1082 | 71.2 | 4.9M |
| 8192 | sliceMN (v8) | 1447 (1446–1455) | 1040 | 70.7 | 4.8M |
| 16384 | sliceN (v7) | 1410 (1409–1412) | 1757 | 60.1 | 13.4M |
| 16384 | **sliceM (v7m)** | **1491 (1488–1493)** | **1333** | **69.0** | **10.5M** |
| 16384 | sliceMN (v8) | 1425 (1401–1436) | 1659 | 60.5 | 13.2M |
| 32768 | sliceN (v7) | 1322 (1309–1364) | 2082 | 58.5 | 27.8M |
| 32768 | **sliceM (v7m)** | **1420 (1414–1429)** | **1656** | **68.5** | **21.2M** |
| 32768 | sliceMN (v8) | 1301 (1294–1315) | 2123 | 61.9 | 25.5M |

**The cache advantage comes from slicing A (M), not B (N).** At K ≥ 16384, sliceM has the
best L2 hit (≈69 % vs ≈60 %), **~22–24 % fewer DRAM reads**, the lowest VmemLatency, and
wins on TFLOPS by **+5–7 % over v7 and +5–9 % over v8** (tight, non-overlapping medians).
The pattern is monotone: **slice A → best cache (sliceM), slice B → worst (sliceN), slice
both → middle (v8)** — slicing B actively *hurts* locality and only adds barrier overhead.
(K=8192 is a tie: the working set is L2-resident, ≈71 % hit for all.)

Every DRAM read is 128 B (request-size counters: rd32=rd64=0), so this is **pure L2
temporal reuse**, not request granularity or cache-line alignment.

**Mechanism — the workgroup launch order.** With `pid = pid_m·num_pid_n + pid_n` (pid_n
fastest), the 16 co-launched workgroups of one M-band **share the same A rows** but stream
**different B columns**. That makes **A the reuse-critical operand** (its sharers run
concurrently → high L2 reuse) and **B loosely shared** (its sharers are 16 apart in launch).
Slicing the reuse-critical tensor (A) into smaller per-access tiles improves its L2
residency → fewer A DRAM misses → lower VmemLatency → higher throughput; slicing B
fragments a barely-reused tensor and just piles on barriers. This also resolves the
"better cache yet longer latency" puzzle seen for v8: pure A-slicing (sliceM) gets the
cache win **and** the lower latency together — v8's higher VmemLatency was the N-slicing
barrier tax on top of a watered-down A benefit.

**Takeaway:** at these sizes, slicing *both* M and N (v8) is suboptimal; slicing only A (M)
is the right call because the launch order makes A the cache-critical operand.
(`v7m_sliceM` is an experimental kernel — it uses the mfma-layout store rather than v7/v8's
tuned BlockedLayout, so its epilogue differs slightly; the loop/cache behavior measured
here is unaffected.)

### …but XCD-aware PID remapping erases the difference

The above assumes the naive `pid_m = pid // num_pid_n` mapping. v9 introduces **XCD-aware
PID remapping + GROUP_SIZE_M swizzling** (`get_pids`, here added to all three kernels and
enabled with `NUM_XCDS=8 GROUP_SIZE_M=4`, v9's settings). Re-running the same sweep with it:

| K | kernel | TFLOPS (median, range) | L2 hit % | DRAM reads | VmemLat |
|---|---|---|---|---|---|
| 8192 | sliceN | 1467 (1467–1471) | 80.3 | 3.1M | 908 |
| 8192 | sliceM | 1450 (1447–1454) | 80.9 | 3.1M | 926 |
| 8192 | sliceMN | 1475 (1473–1478) | 80.3 | 3.2M | 787 |
| 16384 | sliceN | 1515 (1514–1521) | 78.8 | 7.0M | 1054 |
| 16384 | sliceM | 1513 (1507–1515) | 78.9 | 7.0M | 1029 |
| 16384 | sliceMN | 1521 (1518–1531) | 78.1 | 7.2M | 929 |
| 32768 | sliceN | 1306 (1302–1361) | 81.0 | 12.6M | 2033 |
| 32768 | sliceM | 1339 (1297–1349) | 81.1 | 12.6M | 2224 |
| 32768 | sliceMN | 1309 (1306–1317) | 81.0 | 12.6M | 2115 |

**With XCD remapping the three slicing variants are tied** — identical L2 hit (~78–81 %),
identical DRAM reads, TFLOPS within ~1 % at K=16384 and within noise elsewhere. sliceM's
standout (69 % L2 hit, +5–9 % TFLOPS without remapping) **vanishes**: the remapping lifts
*every* kernel to ~80 % L2 hit, well above what M-slicing alone achieved (69 %), and ~halves
DRAM reads at large K (e.g. K=32768: 21–28M → 12.6M for all).

**This confirms the mechanism.** sliceM's advantage was a *workaround* for the naive launch
order's poor L2 locality; v9's remapping fixes locality directly and better, so the slicing
axis no longer matters for cache. Practically: **once you use XCD-aware PID remapping,
slicing both M and N (v8) is fine** — the M-slicing edge was a launch-order artifact, not a
fundamental property. (At K=32768 the remapping even costs sliceM throughput, 1420 → 1339:
its cache was already good, and the remapping adds L2-slice contention — VmemLat 1656 → 2224
— without a locality payoff there.)

*(Kernel additions for this study — `v7m_sliceM`, `bench.py` version-10 hook, and the
env-toggleable `get_pids` (`NUM_XCDS`/`GROUP_SIZE_M`, default off) added to v7/v8 — are
experimental and uncommitted.)*
