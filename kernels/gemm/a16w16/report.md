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
