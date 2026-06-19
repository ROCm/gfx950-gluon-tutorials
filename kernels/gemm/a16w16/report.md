# a16w16 LLIR scheduler — tile-size robustness sweep

How robust is the gfx950 LLIR scheduler (`schedule_hint="gemm-4waves"`) across GEMM
tile sizes? We sweep `BLOCK_M ∈ {256,128,64}`, `BLOCK_N ∈ {256,128,64}`,
`BLOCK_K ∈ {64,32}` on the `v5_local_prefetch` and `v6_loop_unroll` kernels and
compare **base** vs **gemm-4waves (llir)**.

## Method

- The tutorial kernels hardcode their gluon layouts for 256×256×64. To vary the
  tile we made them **tile-parametric**: the `PaddedSharedLayout` is computed by
  [`ttgl.amd.cdna4.compute_efficient_padded_shared_layout`](https://github.com/triton-lang/triton/pull/10302)
  (triton-lang/triton#10302); the matching coalesced global-load `DistributedLinearLayout`
  is derived from it (reimplementing `CoalesceAsyncCopy`'s padded-encoding logic);
  the MFMA layout keeps `warps_per_cta=[2,2]`.
- Tile coverage depends on the kernel: **v5/v6 are correct on 7 of 18 tiles**
  (the other 11 need the exact `load_contig`, currently pinned to 8, which lives
  in the C++ `CoalesceAsyncCopy` pass); **v7, which slices B along N
  (`BLOCK_N // 2`), is correct on all 18** — the smaller B tile keeps the derived
  layouts valid across the whole grid. Only the correct tiles are reported.
- Metrics: **TFLOPS, MFMA-eff, spills, VGPRs**, collected with `--rocprof`
  (rocprofv3 kernel-trace, 1000 dispatches, last-100 avg; MFMA-eff from ATT,
  hot-loop per-iteration). `sched?` = whether the LLIR scheduler actually applied
  (region markers in the generated asm).
- `K = 8192`, fp16.

---

## Experiment 1 — occupancy fixed (M = 16·BLOCK_M, N = 16·BLOCK_N)

Grid = 16×16 = **256 workgroups ≈ 1 per CU** on MI350 → **1 wave/SIMD**, so the
register pressure from the AGPR coupling cannot change occupancy. This isolates
the scheduler's effect from occupancy.

### v6 (loop_unroll)

| tile | M×N | cfg | TFLOPS | MFMA-eff | spills | VGPRs | sched? |
|---|---|---|---|---|---|---|---|
| **256×256×64** | 4096×4096 | base | 992 | 53.20% | 8 | 512 | – |
| | | llir | **1273** | **95.70%** | 0 | 500 | YES |
| 256×256×32 | 4096×4096 | base | 901 | 55.85% | 0 | 476 | – |
| | | llir | 886 | 54.61% | 0 | 472 | YES |
| 256×128×64 | 4096×2048 | base | 892 | 51.79% | 0 | 244 | – |
| | | llir | 889 | 65.26% | 0 | 340 | YES |
| 256×128×32 | 4096×2048 | base | 603 | 36.98% | 0 | 236 | – |
| | | llir | 612 | 38.45% | 0 | 340 | YES |
| 128×256×64 | 2048×4096 | base | 903 | 51.63% | 0 | 288 | – |
| | | llir | 902 | 67.36% | 0 | 324 | YES |
| 128×128×64 | 2048×2048 | base | 619 | 42.46% | 0 | 142 | – |
| | | llir | 614 | 41.27% | 0 | 196 | YES |
| 64×256×64 | 1024×4096 | base | 549 | 36.57% | 0 | 164 | – |
| | | llir | 545 | 35.95% | 0 | 244 | YES |

### v5 (local_prefetch)

| tile | M×N | cfg | TFLOPS | MFMA-eff | spills | VGPRs | sched? |
|---|---|---|---|---|---|---|---|
| **256×256×64** | 4096×4096 | base | 1134 | 58.63% | 0 | 452 | – |
| | | llir | 1190 | 77.59% | **248** | 512 | YES |
| 256×256×32 | 4096×4096 | base | 903 | 55.79% | 0 | 400 | – |
| | | llir | 881 | 55.27% | 10 | 512 | YES |
| 256×128×64 | 4096×2048 | base | 882 | 51.21% | 0 | 244 | – |
| | | llir | 872 | 61.68% | 0 | 356 | YES |
| 256×128×32 | 4096×2048 | base | 610 | 40.56% | 0 | 188 | – |
| | | llir | 606 | 41.70% | 0 | 308 | YES |
| 128×256×64 | 2048×4096 | base | 898 | 51.06% | 0 | 244 | – |
| | | llir | 884 | 64.27% | 0 | 356 | YES |
| 128×128×64 | 2048×2048 | base | 611 | 41.75% | 0 | 141 | – |
| | | llir | 604 | 39.79% | 0 | 196 | YES |
| 64×256×64 | 1024×4096 | base | 545 | 36.64% | 0 | 162 | – |
| | | llir | 529 | 36.38% | 0 | 244 | YES |

### v7 (sliceN) — full 18-tile grid

N-sliced (B = `BLOCK_N // 2`), so correct on every tile. Spills are 0 throughout.

| tile | M×N | cfg | TFLOPS | MFMA-eff | spills | VGPRs | sched? |
|---|---|---|---|---|---|---|---|
| **256×256×64** | 4096×4096 | base | 1226 | 64.99% | 0 | 496 | – |
| | | llir | **1427** | 96.14% | 0 | 460 | YES |
| 256×256×32 | 4096×4096 | base | 1095 | 66.86% | 0 | 420 | – |
| | | llir | 1101 | 87.90% | 0 | 420 | YES |
| 256×128×64 | 4096×2048 | base | 1077 | 56.92% | 0 | 271 | – |
| | | llir | 1183 | 86.51% | 0 | 320 | YES |
| 256×128×32 | 4096×2048 | base | 692 | 44.76% | 0 | 240 | – |
| | | llir | 796 | 51.40% | 0 | 276 | YES |
| 256×64×64 | 4096×1024 | base | 713 | 44.39% | 0 | 236 | – |
| | | llir | 733 | 51.77% | 0 | 248 | YES |
| 256×64×32 | 4096×1024 | base | 460 | 27.95% | 0 | 186 | – |
| | | llir | 464 | 29.12% | 0 | 216 | YES |
| 128×256×64 | 2048×4096 | base | 1011 | 56.48% | 0 | 246 | – |
| | | llir | **1225** | 84.44% | 0 | 268 | YES |
| 128×256×32 | 2048×4096 | base | 764 | 48.38% | 0 | 248 | – |
| | | llir | 868 | 61.16% | 0 | 228 | YES |
| 128×128×64 | 2048×2048 | base | 900 | 54.39% | 0 | 148 | – |
| | | llir | 920 | 64.30% | 0 | 176 | YES |
| 128×128×32 | 2048×2048 | base | 604 | 39.82% | 0 | 136 | – |
| | | llir | 618 | 42.04% | 0 | 148 | YES |
| 128×64×64 | 2048×1024 | base | 526 | 36.52% | 0 | 124 | – |
| | | llir | 537 | 37.89% | 0 | 136 | YES |
| 128×64×32 | 2048×1024 | base | 325 | n/a | 0 | 128 | – |
| | | llir | 342 | n/a | 0 | 144 | YES |
| 64×256×64 | 1024×4096 | base | 742 | 49.00% | 0 | 130 | – |
| | | llir | 731 | 57.50% | 0 | 180 | YES |
| 64×256×32 | 1024×4096 | base | 564 | 34.85% | 0 | 100 | – |
| | | llir | 561 | 35.90% | 0 | 128 | YES |
| 64×128×64 | 1024×2048 | base | 563 | 41.51% | 0 | 90 | – |
| | | llir | 561 | 42.20% | 0 | 108 | YES |
| 64×128×32 | 1024×2048 | base | 362 | n/a | 0 | 118 | – |
| | | llir | 367 | n/a | 0 | 124 | YES |
| 64×64×64 | 1024×1024 | base | 324 | n/a | 0 | 94 | – |
| | | llir | 318 | n/a | 0 | 116 | YES |
| 64×64×32 | 1024×1024 | base | 201 | n/a | 0 | 70 | – |
| | | llir | 202 | n/a | 0 | 80 | YES |

(`n/a` = ATT eff-collection failed on the tiniest kernels; TFLOPS still valid.)

### v8 (sliceMN) — full 18-tile grid

Slices **both** M and N (A = `BLOCK_M//2`, B = `BLOCK_N//2`, 4 accumulator
quadrants), so also correct on all 18 tiles and spill-free throughout.

| tile | M×N | cfg | TFLOPS | MFMA-eff | spills | VGPRs | sched? |
|---|---|---|---|---|---|---|---|
| **256×256×64** | 4096×4096 | base | 1251 | 68.92% | 0 | 496 | – |
| | | llir | **1423** | 93.65% | 0 | 436 | YES |
| 256×256×32 | 4096×4096 | base | 974 | 51.11% | 0 | 496 | – |
| | | llir | 1062 | 91.13% | 0 | 416 | YES |
| 256×128×64 | 4096×2048 | base | 1117 | 63.40% | 0 | 264 | – |
| | | llir | 1193 | 83.39% | 0 | 272 | YES |
| 256×128×32 | 4096×2048 | base | 718 | 54.85% | 0 | 276 | – |
| | | llir | 719 | 57.88% | 0 | 232 | YES |
| 256×64×64 | 4096×1024 | base | 738 | 43.81% | 0 | 254 | – |
| | | llir | 736 | 48.01% | 0 | 200 | YES |
| 256×64×32 | 4096×1024 | base | 431 | 31.84% | 0 | 154 | – |
| | | llir | 426 | 31.48% | 0 | 164 | YES |
| 128×256×64 | 2048×4096 | base | 1116 | 66.04% | 0 | 242 | – |
| | | llir | 1210 | 83.36% | 0 | 260 | YES |
| 128×256×32 | 2048×4096 | base | 810 | 51.45% | 0 | 242 | – |
| | | llir | 833 | 60.79% | 0 | 200 | YES |
| 128×128×64 | 2048×2048 | base | 954 | 55.65% | 0 | 142 | – |
| | | llir | 950 | 68.44% | 0 | 152 | YES |
| 128×128×32 | 2048×2048 | base | 640 | 44.34% | 0 | 130 | – |
| | | llir | 652 | 48.37% | 0 | 132 | YES |
| 128×64×64 | 2048×1024 | base | 620 | 40.09% | 0 | 100 | – |
| | | llir | 626 | 42.85% | 0 | 108 | YES |
| 128×64×32 | 2048×1024 | base | 359 | 26.02% | 0 | 82 | – |
| | | llir | 364 | 26.38% | 0 | 92 | YES |
| 64×256×64 | 1024×4096 | base | 732 | 49.40% | 0 | 234 | – |
| | | llir | 729 | 56.52% | 0 | 180 | YES |
| 64×256×32 | 1024×4096 | base | 535 | 35.33% | 0 | 126 | – |
| | | llir | 515 | 35.51% | 0 | 124 | YES |
| 64×128×64 | 1024×2048 | base | 607 | 42.70% | 0 | 92 | – |
| | | llir | 606 | 47.17% | 0 | 100 | YES |
| 64×128×32 | 1024×2048 | base | 389 | 29.10% | 0 | 66 | – |
| | | llir | 379 | 26.50% | 0 | 72 | YES |
| 64×64×64 | 1024×1024 | base | 367 | 27.49% | 0 | 60 | – |
| | | llir | 370 | 28.65% | 0 | 60 | YES |
| 64×64×32 | 1024×1024 | base | 212 | n/a | 0 | 70 | – |
| | | llir | 210 | n/a | 0 | 84 | YES |

---

## Experiment 2 — variable occupancy (M = N = 4096 for every tile)

Same problem size for all tiles, so smaller tiles launch many more workgroups and
have higher occupancy. Shown for contrast — the small-tile regressions here are an
**occupancy artifact** (see Conclusions).

### v6 (loop_unroll)

| tile | cfg | TFLOPS | MFMA-eff | spills | VGPRs |
|---|---|---|---|---|---|
| **256×256×64** | base | 988 | 52.98% | 8 | 512 |
| | llir | **1260** | **95.54%** | 0 | 500 |
| 256×256×32 | base | 897 | 55.16% | 0 | 476 |
| | llir | 885 | 54.18% | 0 | 472 |
| 256×128×64 | base | 963 | 51.97% | 0 | 244 |
| | llir | 952 | 70.44% | 0 | 340 |
| 256×128×32 | base | 1024 | 29.77% | 0 | 236 |
| | llir | **742** | 44.79% | 0 | 340 |
| 128×256×64 | base | 954 | 51.81% | 0 | 288 |
| | llir | 992 | 71.74% | 0 | 324 |
| 128×128×64 | base | 1085 | 41.43% | 0 | 142 |
| | llir | 1107 | 42.59% | 0 | 196 |
| 64×256×64 | base | 656 | 37.29% | 0 | 164 |
| | llir | 656 | 37.07% | 0 | 244 |

### v5 (local_prefetch)

| tile | cfg | TFLOPS | MFMA-eff | spills | VGPRs |
|---|---|---|---|---|---|
| **256×256×64** | base | 1134 | 58.47% | 0 | 452 |
| | llir | 1193 | 77.75% | **248** | 512 |
| 256×256×32 | base | 902 | 54.60% | 0 | 400 |
| | llir | 881 | 55.18% | 10 | 512 |
| 256×128×64 | base | 955 | 51.30% | 0 | 244 |
| | llir | 936 | 69.89% | 0 | 356 |
| 256×128×32 | base | 948 | 29.03% | 0 | 188 |
| | llir | **723** | 44.99% | 0 | 308 |
| 128×256×64 | base | 943 | 51.22% | 0 | 244 |
| | llir | 962 | 64.51% | 0 | 356 |
| 128×128×64 | base | 1066 | 41.66% | 0 | 141 |
| | llir | 1052 | 40.46% | 0 | 196 |
| 64×256×64 | base | 629 | 37.42% | 0 | 162 |
| | llir | 631 | 36.49% | 0 | 244 |

---

## Conclusions

1. **The scheduler applies successfully at every tile** (both kernels, `sched? = YES`):
   region detection and MFMA interleaving engage on all 7 correct tiles. It is
   correctness-robust everywhere it compiles.

2. **The benefit is concentrated at the designed 256×256×64 4-wave shape.** v6 gains
   **+28%** there (eff 53→96%, and the 8 base spills disappear). Elsewhere it is
   neutral; even where MFMA-eff rises (e.g. 256×128×64: 52→65%, 128×256×64: 52→67%),
   TFLOPS does not follow — those tiles hit a lower eff ceiling and are gated by
   shape/memory, not MFMA interleaving. (For 128×256×64 the hot-loop region demands
   more MFMA cover than it has — `needed=71 > total=64` MFMAs — so the global loads
   stay under-covered: memory-bound.)

3. **Fixing occupancy removes the apparent harm.** In Experiment 2 small tiles show
   large *negative* swings (v6 256×128×32: **−27.5%**); with occupancy pinned to
   1 wave/SIMD (Experiment 1) the same point is **+1.5%** and every non-default tile
   is neutral (±~2%). The cliffs were an **AGPR-coupling × occupancy artifact**:
   `gemm-4waves` raises VGPRs at every tile (e.g. 236→340), which only costs
   throughput when the extra registers reduce waves/CU.

4. **v5 vs v6.** v6 gets the clean +28%; v5 only +5% at the default tile because the
   AGPR coupling makes its `local_prefetch` pipeline **spill (248 VGPRs)**. The
   coupling's register cost is what separates the two kernels.

5. **v7 (sliceN) is where the scheduler is robustly *beneficial* across tiles.**
   Unlike v5/v6 — where MFMA-eff rose at non-default tiles but TFLOPS stayed flat
   (memory-bound) — v7's eff **and** TFLOPS rise together: e.g. 128×256×64 eff
   56→84% / **+21%**, 256×128×64 57→87% / +10%, 256×128×32 +15%, plus the default
   256×256×64 +16%. Wins span the whole M·N ≥ 128×128, N ≥ 128 region; only the
   64×N tiles see diminishing/slightly-negative returns (kernel too small to fill
   the wave). N-slicing keeps the accumulators half-size (`acc_left`/`acc_right`),
   so there are **0 spills** everywhere and the hot loop has enough MFMA headroom
   that better interleaving actually buys throughput. The slicing is what enables
   both the tile-parametric layouts (all 18 correct) and the scheduler's
   effectiveness across tiles.

6. **v8 (sliceMN) confirms the pattern and raises the base.** Slicing both M and N
   is also correct on all 18 tiles and spill-free. Its **base** TFLOPS/eff are
   generally the highest of the four kernels (the finer slicing already pipelines
   well), and the scheduler still adds on top at the larger tiles — 256×256×64
   +14% (eff 69→94%), 256×256×32 **+9% with eff 51→91%**, 256×128×64 +7%,
   128×256×64 +8%. Because v8's base is already strong, the scheduler's *relative*
   gain is a bit smaller than v7's, but absolute llir TFLOPS at the top tiles
   match v7 (~1420). Small tiles (64×N) again see flat/slightly-negative returns.
   Net: v7 and v8 (the sliced kernels) are both robustly schedulable and
   beneficial across the grid; v8 has the better base, v7 the larger relative
   uplift.

### Caveats
- The parametric kernels pin `warps_per_cta=[2,2]` and `load_contig=8`, so non-default
  base perf is partly a kernel-tuning artifact, not purely the scheduler.
- For v5/v6, 11 of 18 tiles are not yet numerically correct; the full grid needs the
  exact `load_contig` exposed from the C++ `CoalesceAsyncCopy` path (a follow-up to
  #10302). v7's N-slicing sidesteps this and covers all 18.
