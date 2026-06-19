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
- **7 of 18 tiles** are numerically correct with this approach; the other 11 need
  the exact `load_contig` (currently pinned to 8), which lives in the C++
  `CoalesceAsyncCopy` pass. Only the correct tiles are reported.
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

### Caveats
- The parametric kernels pin `warps_per_cta=[2,2]` and `load_contig=8`, so non-default
  base perf is partly a kernel-tuning artifact, not purely the scheduler.
- 11 of 18 tiles are not yet numerically correct; the full grid needs the exact
  `load_contig` exposed from the C++ `CoalesceAsyncCopy` path (a follow-up to #10302).
