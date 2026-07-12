# GEMM Kernels in Gluon

This directory contains high-performance GEMM kernels written in Gluon for AMD MI350/355 (gfx950).

The objective of this repository is broader than implementing a fast GEMM.

It explores a fundamental systems question:

> **Where should scheduling intelligence live?**

Once memory hierarchy and tiling are optimized, peak GEMM performance depends on overlapping MFMA execution with memory operations. This repository presents two different scheduling models that solve this problem.

## Directory Structure

```
gemm/
├── utils/                                # shared Gluon device helpers (get_pids), used by both routes
├── intra_wave/                            # 4-wave — compiler interleaves MFMA + loads (LLIR sched + force-agpr + amdgcnas)
│   ├── a16w16/                            # FP16/BF16 — the v0→v9 optimization journey (start here)
│   │   ├── v0_naive/                      #   baseline: explicit layouts, correctness-first
│   │   ├── v1_buffer_load/                #   buffer_load for hardware OOB (branch elimination)
│   │   ├── v2_async_copy/                 #   direct-to-LDS async copy
│   │   ├── v3_lds/                        #   LDS layout design: swizzle vs padding
│   │   ├── v4_global_prefetch/            #   2-stage pipeline (double buffering)
│   │   ├── v5_local_prefetch/             #   3-stage pipeline + LLIR scheduler
│   │   ├── v6_loop_unroll/                #   loop unrolling
│   │   ├── v7_sliceN/                     #   N-slicing (register pressure)
│   │   ├── v8_sliceMN/                    #   M+N slicing
│   │   └── v9_beyond_hotloop/             #   XCD-aware PID remapping (L2 locality)
│   ├── a8w8/                              # BF8 (e5m2) — single kernel (no version subdirs)
│   └── a4w4/                              # MXFP4 (e2m1) — adds the per-group scale pipeline
│       ├── v0_sliceN/                     #   N-slicing + LDS round-trip scales
│       └── v1_sliceMN/                    #   M+N slicing + direct-to-LDS scales
└── inter_wave/                            # 8-wave — two waves ping-pong (warp_pipeline_stage, no AGPRs)
    ├── a16w16/                            # FP16/BF16 — 8-wave warp-pipeline (sliceMN, BLOCK_K=64) — single kernel
    ├── a8w8/                              # BF8 — 8-wave warp-pipeline (sliceMN, BLOCK_K=128) — single kernel
    └── a4w4/                              # MXFP4 — 8-wave warp-pipeline
        ├── v0_sliceMN/          #   byte-shuffle B scale (baseline)
        ├── v1_combineBsc/       #   combined transpose-read B scale (recommended)
        └── v2_mfma32x32x64/     #   32×32×64 MFMA + conflict-free LDS layout
```

## Two scheduling models

### Intra-wave scheduling

The **intra-wave** kernels — [`intra_wave/`](intra_wave/README.md) (`a16w16`, `a8w8`, `a4w4`) — launch **one wave per SIMD**.

Memory instructions and MFMA instructions are interleaved within each wave. This relies on compiler assistance through `llirSched`, `force-agpr`, and `amdgcnas` to preserve the scheduling intent expressed by the Gluon kernel.

These kernels demonstrate how compiler infrastructure can recover an efficient instruction schedule while keeping the kernel relatively straightforward. Build, run, and the full FP16 → BF8 → MXFP4 optimization journey are in [`intra_wave/README.md`](intra_wave/README.md).

### Inter-wave scheduling

The **inter-wave** kernels — [`inter_wave/`](inter_wave/README.md) (`a16w16`, `a8w8`, `a4w4`) — launch **two waves per SIMD**.

Instead of relying on compiler instruction scheduling, the kernel itself organizes execution into alternating MFMA and memory stages. Different waves execute different stages simultaneously, allowing the hardware to overlap computation and data movement naturally.

These kernels require little compiler assistance and run using upstream Triton and LLVM. See [`inter_wave/README.md`](inter_wave/README.md) for the design and per-kernel details.

For a detailed discussion of these two scheduling models, see [`docs/scheduling_models.md`](../../docs/scheduling_models.md).

## Versions

Every version isolates one idea, in the spirit of the a16w16 journey. The `a16w16` series is the
full teaching arc (v0 → v9); the other solutions reuse that design and add only what their data
type or wave count needs. See [`a16w16/README` §3](intra_wave/a16w16/README.md#3-the-optimization-journey)
for the full a16w16 narrative.

| Solution | Version | Focus | Key concept |
|----------|---------|-------|-------------|
| **`intra_wave/a16w16`** (FP16/BF16) | `v0_naive` | Baseline | Explicit layouts, correctness-first MFMA |
| | `v1_buffer_load` | Codegen | Hardware OOB checking, branch elimination |
| | `v2_async_copy` | Codegen | Direct-to-LDS, eliminates register staging |
| | `v3_lds` | Codegen | LDS layout design: swizzling vs padding |
| | `v4_global_prefetch` | Latency hiding | 2-stage pipeline, double buffering |
| | `v5_local_prefetch` | Latency hiding | 3-stage pipeline, LLIR scheduler |
| | `v6_loop_unroll` | Codegen | Unroll to eliminate copy overhead |
| | `v7_sliceN` | Register pressure | N-slicing |
| | `v8_sliceMN` | Register pressure, throughput | M+N slicing, buffer-load stall analysis |
| | `v9_beyond_hotloop` | L2 locality | XCD-aware PID remapping |
| **`intra_wave/a8w8`** (BF8) | *(single kernel)* | Data type | a16w16 design at BF8 parameters |
| **`intra_wave/a4w4`** (MXFP4) | `v0_sliceN` | Scale pipeline | N-slicing + LDS round-trip scales |
| | `v1_sliceMN` | Scale pipeline | M+N slicing + direct-to-LDS async scales |
| **`inter_wave/a16w16`** (FP16/BF16) | *(single kernel)* | Warp pipeline | M+N slicing, `BLOCK_K=64`, 2-buffer, 8-wave ping-pong |
| **`inter_wave/a8w8`** (BF8) | *(single kernel)* | Warp pipeline | M+N slicing, `BLOCK_K=128`, 2-buffer, 8-wave ping-pong |
| **`inter_wave/a4w4`** (MXFP4) | `v0_sliceMN` | Scale + pipeline | Byte-shuffle B scale (baseline) |
| | `v1_combineBsc` | Scale + pipeline | Combined transpose-read B scale *(recommended)* |
| | `v2_mfma32x32x64` | MFMA shape | 32×32×64 MFMA + conflict-free LDS layout |

## Performance Summary

Measured on a single MI355X (gfx950), Triton built from the `gfx950-tutorial-v1.0` tag, rocprof
cold-rotating (1000 dispatches, last-100 average). The **4-wave** kernels run with the LLIR
scheduler + force-agpr + amdgcnas (see [`intra_wave/README.md §2.1`](intra_wave/README.md#21-triton-build-and-the-out-of-tree-plugins)); the
**8-wave** kernels run `warp_pipeline_stage` with no AGPRs (no env vars — see [`inter_wave/README.md`](inter_wave/README.md)).

![GEMM peak throughput: 4-wave vs 8-wave, per precision](images/perf_summary.png)

Bars are peak TFLOPS at each precision's headline shape (FP16/BF16 K=8192, BF8 K=16384, MXFP4 K=32768); the **red** label inside each bar is the per-SIMD loop MFMA efficiency. The 4-wave bars are `intra_wave` (a16w16 v9, a8w8, a4w4 v1); the 8-wave bars are `inter_wave` (a16w16, a8w8, a4w4 v1). The MXFP4 8-wave bar is **v1** (combined B-scale, 4938 TFLOPS / 80.0% MFMA); the alternate **v2** (32×32×64 MFMA) trades throughput for occupancy at **4799 / 98.0%**.

> [!NOTE]
> The **4-wave** bars are the `gfx950-tutorial-v1.0`-build numbers from
> `scripts/run_perf_table.py --rocprof` (1000 dispatches, last-100 average). The **8-wave** bars
> come from `scripts/collect_perf.py`, whose MFMA efficiency is the ATT per-SIMD loop-only figure
> (2 waves/SIMD → per-wave fraction × 2). **BF16 measures ~6% above FP16** here despite the
> nominally identical MFMA rate (a clock/power effect on this build, reproducible across runs).
> Numbers vary run to run (GPU clock) and across MI350-class parts / ROCm / Triton versions. The
> FP16 optimization journey's near-optimal headline (1421 TFLOPS on `gfx950-tutorial-v1.0`) is
> documented in [`a16w16/`](intra_wave/a16w16/).

The 4-wave kernels require the [LLIR Scheduler](../../plugins/llir_scheduler/README.md) and [amdgcnas](../../plugins/amdgcnas/README.md) plugins — build them and enable the stack per [`intra_wave/README.md §2.1`](intra_wave/README.md#21-triton-build-and-the-out-of-tree-plugins). The 8-wave kernels schedule themselves with `warp_pipeline_stage` (no plugins, no env vars).

## ROCm

This tutorial assumes **ROCm ≥ 7.0**. The benchmarking and trace
collection scripts (`scripts/run_perf_table.py`, `scripts/run_att.py`,
`scripts/run_counter_collection.py`, `scripts/calc_kernel_time.py`) drive
`rocprofv3` from the ROCm 7.0 line; in particular they pass `-f csv`
where rocprofv3 7.0+ now defaults to a binary `.db` output, and
`scripts/install_att_decoder.sh` fetches the ROCm 7.0-style
`librocprof-trace-decoder.so` artifact. Earlier ROCm releases (notably
6.5) ship a different rocprofv3 with V2-style trace-decoder libraries
and different CLI defaults, and are not supported by these scripts.

## Learning path

This repository is organized as both a tutorial and a comparison of two system designs.

For readers new to Gluon, we recommend the following path:

1. Start with [`intra_wave/a16w16/`](intra_wave/a16w16/) to learn the optimization journey from a naive GEMM to a near-peak implementation.
2. Continue with [`intra_wave/a8w8/`](intra_wave/a8w8/) and [`intra_wave/a4w4/`](intra_wave/a4w4/) to see how the same design extends to lower-precision kernels.
3. Finally, study the [`inter_wave/`](inter_wave/README.md) kernels to compare an alternative scheduling model that reaches similar performance through a different system design.

Together, these kernels illustrate two complementary approaches to building high-performance GPU software: moving scheduling intelligence into the compiler, or expressing it directly in the kernel structure.
