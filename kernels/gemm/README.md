# GEMM Kernels in Gluon

This directory contains **high-performance GEMM kernels written in Gluon**, targeting **AMD MI350/355 GPUs** (gfx950).

The goal is not just to provide fast kernels, but to **teach how to design, analyze, and optimize GEMM kernels** on AMD hardware—from memory layout to instruction scheduling.

## Directory Structure

```
gemm/
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
        ├── v0_sliceMN_BK256_nS2/          #   byte-shuffle B scale (baseline)
        ├── v1_combineBsc_BK256_nS2/       #   combined transpose-read B scale (recommended)
        └── v2_mfma32x32x64_BK256_nS2/     #   32×32×64 MFMA + conflict-free LDS layout
```

Two routes to peak MFMA utilization on the *same* problems. The **4-wave** kernels in
[`intra_wave/`](intra_wave/) use the LLIR scheduler + force-agpr + amdgcnas ([§2.1](#21-triton-build-and-the-out-of-tree-plugins));
the **8-wave** kernels in [`inter_wave/`](inter_wave/) use wave-level `warp_pipeline_stage`
scheduling with no AGPRs ([§5](#5-8-wave-warp-pipeline-variants)). New here? Start with
[`intra_wave/a16w16/`](intra_wave/a16w16/) for the full step-by-step walkthrough.

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
| **`inter_wave/a4w4`** (MXFP4) | `v0_sliceMN_BK256_nS2` | Scale + pipeline | Byte-shuffle B scale (baseline) |
| | `v1_combineBsc_BK256_nS2` | Scale + pipeline | Combined transpose-read B scale *(recommended)* |
| | `v2_mfma32x32x64_BK256_nS2` | MFMA shape | 32×32×64 MFMA + conflict-free LDS layout |

## 1. Performance Summary

Measured on a single MI355X (gfx950), Triton built from the `gfx950-tutorial-v1.0` tag, rocprof
cold-rotating (1000 dispatches, last-100 average). The **4-wave** kernels run with the LLIR
scheduler + force-agpr + amdgcnas (see [§2.1](#21-triton-build-and-the-out-of-tree-plugins)); the
**8-wave** kernels run `warp_pipeline_stage` with no AGPRs (no env vars — see [§5](#5-8-wave-warp-pipeline-variants)).

| Data Type | Shape           | Solution                    | TFLOPS | MFMA Eff. |
|-----------|-----------------|-----------------------------|--------|-----------|
| FP16      | 4096×4096×8192  | 4-wave (`intra_wave/a16w16` v9)        |   1421 |    98.66% |
| FP16      | 4096×4096×8192  | 8-wave (`inter_wave/a16w16` v1)  |   1442 |    99.8%  |
| BF16      | 4096×4096×8192  | 4-wave (`intra_wave/a16w16` v9)        |   1514 |    98.66% |
| BF16      | 4096×4096×8192  | 8-wave (`inter_wave/a16w16` v1)  |   1534 |    99.8%  |
| BF8       | 4096×4096×16384 | 4-wave (`intra_wave/a8w8`)             |   3232 |    99.52% |
| BF8       | 4096×4096×16384 | 8-wave (`inter_wave/a8w8` v1)    |   3094 |    99.9%  |
| MXFP4     | 4096×4096×32768 | 4-wave (`intra_wave/a4w4` v1)          |   5189 |    93.86% |
| MXFP4     | 4096×4096×32768 | 8-wave (`inter_wave/a4w4` v1)    |   4938 |    80.0%  |
| MXFP4     | 4096×4096×32768 | 8-wave (`inter_wave/a4w4` v2)    |   4799 |    98.0%  |

> [!NOTE]
> The **4-wave** rows are the `gfx950-tutorial-v1.0`-build numbers from
> `scripts/run_perf_table.py --rocprof` (1000 dispatches, last-100 average). The **8-wave** rows
> come from `collect_perf.py`, whose MFMA efficiency is the ATT per-SIMD loop-only figure
> (2 waves/SIMD → per-wave fraction × 2). **BF16 measures ~6% above FP16** here despite the
> nominally identical MFMA rate (a clock/power effect on this build, reproducible across runs).
> Numbers vary run to run (GPU clock) and across MI350-class parts / ROCm / Triton versions. The
> FP16 optimization journey's near-optimal headline (1421 TFLOPS on `gfx950-tutorial-v1.0`) is
> documented in [`a16w16/`](intra_wave/a16w16/).

The 4-wave kernels require the [LLIR Scheduler](../../plugins/llir_scheduler/README.md) and [amdgcnas](../../plugins/amdgcnas/README.md) for optimal performance; the 8-wave kernels schedule themselves with `warp_pipeline_stage`.

## 2. Prerequisites

### 2.0 ROCm

This tutorial assumes **ROCm ≥ 7.0**. The benchmarking and trace
collection scripts (`scripts/run_perf_table.py`, `scripts/run_att.py`,
`scripts/run_counter_collection.py`, `scripts/calc_kernel_time.py`) drive
`rocprofv3` from the ROCm 7.0 line; in particular they pass `-f csv`
where rocprofv3 7.0+ now defaults to a binary `.db` output, and
`scripts/install_att_decoder.sh` fetches the ROCm 7.0-style
`librocprof-trace-decoder.so` artifact. Earlier ROCm releases (notably
6.5) ship a different rocprofv3 with V2-style trace-decoder libraries
and different CLI defaults, and are not supported by these scripts.

### 2.1 Triton Build and the Out-of-tree Plugins

The LLIR Scheduler and amdgcnas ship as **out-of-tree plugins in this repo** — [`plugins/llir_scheduler/`](../../plugins/llir_scheduler/README.md) (an LLVM pass plugin, `libLlirSched.so`) and [`plugins/amdgcnas/`](../../plugins/amdgcnas/README.md) (a pure-Python post-assembly hook). The third component, **force-agpr**, is an env-var RA hint (not a plugin). All three are essential for the kernels (a16w16, a8w8, a4w4); see the component table below.

**Build.** Build Triton from the [`gfx950-tutorial-v1.0`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v1.0) annotated tag on `triton-lang/triton`, **with default symbol visibility** (`TRITON_EXT_ENABLED=1`) so the LLVM plugin can resolve LLVM symbols from `libtriton` at load time:

```bash
git clone https://github.com/triton-lang/triton -b gfx950-tutorial-v1.0 /tmp/triton
cd /tmp/triton && TRITON_EXT_ENABLED=1 pip install -e .
```

Without `TRITON_EXT_ENABLED=1` the default `-fvisibility=hidden` build exports no LLVM symbols and `PassPlugin::Load` fails with `undefined symbol`. The prebuilt `plugins/llir_scheduler/libLlirSched.so` is ABI-locked to this tag's LLVM pin (`62b7cf96`); if the pin moves, rebuild it from `plugins/llir_scheduler/LlirSchedPlugin.cpp` (see that plugin's README). The TFLOPS numbers and counter values quoted in this tutorial are reproduced against `gfx950-tutorial-v1.0`; the relative structure (`llir` vs. `llir+force-agpr` vs. `llir+force-agpr+amdgcnas`) is expected to remain stable across later pins.

**Upstream trajectory.** Shipping these as out-of-tree plugins is a stopgap — all three are targeted for the LLVM backend. The LLIR scheduler will be implemented as a scheduling pass in the LLVM backend; the RA hints will move into the LLVM backend's AMDGPU register allocator (the `RewriteMFMAFormStage` pass — see the force-agpr note above); the post-assembly peephole is a longer-term target for an LLVM MachineInstr-level pass. See [`/docs/performance_philosophy.md §4–§5`](../../docs/performance_philosophy.md#4-llirsched-force-agpr-and-amdgcnas-scaffolding-for-the-new-model) for the full reasoning.

**Why these tools exist.** Upstream LLVM's scheduling and register-allocation passes were designed for the discovery model: they receive thread-level IR, recover dependencies by analysis, and solve the resulting NP-hard problems with heuristics. Gluon's block-level programming model makes those problems smaller — dependencies are *engineered* at the block level (e.g., `DOT`, `local_load`, and `buffer_load` are designed to be independent within a 3-stage pipeline), so at the instruction level, MFMAs, `ds_read`s, and `buffer_load`s can be interleaved by a simple throughput-model pass. Likewise, register budgets have a closed-form expression at block level, so allocation becomes a matter of honoring that budget rather than solving graph coloring.

`llirSched`, `force-agpr`, and `amdgcnas` are the minimum tools that honor this block-level contract today. `llirSched` (scheduling) and `force-agpr` (register allocation) are not general-purpose replacements for LLVM's `misched` and register allocator — on arbitrary C-like code they would not make sense; on Gluon-shaped kernels they just honor the schedule and register budget the kernel already engineered. `amdgcnas` does neither scheduling nor allocation: it is a post-assembly LICM + MFMA/scalar-interleave peephole that closes the residual gaps left after codegen, which the earlier passes structurally cannot reach. Together they recover the MFMA efficiency the upstream LLVM flow loses, and their underlying ideas are being integrated into LLVM itself in collaboration with LLVM engineers, so that upstream LLVM will eventually produce the same output. See [`docs/performance_philosophy.md`](../../docs/performance_philosophy.md) for the full argument.

**The three components.** The speedups come from three independently-toggleable components. Each has its own enable mechanism and its own `run_perf_table.py` config; the configs are **cumulative**, so each perf-table row's number reflects the whole stack up to that point.

| Component | What it does | Enable for a manual (dry) run | `run_perf_table.py` config |
|-----------|--------------|-------------------------------|----------------------------|
| **llirSched** | interleave MFMA with memory ops (throughput-model instruction scheduler) | `LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so` `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1` | `llir` |
| **force-agpr** | force MFMA accumulators into AGPRs (RA hint) | `TRITON_FORCE_MFMA_AGPR=1` | `llir+force-agpr` |
| **amdgcnas** | post-assembly peephole (LICM + MFMA/scalar interleave) | `TRITON_AMDGCNAS_PLUGIN=1` | `llir+force-agpr+amdgcnas` |

**Requirements.** All three need Triton built from the `gfx950-tutorial-v1.0` tag. **llirSched** additionally requires the `TRITON_EXT_ENABLED=1` (default-visibility) build and libtriton loaded with `RTLD_GLOBAL` so the LLVM plugin can resolve LLVM symbols — `bench.py` sets `RTLD_GLOBAL` automatically whenever `LLVM_PASS_PLUGIN_PATH` is set. **force-agpr** and **amdgcnas** work on a stock v1.0 build (no `TRITON_EXT_ENABLED`, no plugin `.so`).

**1. llirSched — the LLIR scheduler** (out-of-tree LLVM pass plugin [`plugins/llir_scheduler/`](../../plugins/llir_scheduler/README.md), `libLlirSched.so`) is an LLVM-IR-level pass that interleaves MFMA instructions with memory operations (global loads, LDS reads/writes, async copies) based on the **throughput model** of those memory operations, matching MFMA issue rate to memory-operation completion rate. To preserve this scheduling, it pins each region with `llvm.amdgcn.sched.barrier(0)` after every memory anchor, so LLVM's machine scheduler keeps the interleave instead of clustering the MFMAs (no misched-disable needed). Without it, the backend clusters all MFMAs together, causing register spills and MFMA stalls. See [a16w16 v5 section 5](intra_wave/a16w16/v5_local_prefetch/README.md#5-introduction-to-the-llir-scheduler) for the motivation and [`plugins/llir_scheduler/`](../../plugins/llir_scheduler/README.md) for the plugin itself. The scheduler:
- Classifies memory operations into GR (global read), LR (local read), and LW (local write) anchors
- Distributes MFMAs among anchors based on throughput (e.g., 4 MFMAs per global load for 16-cycle MFMA, 2 for 32-cycle)
- For MXFP4 kernels, moves scale-related LR instructions to interleave with global loads and allocates remaining MFMAs after ds_write to cover LDS port contention

**2. force-agpr — reserve AGPRs for MFMA accumulators.** A single env var `TRITON_FORCE_MFMA_AGPR=1` drives two paired effects: (a) the tutorial kernels set `llvm_fn_attrs="amdgpu-agpr-alloc=256"`, directing LLVM's register allocator to reserve 256 AGPRs for MFMA accumulators; and (b) `llvm.cc` sets `amdgpu-mfma-vgpr-form=false`, preventing LLVM from using the VGPR form of MFMA instructions. Together they keep accumulators in AGPRs and reduce VGPR pressure. This addresses the register-allocation challenges in [a16w16 v7 sections 4.3–4.4](intra_wave/a16w16/v7_sliceN/README.md#43-register-allocation-workaround). **Tradeoff**: forcing accumulators into AGPRs maximizes `v_accvgpr_read` copies in the epilogue, because `v_cvt` (used to downcast FP32 accumulators to the output dtype) requires VGPR inputs. Acceptable for compute-bound GEMM with large K (~95% time in the main loop), potentially harmful where the epilogue is a larger fraction of runtime.

> **`amdgpu-mfma-vgpr-form=0` is a temporary stopgap.** It is a blunt instrument — it forces *all* MFMA C/D operands into AGPRs, which is what we want inside the loop but emits more `v_accvgpr_*` copies than necessary in the epilogue. The LLVM team is developing the **`RewriteMFMAFormStage`** pass, which chooses AGPR vs. VGPR form for each MFMA's C/D based on register pressure. Once it lands and is on by default, `amdgpu-mfma-vgpr-form=0` can be dropped from `llvm.cc` (and `TRITON_FORCE_MFMA_AGPR` reduced to just the `amdgpu-agpr-alloc` hint).

**3. amdgcnas — the post-assembly peephole** (`TRITON_AMDGCNAS_PLUGIN=1`, a pure-Python `amdgcn`-stage hook installed by `bench.py` — no compiler rebuild) optimizes the final generated assembly. **It is *only* the peephole** — the RA hint that older docs bundled under "amdgcnas" is now the separate **force-agpr** component above. It does:
- **LICM (Loop Invariant Code Motion)**: Hoists loop-invariant instructions (e.g., LDS address calculations) to the loop prologue, with register renaming when the hoisted output is redefined inside the loop.
- **Peephole optimizations**: Interleaves MFMA with scalar instructions (`s_waitcnt`, `s_barrier`, scalar address computation for buffer loads) to maintain continuous MFMA throughput. These scalar instructions are inserted during MIR-level codegen, after the LLIR scheduler has run, so `llirSched` structurally cannot reach them — this peephole is the only pass that can.

**Relative contributions.** force-agpr and amdgcnas are not equally important across dtypes. On FP16 and BF8, `force-agpr` alone closes 75–85% of the MFMA-efficiency gap between `llir` and `llir+force-agpr+amdgcnas`, landing within 1–2% TFLOPS of the full stack; the amdgcnas peephole adds only ~+2pp on top. MXFP4 leans more on the peephole: on `v1_sliceMN`, `force-agpr` closes only about half the gap (`llir` ~71% → `llir+force-agpr` ~84%), and amdgcnas adds the remaining ~+10pp to reach ~94% — the paired scale loads create denser SALU activity for the peephole to pack. The two therefore have different upstream stories: force-agpr maps to an LLVM allocator-policy change that can land soon; the SALU-level peephole's natural home is a MachineInstr-level pass yet to be written.

### 2.2 Running Benchmarks

The easiest way to run benchmarks with all optimizations enabled is `run_perf_table.py`:

```bash
# FP16 (a16w16)
python scripts/run_perf_table.py --kernel a16w16 --versions 8 --configs llir+force-agpr+amdgcnas --K 8192 --dtype fp16 --rocprof

# BF8 (a8w8)
python scripts/run_perf_table.py --kernel a8w8 --configs llir+force-agpr+amdgcnas --K 16384 --rocprof

# MXFP4 (a4w4)
python scripts/run_perf_table.py --kernel a4w4 --versions 1 --configs llir+force-agpr+amdgcnas --K 32768 --rocprof
```

This script automatically:
- Sets the environment variables for llirSched, force-agpr, and amdgcnas
- Collects kernel traces using rocprofv3
- Calculates and reports TFLOPS, VGPRs, spills, and MFMA efficiency

### 2.3 Manual Workflow

To run benchmarks manually, export the component environment variables, then run from the kernel directory. The env is the same for all three kernels — the plugin `.so` path is absolute, so it works from any kernel dir. This is the full `llir+force-agpr+amdgcnas` config; drop `TRITON_AMDGCNAS_PLUGIN` for `llir+force-agpr`, drop `TRITON_FORCE_MFMA_AGPR` as well for `llir`, or unset all of them for `base`:

```bash
# Enable llirSched + force-agpr + amdgcnas (once per shell)
export LLVM_PASS_PLUGIN_PATH=$(git rev-parse --show-toplevel)/plugins/llir_scheduler/libLlirSched.so
export LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1  # llirSched
export TRITON_FORCE_MFMA_AGPR=1                # force-agpr
export TRITON_AMDGCNAS_PLUGIN=1                # amdgcnas

# FP16 (from kernels/gemm/intra_wave/a16w16/)
python bench.py --version 8 --K 8192 --dtype fp16

# BF8 (from kernels/gemm/intra_wave/a8w8/)
python bench.py --K 16384

# MXFP4 (from kernels/gemm/intra_wave/a4w4/)
python bench.py --version 1 --K 32768
```

For accurate performance measurement, the `--rocprof` flag runs the kernel 1000 times with rotating buffers but does not print performance numbers. To collect measurements:

1. Collect the kernel trace (`-d` specifies the output directory; `-f csv`
   selects CSV output, which `calc_kernel_time.py` reads — rocprofv3 on
   ROCm 7.0+ defaults to a binary `.db` format):
   ```bash
   # with the plugin env vars from §2.3 still exported
   rocprofv3 --kernel-trace -f csv -d out -- python bench.py --version 8 --K 8192 --dtype fp16 --rocprof
   ```

2. Calculate kernel time from the trace. The CSV file may be in a nested directory under the output directory—locate it first. Output is in microseconds by default:
   ```bash
   python ../../../scripts/calc_kernel_time.py [trace_csv_file] [kernel_name]
   ```

3. Convert to TFLOPS: `TFLOPS = 2 × M × N × K / (time_in_us × 10^6)`

## 3. FP16: The Optimization Journey

The [a16w16/](intra_wave/a16w16/) directory documents a step-by-step optimization journey from a naive 541 TFLOPS baseline to a near-optimal 1421 TFLOPS implementation—a **~2.6× improvement** through 10 versions (v0–v9).

**Start here** to learn how to write high-performance Gluon kernels. Then proceed to [a8w8/](intra_wave/a8w8/) and [a4w4/](intra_wave/a4w4/) in that order.

## 4. BF8 and MXFP4: Applying the Same Design

The optimization principles from the FP16 journey apply directly to BF8 and MXFP4. The final kernel for all three data types shares the same fundamental design: M+N slicing, 3-stage pipeline, loop unrolling by 2, and the LLIR scheduler + amdgcnas optimizations.

| Aspect | FP16 (a16w16) | BF8 (a8w8) | MXFP4 (a4w4) |
|--------|---------------|------------|--------------|
| Tile size | 256x256x64 | 256x256x128 | 256x256x256 |
| MFMA instruction | `v_mfma_f32_16x16x32_f16` | `v_mfma_scale_f32_16x16x128_f8f6f4` | `v_mfma_scale_f32_16x16x128_f8f6f4` |
| cbsz / blgp | N/A | 1 / 1 (E5M2) | 4 / 4 (E2M1) |
| MFMA cycles | 16 | 32 (cbsz/blgp <= 1) | 16 (cbsz/blgp > 1) |
| Scaling | None | None | Per-group e8m0 |

The [a8w8/](intra_wave/a8w8/) directory provides the final optimized BF8 kernel. If you understand the FP16 journey, you will understand the BF8 kernel. The key differences are tile shape, MFMA instruction, and LDS padding.

The [a4w4/](intra_wave/a4w4/) directory implements the MXFP4 kernel, whose genuinely new element is the per-group scale pipeline: every 32 e2m1 elements share an 8-bit e8m0 scale that must be loaded and laid out for `mfma_scaled`. It ships in two versions — `v0_sliceN` stages scales through LDS with a `local_store` → `local_load` round-trip, while the final `v1_sliceMN` loads them straight into LDS via `buffer_load_to_lds` alongside the input tiles (no `local_store`) and uses M+N slicing for a more balanced design. See the [a4w4 README](intra_wave/a4w4/README.md) for full details.

## 5. 8-Wave Warp-Pipeline Variants

Alongside the 4-wave `llir+force-agpr+amdgcnas` kernels above, the repo carries an **8-wave warp-pipeline** version of each GEMM — [`inter_wave/a16w16/`](inter_wave/a16w16/), [`inter_wave/a8w8/`](inter_wave/a8w8/), and [`inter_wave/a4w4/`](inter_wave/a4w4/). These reach high MFMA utilization on the *same* problems by a different route.

Instead of the LLIR scheduler + force-agpr + amdgcnas, they launch **8 warps/CTA (2 waves/SIMD)** and schedule the hot loop at the **wave level** with `warp_pipeline_stage`: the two resident waves per SIMD are kept out of phase so one issues MFMAs while the other issues loads, then they swap (a "ping-pong"). They run with **no AGPRs** (`amdgpu-agpr-alloc=0,0` via `llvm_fn_attrs`), so the f32 accumulators live in VGPRs and **no environment variables are needed**. The theory is in [`docs/warp_pipelining.md`](../../docs/warp_pipelining.md).

> [!IMPORTANT]
> The 4-wave `llir+force-agpr+amdgcnas` toolchain is built around the 4-wave register/schedule model and **fails register allocation at 8 waves**, so it is not used here.

| | inter_wave/a16w16 | inter_wave/a8w8 | inter_wave/a4w4 |
|---|---|---|---|
| Data type | FP16 / BF16 | BF8 (e5m2) | MXFP4 (e2m1) |
| Versions | *(single kernel)* | *(single kernel)* | `v0_sliceMN_BK256_nS2`, `v1_combineBsc_BK256_nS2`, `v2_mfma32x32x64_BK256_nS2` |
| Tile M×N×K | 256×256×64 | 256×256×128 | 256×256×256 |
| MFMA | `mfma` `[16,16,32]` | `mfma_scaled` e5m2 `[16,16,128]` | `mfma_scaled` e2m1 `[16,16,128]` |
| Scheduling | `warp_pipeline_stage`, no-AGPR | same | same |

**Performance** (MI355X, gfx950, 4096×4096, Triton `gfx950-tutorial-v1.0` — a4w4 rows also need `fence_loads` PR #10840 — rocprof cold-rotating; per-SIMD loop MFMA eff):

| Kernel (final version) | K=8192 | K=16384 | K=32768 | VGPR / spills |
|---|---|---|---|---|
| inter_wave/a16w16 (fp16) | 1442 / 99.8% | 1489 / 98.1% | 1287 / 81.6% | 242 / 0 |
| inter_wave/a8w8 (BF8)    | 2853 / 99.7% | 3094 / 99.9% | 2968 / 96.8% | 256 / 13 (loop 0) |
| inter_wave/a4w4 `v1` (MXFP4)  | 4116 / 79.7% | 4630 / 79.9% | 4938 / 80.0% | 256 / 12 (loop 0) |

Run them with each kernel's `collect_perf.py` (no env vars):

```bash
cd kernels/gemm/inter_wave/a16w16 && python collect_perf.py --K 8192 --dtype fp16
cd kernels/gemm/inter_wave/a8w8   && python collect_perf.py --K 8192
cd kernels/gemm/inter_wave/a4w4   && python collect_perf.py --version 1 --K 8192
```

**Where the 8-wave lands vs the 4-wave** (v1.0 build): for **FP16**, the 8-wave kernel now edges the 4-wave `v9` by ~1.5% (1442 vs 1421 @ K=8192) — on the v1.0 build the 4-wave FP16 path sits at 1421. For **BF8**, the tuned 4-wave `llir+force-agpr+amdgcnas` leads (3232 vs 3094 @ K=16384). For **MXFP4**, `v1` (combined B-scale, the default) **beats the 4-wave *base*** at large K (4938 vs 4137 @ K=32768): combining the B scale so it transpose-reads instead of byte-shuffling deleted 118 loop `v_perm`, and the intra-stage `fence_loads` (PR #10840) lifts loop MFMA from ~57% (`v0`) to ~80%, TFLOPS +16–22%. The tuned 4-wave `llir+force-agpr+amdgcnas` (~5.2 PFLOP/s) still leads, as the loop remains LDS/scale-throughput bound. See each `-8wave/README.md` for the full breakdown.

