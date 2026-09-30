# intra_wave — 4-wave GEMM kernels (LLIR scheduler + amdgcnas)

The **4-wave** route: one wave per SIMD, where the out-of-tree LLIR scheduler and the
amdgcnas post-assembly peephole interleave MFMA instructions with memory operations, and the
kernels pin their MFMA accumulators to AGPRs. This is the tutorial's main teaching arc — the FP16 `a16w16`
optimization journey (v0 → v9) and its BF8 (`a8w8`) and MXFP4 (`a4w4`) applications.

> See the [GEMM family README](../README.md) for the directory map, the full version
> catalog, the 4-wave-vs-8-wave performance summary, and the ROCm requirement.

## 1. Overview

The 4-wave kernels run **1 wave/SIMD** and rely on the compiler to interleave the MFMA and
memory streams. The two plugins that make that work — `llirSched` and `amdgcnas` — and the
accumulator pinning the kernels do themselves are described in [§2.1](#21-triton-build-and-the-out-of-tree-plugins); §2.2–§2.3
show how to run the benchmarks, and §3–§4 walk through the FP16 journey and its BF8 / MXFP4
applications.

## 2. Prerequisites

### 2.1 Triton Build and the Out-of-tree Plugins

The LLIR Scheduler and amdgcnas ship as **out-of-tree plugins in this repo** — [`plugins/llir_scheduler/`](../../../plugins/llir_scheduler/README.md) (an LLVM pass plugin, `libLlirSched.so`) and [`plugins/amdgcnas/`](../../../plugins/amdgcnas/README.md) (a pure-Python post-assembly hook). The kernels themselves keep their MFMA accumulators in AGPRs from a16w16 v7 on (and in a8w8 and a4w4), through Gluon's per-call `cd_regclass="a"` option — not a plugin, see [a16w16 v7 §4.3](a16w16/v7_sliceN/README.md#43-pinning-the-accumulators-cd_regclass). Both plugins and the pins are essential for the kernels' performance; see the component table below.

**Build.** Build Triton from the [`gfx950-tutorial-v2.3`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v2.3) annotated tag on `triton-lang/triton`, **with default symbol visibility** (`TRITON_EXT_ENABLED=1`) so the LLVM plugin can resolve LLVM symbols from `libtriton` at load time:

```bash
git clone https://github.com/triton-lang/triton -b gfx950-tutorial-v2.3 /tmp/triton
cd /tmp/triton && TRITON_EXT_ENABLED=1 pip install -e .
```

The tag is plain upstream `main` at `d19d4ca14`: nothing is carried on top of it. What the tutorial relies on is all upstream at that commit — `cd_regclass` ([#11792](https://github.com/triton-lang/triton/pull/11792)); [#10849](https://github.com/triton-lang/triton/pull/10849), which keeps the target machine for LLVM pass plugins; and [#11763](https://github.com/triton-lang/triton/pull/11763), which stops passing `amdgpu-use-amdgpu-trackers` for gfx950. Upstream Triton builds AMDGCN with a separately pinned LLVM: the core LLVM is `b010a18d` and the AMD codegen LLVM is `6bc4aaf6`.

Without `TRITON_EXT_ENABLED=1` the default `-fvisibility=hidden` build exports no LLVM symbols and `PassPlugin::Load` fails with `undefined symbol`. The prebuilt `plugins/llir_scheduler/libLlirSched.so` is ABI-locked to the core LLVM (`b010a18d`); if that pin moves, rebuild it from `plugins/llir_scheduler/LlirSchedPlugin.cpp` (see that plugin's README). The TFLOPS numbers quoted in this tutorial are reproduced against `gfx950-tutorial-v2.3` on a well-performing MI355X; the relative structure (`base` vs. `llir` vs. `llir+amdgcnas`) is expected to remain stable across later pins.

**Upstream trajectory.** Shipping these as out-of-tree plugins is a stopgap — both plugins, and the accumulator pinning, are targeted for the LLVM backend. The LLIR scheduler will be implemented as a scheduling pass in the LLVM backend; the AGPR placement is already a per-MFMA Gluon option (`cd_regclass`), and the LLVM backend's AMDGPU register allocator is gaining the same choice (the `RewriteMFMAFormStage` pass); the post-assembly peephole is a longer-term target for an LLVM MachineInstr-level pass. See [`/docs/performance_philosophy.md §4–§5`](../../../docs/performance_philosophy.md#4-llirsched-cd_regclass-pins-and-amdgcnas-scaffolding-for-the-new-model) for the full reasoning.

**Why these tools exist.** Upstream LLVM's scheduling and register-allocation passes were designed for the discovery model: they receive thread-level IR, recover dependencies by analysis, and solve the resulting NP-hard problems with heuristics. Gluon's block-level programming model makes those problems smaller — dependencies are *engineered* at the block level (e.g., `DOT`, `local_load`, and `buffer_load` are designed to be independent within a 3-stage pipeline), so at the instruction level, MFMAs, `ds_read`s, and `buffer_load`s can be interleaved by a simple throughput-model pass. Likewise, register budgets have a closed-form expression at block level, so allocation becomes a matter of honoring that budget rather than solving graph coloring.

`llirSched`, the `cd_regclass` pins, and `amdgcnas` are the minimum tools that honor this block-level contract today. `llirSched` (scheduling) and the pins (register allocation) are not general-purpose replacements for LLVM's `misched` and register allocator — on arbitrary C-like code they would not make sense; on Gluon-shaped kernels they just honor the schedule and register budget the kernel already engineered. `amdgcnas` does neither scheduling nor allocation: it is a post-assembly LICM + MFMA/scalar-interleave peephole that closes the residual gaps left after codegen, which the earlier passes structurally cannot reach. Together they recover the MFMA efficiency the upstream LLVM flow loses, and their underlying ideas are being integrated into LLVM itself in collaboration with LLVM engineers, so that upstream LLVM will eventually produce the same output. See [`docs/performance_philosophy.md`](../../../docs/performance_philosophy.md) for the full argument.

**The two plugins.** The speedups come from two independently-toggleable plugins. Each has its own enable mechanism and its own `run_perf_table.py` config; the configs are **cumulative**, so each perf-table row's number reflects the whole stack up to that point.

| Plugin | What it does | Enable for a manual (dry) run | `run_perf_table.py` config |
|-----------|--------------|-------------------------------|----------------------------|
| **llirSched** | interleave MFMA with memory ops (throughput-model instruction scheduler) | `LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so` | `llir` |
| **amdgcnas** | post-assembly peephole (LICM + MFMA/scalar interleave) | `TRITON_AMDGCNAS_PLUGIN=1` | `llir+amdgcnas` |

**Accumulator pinning is part of the kernels, not a config.** From a16w16 v7 on, and in a8w8 and a4w4, every MFMA call passes `cd_regclass="a"` (Gluon's `gl.amd.cdna3.mfma` / `gl.amd.cdna4.mfma_scaled` option, upstream since [#11792](https://github.com/triton-lang/triton/pull/11792)). Triton pins each accumulator's C and D operands in AGPRs with an empty tied inline asm (`"=a,0"`), so the register allocator never gets the choice of the VGPR form and never shuffles accumulators between register files inside the loop. The ladder introduces this in [v7](a16w16/v7_sliceN/README.md#43-pinning-the-accumulators-cd_regclass), where the copy problem that starts in v5 is fixed; v0–v6 are unpinned. The LLIR scheduler keeps each pin next to its MFMA when it moves the MFMA and fences the pinned MFMAs, otherwise the pins turn into `v_accvgpr` copies (see [`plugins/llir_scheduler/`](../../../plugins/llir_scheduler/README.md#register-class-pins)). **Tradeoff**: accumulators in AGPRs need `v_accvgpr_read` copies in the epilogue, because `v_cvt` (used to downcast FP32 accumulators to the output dtype) requires VGPR inputs. Acceptable for compute-bound GEMM with large K (~95% time in the main loop), potentially harmful where the epilogue is a larger fraction of runtime.

> **Before `gfx950-tutorial-v2.2`, pinning was a process-wide switch** called force-agpr (`TRITON_FORCE_MFMA_AGPR=1`, carried on the tutorial's Triton fork): it set `amdgpu-mfma-vgpr-form=false` for every kernel and asked the kernels for `amdgpu-agpr-alloc=256`, and the perf tables carried it as a separate config. `cd_regclass` makes the same choice per MFMA call, from the kernel source, on upstream Triton. The LLVM team's **`RewriteMFMAFormStage`** pass, which chooses AGPR vs. VGPR form for each MFMA's C/D based on register pressure, is the longer-term replacement for both.

**Requirements.** Both plugins need Triton built from the `gfx950-tutorial-v2.3` tag. **llirSched** additionally requires the `TRITON_EXT_ENABLED=1` (default-visibility) build and libtriton loaded with `RTLD_GLOBAL` so the LLVM plugin can resolve LLVM symbols — `bench.py` sets `RTLD_GLOBAL` automatically whenever `LLVM_PASS_PLUGIN_PATH` is set. **amdgcnas** is pure Python and needs neither `TRITON_EXT_ENABLED` nor the plugin `.so`. The pins need only upstream Triton's `cd_regclass` (any build after [#11792](https://github.com/triton-lang/triton/pull/11792)).

**1. llirSched — the LLIR scheduler** (out-of-tree LLVM pass plugin [`plugins/llir_scheduler/`](../../../plugins/llir_scheduler/README.md), `libLlirSched.so`) is an LLVM-IR-level pass that interleaves MFMA instructions with memory operations (global loads, LDS reads/writes, async copies) based on the **throughput model** of those memory operations, matching MFMA issue rate to memory-operation completion rate. To preserve this scheduling, it pins each region with `llvm.amdgcn.sched.barrier(0)` in front of every memory anchor, so LLVM's machine scheduler keeps the interleave instead of clustering the MFMAs (no misched-disable needed). Without it, the backend clusters all MFMAs together, causing register spills and MFMA stalls. See [a16w16 v5 section 5](a16w16/v5_local_prefetch/README.md#5-introduction-to-the-llir-scheduler) for the motivation and [`plugins/llir_scheduler/`](../../../plugins/llir_scheduler/README.md) for the plugin itself. The scheduler:
- Classifies memory operations into GR (global read), LR (local read), and LW (local write) anchors
- Distributes MFMAs among anchors based on throughput (e.g., 4 MFMAs per global load for 16-cycle MFMA, 2 for 32-cycle)
- For MXFP4 kernels, moves scale-related LR instructions to interleave with global loads and allocates remaining MFMAs after ds_write to cover LDS port contention

**2. amdgcnas — the post-assembly peephole** (`TRITON_AMDGCNAS_PLUGIN=1`, a pure-Python `amdgcn`-stage hook installed by `bench.py` — no compiler rebuild) optimizes the final generated assembly. **It is *only* the peephole** — the AGPR placement that older docs bundled under "amdgcnas" is done by the kernels' `cd_regclass` pins above. It does:
- **LICM (Loop Invariant Code Motion)**: Hoists loop-invariant instructions (e.g., LDS address calculations) to the loop prologue, with register renaming when the hoisted output is redefined inside the loop.
- **Peephole optimizations**: Interleaves MFMA with scalar instructions (`s_waitcnt`, `s_barrier`, scalar address computation for buffer loads) to maintain continuous MFMA throughput. These scalar instructions are inserted during MIR-level codegen, after the LLIR scheduler has run, so `llirSched` structurally cannot reach them — this peephole is the only pass that can.

**Relative contributions.** On the `gfx950-tutorial-v2.3` pin, all four kernels
below carry the pins in every config, so the split is between the two plugins:

| kernel | `base` | `llir` | `llir+amdgcnas` |
|---|---|---|---|
| a16w16 v7 | 1210 TFLOPS, 66.5% | 1554, 97.3% | 1583, 98.1% |
| a16w16 v9 | 1389, 71.8% | 1591, 95.9% | 1605, 98.4% |
| a8w8      | 2807, 72.0% | 3351, 96.2% | 3417, 99.7% |
| a4w4 v1   | 5134, 68.0% | 5644, 84.2% | 5815, 94.1% |

The LLIR scheduler is the large step: +10% to +28% of throughput and 16–31 points of MFMA
efficiency, because the stock scheduler does not interleave the MFMA and memory streams. amdgcnas
adds another 1–10 points of efficiency — most on MXFP4, where the scale pipeline leaves the densest
SALU activity — worth 1–3% of throughput. The pins do their work underneath both: measured on
v7 with and without them, they take the `llir` loop from 116 `v_accvgpr_*` copies to none (+8.8%)
([v7 §4.3](a16w16/v7_sliceN/README.md#43-pinning-the-accumulators-cd_regclass)), and amdgcnas
runs only on the pinned kernels, since it assumes the accumulators are in AGPRs. The kernels
that sit at the 512-register ceiling without pins remain at the mercy of allocator policy — on this
pin the unpinned stock v6 build spills 241 registers (see [v6](a16w16/v6_loop_unroll/README.md)).
The upstream stories differ accordingly: pinning maps to an allocator-policy change, the
scheduler to a backend scheduling pass, and the SALU-level peephole needs a MachineInstr-level pass.

### 2.2 Running Benchmarks

The easiest way to run benchmarks with all optimizations enabled is `run_perf_table.py`:

```bash
# FP16 (a16w16)
python scripts/run_perf_table.py --kernel a16w16 --versions 8 --configs llir+amdgcnas --K 8192 --dtype fp16 --rocprof

# BF8 (a8w8)
python scripts/run_perf_table.py --kernel a8w8 --configs llir+amdgcnas --K 16384 --rocprof

# MXFP4 (a4w4)
python scripts/run_perf_table.py --kernel a4w4 --versions 1 --configs llir+amdgcnas --K 32768 --rocprof
```

This script automatically:
- Sets the environment variables for llirSched and amdgcnas
- Collects kernel traces using rocprofv3
- Calculates and reports TFLOPS, VGPRs, spills, and MFMA efficiency

### 2.3 Manual Workflow

To run benchmarks manually, export the plugin environment variables, then run from the kernel directory. The env is the same for all three kernels — the plugin `.so` path is absolute, so it works from any kernel dir. This is the full `llir+amdgcnas` config; drop `TRITON_AMDGCNAS_PLUGIN` for `llir`, or unset both for `base`:

```bash
# Enable llirSched + amdgcnas (once per shell)
export LLVM_PASS_PLUGIN_PATH=$(git rev-parse --show-toplevel)/plugins/llir_scheduler/libLlirSched.so
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

The [a16w16/](a16w16/) directory documents a step-by-step optimization journey from a naive 542 TFLOPS baseline to a near-optimal 1605 TFLOPS implementation—a **~3.0× improvement** through 10 versions (v0–v9).

**Start here** to learn how to write high-performance Gluon kernels. Then proceed to [a8w8/](a8w8/) and [a4w4/](a4w4/) in that order.

## 4. BF8 and MXFP4: Applying the Same Design

The optimization principles from the FP16 journey apply directly to BF8 and MXFP4. The final kernel for all three data types shares the same fundamental design: M+N slicing, 3-stage pipeline, loop unrolling by 2, accumulators pinned to AGPRs, and the LLIR scheduler + amdgcnas optimizations.

| Aspect | FP16 (a16w16) | BF8 (a8w8) | MXFP4 (a4w4) |
|--------|---------------|------------|--------------|
| Tile size | 256x256x64 | 256x256x128 | 256x256x256 |
| MFMA instruction | `v_mfma_f32_16x16x32_f16` | `v_mfma_scale_f32_16x16x128_f8f6f4` | `v_mfma_scale_f32_16x16x128_f8f6f4` |
| cbsz / blgp | N/A | 1 / 1 (E5M2) | 4 / 4 (E2M1) |
| MFMA cycles | 16 | 32 (cbsz/blgp <= 1) | 16 (cbsz/blgp > 1) |
| Scaling | None | None | Per-group e8m0 |

The [a8w8/](a8w8/) directory provides the final optimized BF8 kernel. If you understand the FP16 journey, you will understand the BF8 kernel. The key differences are tile shape, MFMA instruction, and LDS padding.

The [a4w4/](a4w4/) directory implements the MXFP4 kernel, whose genuinely new element is the per-group scale pipeline: every 32 e2m1 elements share an 8-bit e8m0 scale that must be loaded and laid out for `mfma_scaled`. It ships in two versions — `v0_sliceN` stages scales through LDS with a `local_store` → `local_load` round-trip, while the final `v1_sliceMN` loads them straight into LDS via `buffer_load_to_lds` alongside the input tiles (no `local_store`) and uses M+N slicing for a more balanced design. See the [a4w4 README](a4w4/README.md) for full details.
