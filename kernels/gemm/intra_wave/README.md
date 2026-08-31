# intra_wave — 4-wave GEMM kernels (LLIR scheduler + force-agpr + amdgcnas)

The **4-wave** route: one wave per SIMD, where the out-of-tree LLIR scheduler, the
force-agpr RA hints, and the amdgcnas post-assembly peephole interleave MFMA instructions
with memory operations. This is the tutorial's main teaching arc — the FP16 `a16w16`
optimization journey (v0 → v9) and its BF8 (`a8w8`) and MXFP4 (`a4w4`) applications.

> See the [GEMM family README](../README.md) for the directory map, the full version
> catalog, the 4-wave-vs-8-wave performance summary, and the ROCm requirement.

## 1. Overview

The 4-wave kernels run **1 wave/SIMD** and rely on the compiler to interleave the MFMA and
memory streams. The three tools that make that work — `llirSched`, `force-agpr`, and
`amdgcnas` — are described in [§2.1](#21-triton-build-and-the-out-of-tree-plugins); §2.2–§2.3
show how to run the benchmarks, and §3–§4 walk through the FP16 journey and its BF8 / MXFP4
applications.

## 2. Prerequisites

### 2.1 Triton Build and the Out-of-tree Plugins

The LLIR Scheduler and amdgcnas ship as **out-of-tree plugins in this repo** — [`plugins/llir_scheduler/`](../../../plugins/llir_scheduler/README.md) (an LLVM pass plugin, `libLlirSched.so`) and [`plugins/amdgcnas/`](../../../plugins/amdgcnas/README.md) (a pure-Python post-assembly hook). The third component, **force-agpr**, is an env-var RA hint (not a plugin). All three are essential for the kernels (a16w16, a8w8, a4w4); see the component table below.

**Build.** Build Triton from the [`gfx950-tutorial-v2.1`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v2.1) annotated tag on `triton-lang/triton`, **with default symbol visibility** (`TRITON_EXT_ENABLED=1`) so the LLVM plugin can resolve LLVM symbols from `libtriton` at load time:

```bash
git clone https://github.com/triton-lang/triton -b gfx950-tutorial-v2.1 /tmp/triton
cd /tmp/triton && TRITON_EXT_ENABLED=1 pip install -e .
```

Without `TRITON_EXT_ENABLED=1` the default `-fvisibility=hidden` build exports no LLVM symbols and `PassPlugin::Load` fails with `undefined symbol`. The prebuilt `plugins/llir_scheduler/libLlirSched.so` is ABI-locked to this tag's LLVM pin (`b010a18d`); if the pin moves, rebuild it from `plugins/llir_scheduler/LlirSchedPlugin.cpp` (see that plugin's README). The TFLOPS numbers and counter values quoted in this tutorial are reproduced against `gfx950-tutorial-v1.1`; the relative structure (`llir` vs. `llir+force-agpr` vs. `llir+force-agpr+amdgcnas`) is expected to remain stable across later pins.

**Upstream trajectory.** Shipping these as out-of-tree plugins is a stopgap — all three are targeted for the LLVM backend. The LLIR scheduler will be implemented as a scheduling pass in the LLVM backend; the RA hints will move into the LLVM backend's AMDGPU register allocator (the `RewriteMFMAFormStage` pass — see the force-agpr note above); the post-assembly peephole is a longer-term target for an LLVM MachineInstr-level pass. See [`/docs/performance_philosophy.md §4–§5`](../../../docs/performance_philosophy.md#4-llirsched-force-agpr-and-amdgcnas-scaffolding-for-the-new-model) for the full reasoning.

**Why these tools exist.** Upstream LLVM's scheduling and register-allocation passes were designed for the discovery model: they receive thread-level IR, recover dependencies by analysis, and solve the resulting NP-hard problems with heuristics. Gluon's block-level programming model makes those problems smaller — dependencies are *engineered* at the block level (e.g., `DOT`, `local_load`, and `buffer_load` are designed to be independent within a 3-stage pipeline), so at the instruction level, MFMAs, `ds_read`s, and `buffer_load`s can be interleaved by a simple throughput-model pass. Likewise, register budgets have a closed-form expression at block level, so allocation becomes a matter of honoring that budget rather than solving graph coloring.

`llirSched`, `force-agpr`, and `amdgcnas` are the minimum tools that honor this block-level contract today. `llirSched` (scheduling) and `force-agpr` (register allocation) are not general-purpose replacements for LLVM's `misched` and register allocator — on arbitrary C-like code they would not make sense; on Gluon-shaped kernels they just honor the schedule and register budget the kernel already engineered. `amdgcnas` does neither scheduling nor allocation: it is a post-assembly LICM + MFMA/scalar-interleave peephole that closes the residual gaps left after codegen, which the earlier passes structurally cannot reach. Together they recover the MFMA efficiency the upstream LLVM flow loses, and their underlying ideas are being integrated into LLVM itself in collaboration with LLVM engineers, so that upstream LLVM will eventually produce the same output. See [`docs/performance_philosophy.md`](../../../docs/performance_philosophy.md) for the full argument.

**The three components.** The speedups come from three independently-toggleable components. Each has its own enable mechanism and its own `run_perf_table.py` config; the configs are **cumulative**, so each perf-table row's number reflects the whole stack up to that point.

| Component | What it does | Enable for a manual (dry) run | `run_perf_table.py` config |
|-----------|--------------|-------------------------------|----------------------------|
| **llirSched** | interleave MFMA with memory ops (throughput-model instruction scheduler) | `LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so` `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1` | `llir` |
| **force-agpr** | force MFMA accumulators into AGPRs (RA hint) | `TRITON_FORCE_MFMA_AGPR=1` | `llir+force-agpr` |
| **amdgcnas** | post-assembly peephole (LICM + MFMA/scalar interleave) | `TRITON_AMDGCNAS_PLUGIN=1` | `llir+force-agpr+amdgcnas` |

**Requirements.** All three need Triton built from the `gfx950-tutorial-v2.1` tag. **llirSched** additionally requires the `TRITON_EXT_ENABLED=1` (default-visibility) build and libtriton loaded with `RTLD_GLOBAL` so the LLVM plugin can resolve LLVM symbols — `bench.py` sets `RTLD_GLOBAL` automatically whenever `LLVM_PASS_PLUGIN_PATH` is set. **force-agpr** and **amdgcnas** work on a stock v1.0 build (no `TRITON_EXT_ENABLED`, no plugin `.so`).

**1. llirSched — the LLIR scheduler** (out-of-tree LLVM pass plugin [`plugins/llir_scheduler/`](../../../plugins/llir_scheduler/README.md), `libLlirSched.so`) is an LLVM-IR-level pass that interleaves MFMA instructions with memory operations (global loads, LDS reads/writes, async copies) based on the **throughput model** of those memory operations, matching MFMA issue rate to memory-operation completion rate. To preserve this scheduling, it pins each region with `llvm.amdgcn.sched.barrier(0)` after every memory anchor, so LLVM's machine scheduler keeps the interleave instead of clustering the MFMAs (no misched-disable needed). Without it, the backend clusters all MFMAs together, causing register spills and MFMA stalls. See [a16w16 v5 section 5](a16w16/v5_local_prefetch/README.md#5-introduction-to-the-llir-scheduler) for the motivation and [`plugins/llir_scheduler/`](../../../plugins/llir_scheduler/README.md) for the plugin itself. The scheduler:
- Classifies memory operations into GR (global read), LR (local read), and LW (local write) anchors
- Distributes MFMAs among anchors based on throughput (e.g., 4 MFMAs per global load for 16-cycle MFMA, 2 for 32-cycle)
- For MXFP4 kernels, moves scale-related LR instructions to interleave with global loads and allocates remaining MFMAs after ds_write to cover LDS port contention

**2. force-agpr — reserve AGPRs for MFMA accumulators.** A single env var `TRITON_FORCE_MFMA_AGPR=1` drives two paired effects: (a) the tutorial kernels set `llvm_fn_attrs="amdgpu-agpr-alloc=256"`, directing LLVM's register allocator to reserve 256 AGPRs for MFMA accumulators; and (b) `llvm.cc` sets `amdgpu-mfma-vgpr-form=false`, preventing LLVM from using the VGPR form of MFMA instructions. Together they keep accumulators in AGPRs and reduce VGPR pressure. This addresses the register-allocation challenges in [a16w16 v7 sections 4.3–4.4](a16w16/v7_sliceN/README.md#43-register-allocation-workaround-force-agpr). **Tradeoff**: forcing accumulators into AGPRs maximizes `v_accvgpr_read` copies in the epilogue, because `v_cvt` (used to downcast FP32 accumulators to the output dtype) requires VGPR inputs. Acceptable for compute-bound GEMM with large K (~95% time in the main loop), potentially harmful where the epilogue is a larger fraction of runtime.

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

The [a16w16/](a16w16/) directory documents a step-by-step optimization journey from a naive 541 TFLOPS baseline to a near-optimal 1421 TFLOPS implementation—a **~2.6× improvement** through 10 versions (v0–v9).

**Start here** to learn how to write high-performance Gluon kernels. Then proceed to [a8w8/](a8w8/) and [a4w4/](a4w4/) in that order.

## 4. BF8 and MXFP4: Applying the Same Design

The optimization principles from the FP16 journey apply directly to BF8 and MXFP4. The final kernel for all three data types shares the same fundamental design: M+N slicing, 3-stage pipeline, loop unrolling by 2, and the LLIR scheduler + amdgcnas optimizations.

| Aspect | FP16 (a16w16) | BF8 (a8w8) | MXFP4 (a4w4) |
|--------|---------------|------------|--------------|
| Tile size | 256x256x64 | 256x256x128 | 256x256x256 |
| MFMA instruction | `v_mfma_f32_16x16x32_f16` | `v_mfma_scale_f32_16x16x128_f8f6f4` | `v_mfma_scale_f32_16x16x128_f8f6f4` |
| cbsz / blgp | N/A | 1 / 1 (E5M2) | 4 / 4 (E2M1) |
| MFMA cycles | 16 | 32 (cbsz/blgp <= 1) | 16 (cbsz/blgp > 1) |
| Scaling | None | None | Per-group e8m0 |

The [a8w8/](a8w8/) directory provides the final optimized BF8 kernel. If you understand the FP16 journey, you will understand the BF8 kernel. The key differences are tile shape, MFMA instruction, and LDS padding.

The [a4w4/](a4w4/) directory implements the MXFP4 kernel, whose genuinely new element is the per-group scale pipeline: every 32 e2m1 elements share an 8-bit e8m0 scale that must be loaded and laid out for `mfma_scaled`. It ships in two versions — `v0_sliceN` stages scales through LDS with a `local_store` → `local_load` round-trip, while the final `v1_sliceMN` loads them straight into LDS via `buffer_load_to_lds` alongside the input tiles (no `local_store`) and uses M+N slicing for a more balanced design. See the [a4w4 README](a4w4/README.md) for full details.
