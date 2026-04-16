# GEMM Kernels in Gluon

This directory contains **high-performance GEMM kernels written in Gluon**, targeting **AMD MI350/355 GPUs** (gfx950).

The goal is not just to provide fast kernels, but to **teach how to design, analyze, and optimize GEMM kernels** on AMD hardware—from memory layout to instruction scheduling.

## 1. Performance Summary

Measured on MI355:

| Data Type | Shape | TFLOPS | MFMA Eff. |
|-----------|-------|--------|-----------|
| FP16 | 4096x4096x8192 | 1634 | 98% |
| BF8 | 4096x4096x16384 | 3383 | 99% |
| MXFP4 | 4096x4096x32768 | 5270 | 92% |

All kernels require the [LLIR Scheduler](https://github.com/ROCm/triton/tree/matmul_4waves) and [amdgcnas](https://github.com/ROCm/triton/tree/matmul_4waves) for optimal performance.

## 2. Prerequisites

### 2.1 Triton Branch — LLIR Scheduler and amdgcnas

The LLIR Scheduler and amdgcnas are available on the [`matmul_4waves`](https://github.com/ROCm/triton/tree/matmul_4waves) development branch. Build Triton from this branch to use these features. Both passes are essential for all three kernels (a16w16, a8w8, a4w4).

**Why these tools exist.** Upstream LLVM's scheduling and register-allocation passes were designed for the discovery model: they receive thread-level IR, recover dependencies by analysis, and solve the resulting NP-hard problems with heuristics. Gluon's block-level programming model makes those problems smaller — dependencies are *engineered* at the block level (e.g., `DOT`, `local_load`, and `buffer_load` are designed to be independent within a 3-stage pipeline), so at the instruction level, MFMAs, `ds_read`s, and `buffer_load`s can be interleaved by a simple throughput-model pass. Likewise, register budgets have a closed-form expression at block level, so allocation becomes a matter of honoring that budget rather than solving graph coloring.

`llirSched` and `amdgcnas` are the minimum tools that honor this block-level contract today. They are not general-purpose replacements for LLVM's `misched` or register allocator — on arbitrary C-like code they would not make sense. On Gluon-shaped kernels they recover the MFMA efficiency the upstream LLVM flow loses, and their underlying ideas are being integrated into LLVM itself in collaboration with LLVM engineers, so that upstream LLVM will eventually produce the same output. See [`docs/performance_philosophy.md`](../../docs/performance_philosophy.md) for the full argument.

**LLIR Scheduler** (`TRITON_ENABLE_LLIR_SCHED=1`) is a Triton LLIR-level pass that interleaves MFMA instructions with memory operations (global loads, LDS reads/writes, async copies) based on the **throughput model** of those memory operations, matching MFMA issue rate to memory operation completion rate. To preserve this scheduling, it disables the LLVM backend's pre-RA and post-RA machine schedulers. Without the LLIR scheduler, the backend compiler clusters all MFMAs together, causing register spills and MFMA stalls. See [a16w16 v5 section 5](a16w16/v5_local_prefetch/README.md#5-introduction-to-the-llir-scheduler) for the motivation. The scheduler:
- Classifies memory operations into GR (global read), LR (local read), and LW (local write) anchors
- Distributes MFMAs among anchors based on throughput (e.g., 4 MFMAs per global load for 16-cycle MFMA, 2 for 32-cycle)
- For MXFP4 kernels, moves scale-related LR instructions to interleave with global loads and allocates remaining MFMAs after ds_write to cover LDS port contention

**amdgcnas** (`TRITON_ENABLE_AMDGCN_AS=1`) addresses the register allocation challenges described in [a16w16 v7 sections 4.3–4.4](a16w16/v7_sliceN/README.md#43-register-allocation-workaround). It does two things:

1. **LLVM register hints**: Sets `amdgpu-agpr-alloc=256` on the kernel function, directing LLVM's register allocator to reserve 256 AGPRs for MFMA accumulators. Also sets `amdgpu-mfma-vgpr-form=false` to prevent LLVM from using the VGPR form of MFMA instructions, keeping accumulators in AGPRs and reducing VGPR pressure.

2. **Post-assembly processing**: Optimizes the final generated assembly:
   - **LICM (Loop Invariant Code Motion)**: Hoists loop-invariant instructions (e.g., LDS address calculations) to the loop prologue. When the hoisted instruction's output register is redefined inside the loop, it applies register renaming.
   - **Peephole optimizations**: Interleaves MFMA with scalar instructions (`s_waitcnt`, `s_barrier`, scalar address computation for buffer loads) to maintain continuous MFMA throughput.

### 2.2 Running Benchmarks

The easiest way to run benchmarks with all optimizations enabled is `run_perf_table.py`:

```bash
# FP16 (a16w16)
python scripts/run_perf_table.py --kernel a16w16 --versions 8 --configs llir+amdgcnas --K 8192 --dtype fp16 --use-rocprof

# BF8 (a8w8)
python scripts/run_perf_table.py --kernel a8w8 --configs llir+amdgcnas --K 16384 --use-rocprof

# MXFP4 (a4w4)
python scripts/run_perf_table.py --kernel a4w4 --configs llir+amdgcnas --K 32768 --use-rocprof
```

This script automatically:
- Sets the environment variables for llirSched and amdgcnas
- Collects kernel traces using rocprofv3
- Calculates and reports TFLOPS, VGPRs, spills, and MFMA efficiency

### 2.3 Manual Workflow

To run benchmarks manually, set the environment variables directly. Run from the kernel directory:

```bash
# FP16 (from kernels/gemm/a16w16/)
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 python bench.py --version 8 --K 8192 --dtype fp16

# BF8 (from kernels/gemm/a8w8/)
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 python bench.py --K 16384

# MXFP4 (from kernels/gemm/a4w4/)
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 python bench.py --K 32768
```

For accurate performance measurement, the `--rocprof` flag runs the kernel 1000 times with rotating buffers but does not print performance numbers. To collect measurements:

1. Collect the kernel trace (`-d` specifies the output directory):
   ```bash
   TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
       rocprofv3 --kernel-trace -d out -- python bench.py --version 8 --K 8192 --dtype fp16 --rocprof
   ```

2. Calculate kernel time from the trace. The CSV file may be in a nested directory under the output directory—locate it first. Output is in microseconds by default:
   ```bash
   python ../../../scripts/calc_kernel_time.py [trace_csv_file] [kernel_name]
   ```

3. Convert to TFLOPS: `TFLOPS = 2 × M × N × K / (time_in_us × 10^6)`

## 3. FP16: The Optimization Journey

The [a16w16/](a16w16/) directory documents a step-by-step optimization journey from a naive 524 TFLOPS baseline to a near-optimal 1634 TFLOPS implementation—a **3× improvement** through 10 versions (v0–v9).

**Start here** to learn how to write high-performance Gluon kernels. Then proceed to [a8w8/](a8w8/) and [a4w4/](a4w4/) in that order.

## 4. BF8 and MXFP4: Applying the Same Design

The optimization principles from the FP16 journey apply directly to BF8 and MXFP4. All three kernels share the same fundamental design: N-slicing, 3-stage pipeline, loop unrolling by 2, and the LLIR scheduler + amdgcnas optimizations.

| Aspect | FP16 (a16w16) | BF8 (a8w8) | MXFP4 (a4w4) |
|--------|---------------|------------|--------------|
| Tile size | 256x256x64 | 256x256x128 | 256x256x256 |
| MFMA instruction | `v_mfma_f32_16x16x32_f16` | `v_mfma_scale_f32_16x16x128_f8f6f4` | same |
| cbsz / blgp | N/A | 1 / 1 (E5M2) | 4 / 4 (E2M1) |
| MFMA cycles | 16 | 32 (cbsz/blgp <= 1) | 16 (cbsz/blgp > 1) |
| Scaling | None | None | Per-group e8m0 |

The [a8w8/](a8w8/) directory provides the final optimized BF8 kernel. If you understand the FP16 journey, you will understand the BF8 kernel. The key differences are tile shape, MFMA instruction, and LDS padding.

The [a4w4/](a4w4/) directory implements the MXFP4 kernel, which introduces new challenges: per-group scale loading (GR → LW → LR round-trip), LDS port contention between ds_write and buffer_load_to_lds, and scale layout conversion. See the [a4w4 README](a4w4/README.md) for full details.

