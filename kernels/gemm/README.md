# GEMM Kernels in Gluon

This directory contains **high-performance GEMM kernels written in Gluon**, targeting **AMD MI350/355 GPUs** (gfx950).

The goal is not just to provide fast kernels, but to **teach how to design, analyze, and optimize GEMM kernels** on AMD hardware—from memory layout to instruction scheduling.

## 1. Performance Summary

Measured on MI355:

| Data Type | Shape | TFLOPS | MFMA Eff. |
|-----------|-------|--------|-----------|
| FP16 | 4096×4096×8192 | 1634 | 98% |
| FP8 | 4096×4096×16384 | 3383 | 99% |

Both kernels require the [LLIR Scheduler](https://github.com/ROCm/triton/tree/matmul_4waves) and [amdgcnas](https://github.com/ROCm/triton/tree/matmul_4waves) for optimal performance.

## 2. Prerequisites

### 2.1 Triton Branch

The LLIR Scheduler and amdgcnas are available on the [`matmul_4waves`](https://github.com/ROCm/triton/tree/matmul_4waves) development branch. Build Triton from this branch to use these features.

### 2.2 Running Benchmarks

The easiest way to run benchmarks with all optimizations enabled is `run_perf_table.py`:

```bash
# FP16 (a16w16)
python scripts/run_perf_table.py --kernel a16w16 --versions 8 --configs llir+amdgcnas --K 8192 --dtype fp16 --use-rocprof

# FP8 (a8w8)
python scripts/run_perf_table.py --kernel a8w8 --configs llir+amdgcnas --K 16384 --use-rocprof
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

# FP8 (from kernels/gemm/a8w8/)
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 python bench.py --K 16384
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

The [a16w16/](a16w16/) directory is the heart of this repository. It documents a step-by-step optimization journey from a naive 524 TFLOPS baseline to a near-optimal 1634 TFLOPS implementation—a **3× improvement**.

The journey progresses through 9 versions:

| Version | Focus | Key Concept |
|---------|-------|-------------|
| v0 | Baseline | Explicit layouts, correctness-first |
| v1 | Codegen | Hardware OOB checking, branch elimination |
| v2 | Codegen | Direct-to-LDS async copy |
| v3 | Codegen | LDS layout design (swizzling vs. padding) |
| v4 | Latency hiding | 2-stage pipeline, double buffering |
| v5 | Latency hiding | 3-stage pipeline, LLIR Scheduler |
| v6 | Codegen | Loop unrolling, DIDT/PIT analysis |
| v7 | Register pressure | N-slicing, amdgcnas post-processor |
| v8 | Power efficiency | XCD-aware PID remapping, GROUP_SIZE_M |

Each version focuses on one concept, with detailed analysis of what changed, why it matters, and how it affects hardware execution.

**Start here** to learn how to write high-performance Gluon kernels.

## 4. FP8: Applying the Same Ideas

The [a8w8/](a8w8/) directory provides only the final optimized kernel.

Why? The optimization principles are identical to FP16. The main differences are tile shape, MFMA instruction, and LDS padding:

| Aspect | FP16 | FP8 |
|--------|------|-----|
| Tile size | 256×256×64 | 256×256×128 |
| MFMA instruction | mfma_f16_16x16x32 | mfma_f8_16x16x128 |
| LDS padding | [[512, 16]] | [[1024, 16], [2048, 32]] |

If you understand the FP16 journey, you understand the FP8 kernel.

## 5. Philosophy

Performance emerges from the combination of:

- **Good layouts** — explicit memory organization for conflict-free access
- **Latency hiding** — pipelining to overlap memory and compute
- **Register management** — slicing and allocation strategies
- **Power efficiency** — L2 locality and stable power draw

This repository is built around the idea that **performance is a process**, and that process should be visible.

## 6. How to Use This Directory

| Goal | Recommendation |
|------|----------------|
| Learning Gluon | Start with [a16w16/](a16w16/) and follow versions in order |
| Need a fast kernel | Jump to the latest version for your data type |
| Understanding AMD GPU performance | Focus on LDS behavior, MFMA utilization, and prefetch pipelines |
