# FP16 GEMM Kernel Optimization on AMD GFX9 (Gluon)

This directory presents a **step-by-step optimization journey of an FP16 GEMM kernel written in Gluon**, targeting **AMD MI350/355 GPUs** (gfx950).

Rather than showing a single "final" kernel, this repository documents **how high performance is achieved**—from a naive baseline to a near-optimal design—covering **memory movement, layout design, latency hiding, and instruction scheduling** along the way.

If you are familiar with Triton, think of this as:

> [!IMPORTANT]
> Learning Gluon by learning layouts, pipelines, and hardware behavior.

## 1. Directory Structure

```
a16w16/
├── bench.py              # Benchmark and correctness test
├── images/               # Layout visualizations and trace screenshots
├── v0_naive/             # Baseline kernel with explicit layouts
├── v1_buffer_load/       # Buffer operations for masked loads
├── v2_async_copy/        # Direct-to-LDS async copy
├── v3_lds/               # LDS layout design and evaluation
├── v4_global_prefetch/   # 2-stage pipeline with double buffering
├── v5_local_prefetch/    # 3-stage pipeline with local prefetch
├── v6_loop_unroll/       # Loop unrolling to eliminate copy overhead
├── v7_slice/             # N-slicing for register pressure reduction
└── v8_beyond_hotloop/    # L2 cache locality optimization
```

## 2. How to Run

From the `a16w16` directory:

```bash
python bench.py --version 8 --K 8192 --dtype fp16 --use-rocprof
```

This runs correctness checks against torch.matmul and reports TFLOPS. Use `--version` to select a kernel version (0-8) and `--use-rocprof` for accurate performance measurement.

## 3. Optimization Philosophy

Writing a Gluon kernel is only the starting point. Real performance comes from:

- **Codegen quality** (instruction selection, register pressure)
- **Latency hiding** (overlapping memory and compute via pipelining)
- **Instruction scheduling** (MFMA utilization inside the hot loop)
- **Power efficiency** (L2 cache locality, stable power draw)

Every intermediate kernel version is kept intentionally, so readers can see *what changed*, *why it matters*, and *how it affects hardware execution*.

> [!NOTE]
> A key theme throughout: **think at the block level, not the instruction level**. Gluon kernels are designed at tensor granularity; fine-grained scheduling belongs in the backend.

## 4. Kernel Versions

Each version introduces **one new idea** and builds on the previous one.

| Version | Name | Focus | Key Concept |
|---------|------|-------|-------------|
| v0 | naive | Baseline | Explicit layouts, correctness-first MFMA kernel |
| v1 | buffer_load | Codegen | Hardware OOB checking, branch elimination |
| v2 | async_copy | Codegen | Direct-to-LDS, eliminates register staging |
| v3 | lds | Codegen | LDS layout design: swizzling vs padding |
| v4 | global_prefetch | Latency hiding | 2-stage pipeline, double buffering |
| v5 | local_prefetch | Latency hiding | 3-stage pipeline, LLIR scheduler introduction |
| v6 | loop_unroll | Codegen | Eliminate copy overhead, DIDT/PIT analysis |
| v7 | slice | Register pressure | N-slicing, register allocation workarounds |
| v8 | beyond_hotloop | Power efficiency | XCD-aware PID remapping, GROUP_SIZE_M optimization |

## 5. Performance Results

Measured on MI355 with shape 4096×4096×8192, FP16:

![Performance Chart](images/performance_chart.png)

| Version | TFLOPS | MFMA Eff. | Notes                      |
|---------|--------|-----------|----------------------------|
| v0      |    524 |       25% | Baseline                   |
| v1      |    514 |       24% |                            |
| v2      |    697 |       36% |                            |
| v3      |    774 |       42% |                            |
| v4      |   1113 |       57% |                            |
| v5      |   1134 |       59% |                            |
| v5      |   1283 |       76% | + llirSched                |
| v6      |   1088 |       61% |                            |
| v6      |   1260 |       88% | + llirSched                |
| v7      |   1279 |       65% |                            |
| v7      |   1411 |       79% | + llirSched                |
| v7      |   1538 |       98% | + llirSched + amdgcnas     |
| v8      |   1336 |       67% |                            |
| v8      |   1470 |       77% | + llirSched                |
| v8      |   1634 |       98% | + llirSched + amdgcnas     |

Performance is measured and explained using:

- Microbenchmarking for throughput
- `rocprofv3` traces for cycle-level analysis
- Hardware counters for L2 cache behavior
- A custom trace tool to compute **MFMA efficiency**

For methodology details, see [MFMA Efficiency](../../../docs/mfma_efficiency.md).

## 6. Tools and Infrastructure

This tutorial relies on several tools:

- **LLIR Scheduler**: Instruction-level scheduling at LLVM IR level (`TRITON_ENABLE_LLIR_SCHED=1`)
- **amdgcnas**: Assembly post-processor for peephole optimizations (`TRITON_ENABLE_AMDGCN_AS=1`)
- **Layout plotting tool**: Visualize blocked, MFMA, and LDS layouts
- **run_perf_table.py**: Automated performance collection across versions
- **run_counter_collection.py**: Hardware counter collection for cache analysis

See [scripts/README.md](../../../scripts/README.md) for usage details.

## 7. Beyond FP16

Although this directory focuses on **FP16 compute-bound GEMM**, the same optimization strategy applies to other precisions:

| Data Type | Tile Size | Compute Intensity |
|-----------|-----------|-------------------|
| FP16      | 256×256×64  | Compute-bound |
| MXFP4     | 256×256×256 | Compute-bound |

The optimization journey remains the same—only the tile shape and MFMA instruction variant change.

## 8. How to Read This

Recommended order:

1. Start with `v0_naive` to understand the baseline and explicit layouts
2. Progress version by version, reading both code and README
3. Use thread traces and layout visualizations to build intuition
4. Pay attention to bottleneck analysis sections—they motivate the next version

If you only want the fastest kernel, jump to v8 with `llirSched + amdgcnas`. If you want to understand **why** it is fast, start from the beginning.

> [!TIP]
> Each README follows a consistent structure: Motivation → Design → Performance Analysis → What Comes Next. This progression builds understanding incrementally.
