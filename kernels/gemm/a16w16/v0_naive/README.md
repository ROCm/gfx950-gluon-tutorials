# v0_naive — Correctness First

This is the very first GEMM kernel in the FP16 optimization journey.

At this stage, the goal is not performance. The goal is to produce a **correct MFMA-based GEMM kernel** and to make all Gluon-specific concepts explicit. This version serves as a reference point for everything that follows.

If you are new to Gluon, this kernel answers a simple question:

> What does a correct GEMM kernel look like when nothing is hidden?

---

## What This Kernel Establishes

The kernel implements a compute-bound FP16 GEMM using MFMA and produces correct results for all supported problem sizes. It intentionally avoids prefetching, pipelining, or latency hiding. As a result, performance is expected to be poor.

That is by design.

This version exists so that future optimizations have a solid and understandable baseline.

---

## How This Differs from the Triton Tutorial

Structurally, this kernel is very close to the standard Triton GEMM tutorial. The important differences are not in control flow, but in **how data is described and moved**.

All data movement in this kernel uses **explicit layouts**. Global loads for operands A and B specify their layouts directly. The dot operation uses explicitly defined layouts for both operands and for the result tensor. Whenever the required layout changes, an explicit `convert_layout` operation is inserted.

In addition, the dot operation is written using `gl.amd.cdna3.mfma()` directly, rather than relying on a higher-level abstraction.

With the exception of this explicit MFMA usage, all differences from Triton come down to one idea:

> Learning Gluon kernels is learning layouts.

This kernel makes that idea concrete.

---

## About Layouts

This README does not attempt to explain the layout definitions in detail.

If you are coming from Triton, the best introduction to bespoke layouts is Lei’s blog post:

https://www.lei.chat/posts/triton-bespoke-layouts/

That post explains the *what* and the *why*. What this tutorial adds is a way to **see layouts directly** and connect them to MFMA execution.

---

## Visualizing Layouts with `layout_plot`

To make layouts tangible, this repository includes a layout plotting tool. Instead of reasoning about layouts abstractly, you can render them and inspect how work is distributed across threads, warps, and MFMA instructions.

For this kernel, the following commands were used to generate the layout visualizations.

### Operand A (Blocked Layout)

```bash
python3 plot_layout.py blocked \
  -b 256 64 \
  --rowName BM \
  --colName BK \
  --sizePerThread 1 8 \
  --threadsPerWarp 8 8 \
  --warpsPerCTA 4 1 \
  --order 1 0 \
  --output v0_naive_blocked-layout-A
```

### Operand B (Blocked Layout)

```bash
python3 plot_layout.py blocked \
  -b 64 256 \
  --rowName BK \
  --colName BN \
  --sizePerThread 8 1 \
  --threadsPerWarp 8 8 \
  --warpsPerCTA 1 4 \
  --order 0 1 \
  --output v0_naive_blocked-layout-B
```

### Dot Operation Layout (MFMA)

```bash
python3 plot_layout.py dot \
  --dotShape 256 256 64 \
  --warpsPerCTA 2 2 \
  --dtypeA fp16 \
  --dtypeB fp16 \
  --kWidth 8 \
  --mfmaTrans \
  --output v0_naive_dot
```

This final visualization is particularly important. 
It shows the layouts of both dot operands, the resulting MFMA layout, and a zoomed-in view of a single MFMA instruction. 
Together, these views reveal how operand layouts map onto MFMA execution and how results are produced.

The generated images can be found at [../images](../images).

## Why the Layout Plot Tool Matters

The layout plot tool is more than a convenience utility. It reflects a core idea behind Gluon:

> Layouts are not static properties of individual tensors.
> They are part of the lowering process and define how operations connect.

Understanding the relationship between operand layouts and result layouts is essential for writing efficient Gluon kernels. 
This tool makes those relationships visible.

## Performance (Intentionally Ignored)

In this version, performance is not analyzed. The benchmark script only checks correctness. There is no codegen inspection, no profiling, and no attempt to hide latency.

That changes in the next version.


## What Comes Next

In `v1_buffer_load`, we will start looking at the generated code. 
A specific codegen issue will be identified and fixed using buffer operations. 
This marks the beginning of performance-driven kernel development.

The optimization journey starts there.

Stay tuned 🚀
