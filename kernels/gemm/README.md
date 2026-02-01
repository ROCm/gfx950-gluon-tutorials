# GEMM Kernels in Gluon (AMD GFX9)

This directory contains **high-performance GEMM kernels written in Gluon**, targeting **AMD GFX9 GPUs**.

The goal is not just to provide fast kernels, but to **teach how to design, analyze, and optimize GEMM kernels** on AMD hardware—from memory layout to instruction scheduling.

---

## What You’ll Find Here

- Compute-bound GEMM kernels for multiple data types
- Hardware-aware kernel designs built around MFMA
- A strong emphasis on **performance reasoning**, not just results
- Reusable optimization ideas that extend beyond a single data type

---

## Structure

```text
kernels/gemm/
├── a16w16/        # Full optimization journey (v0 → v7)
├── a8w8/          # Final optimized kernel (f8)
├── amxfp4wmxfp4/  # Final optimized kernel (mxfp4)
└── README.md      # You are here
```

---

## FP16: The Full Optimization Journey

The FP16 GEMM directory is the heart of this repository.
It presents a versioned progression of kernels, starting from a naive baseline and incrementally introducing:

- Buffer operations
- Async copy
- LDS design fundamentals
- Global and local prefetch pipelines
- Loop unrolling
- Kernel-level optimizations beyond the hot loop

Each version focuses on one new concept, allowing readers to understand:

- What problem is being solved
- Why the optimization works
- How it affects hardware execution

👉 Start here if you want to learn how to write high-performance Gluon kernels:
```bash
kernels/gemm/a16w16/
```

---

## 8-bit and 4-bit: Applying the Same Ideas

For 8-bit and 4-bit, this repository provides final optimized kernels only.

Why?

- The optimization principles are identical
- The main difference is tile shape and data packing
- Repeating the full journey would add volume, not insight

| Data Type | Tile Size       |
|-----------|-----------------|
| 16-bit    | 256 × 256 × 64  |
| 8-bit     | 256 × 256 × 128 |
| 4-bit     | 256 × 256 × 256 |

If you understand the FP16 journey, you understand these kernels.

---

## Philosophy

Performance does not come from a single trick.
It emerges from:

- Good layouts
- Explicit memory movement
- Latency hiding via pipelining
- Careful control of registers and LDS
- Awareness of backend codegen and scheduling limits

This repository is built around the idea that performance is a process, 
and that process should be visible.

---

## How to Use This Directory

- Learning Gluon?
  Start with a16w16/ and follow the versions in order.
- Looking for a fast kernel?
  Jump directly to the latest version for your data type.
- Interested in AMD GPU performance?
  Focus on LDS behavior, MFMA utilization, and prefetch pipelines.
  
Happy optimizing 🔧
