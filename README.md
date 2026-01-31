# High-Performance Gluon Kernels on AMD GFX9

A hands-on tutorial for writing **high-performance gluon kernels on AMD GFX9 GPUs**.

This repository goes beyond showing kernels. It teaches the **full performance engineering workflow**:
writing kernels, analyzing bottlenecks, reasoning about layouts, and validating optimizations at the
cycle level.

---

## What Makes This Repo Different

- Kernels are developed **incrementally**, from naive to optimized
- Every optimization is motivated by a **measured bottleneck**
- Old versions are kept to document the **optimization journey**
- Layouts are treated as **first-class concepts**, not implementation details
- Performance is analyzed using **instruction traces**, not just throughput numbers

This is especially useful for **Triton developers learning gluon**, where understanding layouts
and MFMA execution is essential.

---

## What You’ll Find in This Repo

```
.
├── docs/ # Concepts, mental models, and background
├── kernels/ # Step-by-step kernel implementations
├── layout_plot/ # Layout visualization tools
├── profiling/ # Performance measurement and analysis workflows
└── scripts/ # Helper scripts for building and running examples
```


## Who This Is For

- Gluon users aiming for **high performance**
- Triton developers learning gluon
- AMD GPU performance and compiler engineers

---

## Getting Started

Start with:
- `docs/getting_started.md`
- `docs/performance_philosophy.md`

Then pick a kernel and follow it version by version.

---

Performance engineering is not magic.
It’s a process. This repo tries to make that process explicit.
