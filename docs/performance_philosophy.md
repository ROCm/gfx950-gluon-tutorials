# Performance Philosophy

This document explains the design philosophy behind this repository and how it is meant to be used.

The goal is not just to show how to write gluon kernels, but to teach **how to reason about performance**
on AMD GFX9 GPUs.

---

## Writing a Kernel Is Only Step One

A gluon kernel that produces correct results is only the starting point.

High performance requires careful consideration of:
- Data and thread layouts
- Instruction scheduling and latency hiding
- Register usage and occupancy
- Hardware-specific execution behavior (e.g., MFMA)

This repository treats kernel correctness as a prerequisite—not an achievement.

---

## Optimization Is a Journey

In real kernel development, performance does not appear all at once.

Instead, it emerges through a repeated process:
1. Write a baseline kernel
2. Measure its performance
3. Identify bottlenecks
4. Form a hypothesis
5. Apply a targeted optimization
6. Measure again

Each kernel in this repository is developed through multiple versions that reflect this process.
Older versions are preserved intentionally—they document *why* later versions exist.

The goal is not to present a final answer, but to expose the reasoning path.

---

## Layout Is the Language of Gluon

One of the biggest differences between gluon and Triton is the central role of layouts.

In gluon:
- Layouts define how threads map to data
- Layouts determine how MFMA instructions consume operands
- Layout choices directly affect both correctness and performance

For developers coming from Triton, learning gluon largely means learning how to think in terms of layouts.

Because layouts are difficult to reason about from code alone, this repository includes a layout
visualization tool to make these mappings explicit and observable.

---

## Measuring Performance at the Right Level

Simple benchmarks (e.g., average runtime or TFLOPS) are useful, but they do not explain *why* a kernel
performs the way it does.

High-performance kernels depend on:
- Instruction overlap
- Latency hiding
- Pipeline utilization

To understand these effects, this repository emphasizes:
- Collecting instruction traces with `rocprofv3`
- Visualizing execution timelines
- Analyzing cycle-level behavior inside loops

In particular, we focus on **MFMA efficiency** as a way to quantify how effectively compute resources
are utilized during kernel execution.

---

## Tooling Is Part of Kernel Design

Profiling tools and visualization tools are not treated as optional add-ons.

They are an integral part of the kernel development process:
- Layout plots inform layout design
- Instruction traces validate scheduling decisions
- MFMA efficiency metrics guide further optimization

A kernel that cannot be measured is a kernel that cannot be optimized reliably.

---

## Advanced Features Require Advanced Validation

As kernel versions become more sophisticated:
- Optimizations interact in non-obvious ways
- Bottlenecks shift from memory to compute to scheduling
- Simple metrics become insufficient

Accordingly, each new class of optimization introduced in this repository is paired with
more advanced analysis techniques to validate its effect.

---

## How to Use This Repository

This repository is best approached deliberately:
- Read the documentation before the code
- Follow kernel versions in order
- Study profiling data alongside kernel changes
- Use visualization tools to build intuition

The intent is not speed, but understanding.

---

## Final Thoughts

Performance engineering is not about memorizing tricks.
It is about developing the ability to observe, reason, and validate.

This repository is an attempt to make that process visible.
