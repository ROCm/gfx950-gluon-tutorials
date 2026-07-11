# Scheduling Models for High-Performance GEMM

## Why another document?

Most GEMM tutorials focus on **tiling**, **memory hierarchy**, and **data movement**. Those techniques are essential, but they answer only half of the performance question.

Once the memory hierarchy is optimized, another question dominates:

> **How should MFMA instructions and memory operations be scheduled?**

Modern GPUs can execute memory operations and tensor instructions concurrently. Achieving peak throughput therefore depends not only on *what* instructions are generated, but also on *how* they are scheduled.

This repository explores two different scheduling models that both achieve near-peak MFMA utilization on AMD gfx950 GPUs.

* **Intra-wave scheduling**
* **Inter-wave scheduling**

Both reach similar performance, but they place the scheduling responsibility in different parts of the software stack.

---

# The scheduling problem

A GEMM main loop repeatedly performs three kinds of work:

* Load tiles from global memory
* Load tiles from LDS
* Execute MFMA instructions

Naively executing these operations sequentially leaves hardware idle.

```
Load
↓

Load
↓

MFMA
↓

Load
↓

Load
↓

MFMA
```

The goal is to overlap these operations so that memory latency is hidden behind MFMA execution.

There are two fundamentally different ways to achieve this overlap.

---

# Model 1 — Intra-wave scheduling

The first model keeps **one wave resident on each SIMD**.

Within that wave, memory instructions and MFMA instructions are carefully interleaved.

```
Wave 0

buffer_load
MFMA
ds_read
MFMA
buffer_load
MFMA
...
```

The overlap happens **inside one wave**.

The compiler is responsible for constructing the instruction schedule.

To make this possible, the kernel is written so that neighboring tensor operations are intentionally independent. This gives the scheduler freedom to interleave individual instructions while preserving correctness.

Today, the repository uses three components to realize this model:

* llirSched
* force-agpr
* amdgcnas

These are not arbitrary optimizations.

They are the minimum infrastructure needed to preserve the scheduling intent expressed by the Gluon kernel.

---

# Model 2 — Inter-wave scheduling

The second model takes a different approach.

Instead of interleaving instructions within a wave, it launches **two waves on every SIMD**.

Each wave executes different regions of the pipeline.

```
Wave A

MFMA region
MFMA region
MFMA region

Wave B

Memory region
Memory region
Memory region
```

While one wave issues MFMA instructions, the other wave issues memory instructions.

The hardware naturally overlaps the execution of the two waves.

Instead of scheduling individual instructions, the kernel schedules **pipeline stages**.

The compiler no longer needs to discover a fine-grained instruction interleaving.

---

# Two different locations for scheduling intelligence

The two models differ primarily in where the scheduling intelligence lives.

|                           | Intra-wave              | Inter-wave         |
| ------------------------- | ----------------------- | ------------------ |
| Scheduling unit           | Individual instructions | Pipeline stages    |
| Execution                 | One wave per SIMD       | Two waves per SIMD |
| Scheduling responsibility | Compiler                | Kernel             |
| Compiler involvement      | High                    | Low                |
| LLVM dependence           | Higher                  | Lower              |
| Hardware overlap          | Instruction-level       | Wave-level         |

Neither model is universally better.

They represent different system design choices.

---

# Tradeoffs

## Compiler involvement

Intra-wave scheduling relies on compiler infrastructure to construct an efficient instruction schedule.

Current LLVM scheduling heuristics are designed for general-purpose programs and do not fully exploit the structured dependencies already present in Gluon kernels.

The llirSched plugin bridges this gap.

Inter-wave scheduling largely avoids this problem by expressing the overlap directly in the kernel structure.

---

## Scheduling granularity

Intra-wave scheduling optimizes individual instructions.

Inter-wave scheduling optimizes larger execution regions.

This is similar to the difference between instruction scheduling and task scheduling.

---

## Hardware evolution

Inter-wave scheduling resembles an asynchronous execution model.

The kernel exposes separate MFMA stages and memory stages, allowing hardware to overlap them naturally.

This direction appears compatible with the increasing use of asynchronous execution models on modern GPUs.

---

## LDS bandwidth

Inter-wave scheduling requires two resident waves per SIMD.

Because multiple waves read the same shared tiles, this increases pressure on LDS bandwidth.

Intra-wave scheduling performs less redundant LDS traffic and therefore uses LDS bandwidth more efficiently.

Depending on the kernel, this can offset some of the benefits of reduced compiler complexity.

---

## Compiler complexity

Intra-wave scheduling asks the compiler to solve a difficult instruction scheduling problem.

Inter-wave scheduling shifts that complexity into the kernel design.

The compiler becomes simpler, while the kernel becomes more structured.

---

# Which model should I choose?

If your goal is to understand GPU instruction scheduling and compiler optimization, start with the **intra-wave** kernels.

They expose every important optimization problem:

* instruction scheduling
* register allocation
* latency hiding
* instruction-level throughput

If your goal is to build production kernels using upstream Triton and LLVM, the **inter-wave** kernels provide a simpler implementation strategy with comparable performance.

---

# This repository

This repository intentionally contains both scheduling models.

The goal is not to declare one universally superior.

Instead, it demonstrates that high-performance GEMM can be achieved through two different system designs:

* one that places scheduling intelligence in the compiler,
* and one that places scheduling intelligence in the kernel itself.

Understanding both provides a broader perspective on GPU kernel design than studying either model alone.
