# FP16 GEMM Kernel Optimization on AMD GFX9 (Gluon)

This directory presents a **step-by-step optimization journey of an FP16 GEMM kernel written in Gluon**, targeting **AMD GFX9 GPUs**.

Rather than showing a single “final” kernel, this repository documents **how high performance is achieved**—from a naive baseline to a near-optimal design—covering **memory movement, layout design, latency hiding, and instruction scheduling** along the way.

If you are familiar with Triton, think of this as:
> **Learning Gluon by learning layouts, pipelines, and hardware behavior.**

---

## What This Repository Is

- A **progressive sequence of GEMM kernels** (`v0` → `v7`)
- Each version introduces **one core optimization concept**
- A focus on **analysis-driven performance engineering**
- Deep coverage of AMD-specific features:
  - MFMA
  - LDS
  - Buffer operations
  - Async copy
  - Software pipelining

This is a **learning-oriented** repository, not a black-box kernel drop.

---

## Optimization Philosophy

Writing a Gluon kernel is only the starting point.

Real performance comes from:
- **Codegen quality** (instruction count, register pressure)
- **Latency hiding** (overlapping memory and compute)
- **Instruction scheduling** (MFMA utilization inside the hot loop)
- **Kernel-level effects** (epilogues, cache locality, PID mapping)

Every intermediate kernel version is kept intentionally, so readers can see:
- *What changed*
- *Why it matters*
- *How it affects hardware execution*

---

## Kernel Versions

Each version introduces **one new idea** and builds on the previous one.

### v0_naive — Baseline
- Global loads only
- No prefetching
- No latency hiding

Establishes baseline behavior and exposes raw memory latency.

---

### v1_buffer_load — Buffer Operations
- Replace `global_load` with `buffer_load`
- Introduces AMD buffer ops
- Discusses benefits and limitations

Focus: **better codegen and memory access semantics**.

---

### v2_async_copy — Global → LDS
- Use async copy to load directly into LDS
- Eliminates the register → LDS copy path
- Requires explicit LDS control

Focus: **instruction reduction and dataflow simplification**.

---

### v3_LDS — LDS Performance Fundamentals
- Deep dive into LDS behavior:
  - Vectorization
  - Addressing
  - Issue vs execution latency
- Supported by a standalone LDS document

Focus: **writing LDS-efficient kernels**.

---

### v4_global_prefetch — 2-Stage Pipeline
- Introduces global data prefetch
- Software pipelining to hide global memory latency
- Discusses overlap and resource trade-offs

Focus: **latency hiding via global prefetch**.

---

### v5_local_prefetch — 3-Stage Pipeline
- Adds LDS prefetch on top of global prefetch
- Partial prefetch to control register pressure
- Introduces op-level scheduling

Focus: **hiding LDS latency and improving MFMA utilization**.

---

### v6_loop_unroll — Hot Loop Finalization
- Unroll the K loop
- Remove register copy overhead
- Reduce instruction count inside the hot loop

At this point, the **hot loop design is considered optimal at the Gluon level**.

---

### v7_beyond_hotloop — Kernel-Level Optimization
Once the hot loop is optimized, remaining bottlenecks lie elsewhere:
- Epilogue optimization
- PID remapping based on XCD configuration
- L2 locality improvements
- Backend-level instruction scheduling

Focus: **performance beyond what Gluon directly exposes**.

---

## Codegen vs Latency

| Version | Primary Focus |
|------|---------------|
| v0–v2 | Instruction reduction (codegen) |
| v3 | LDS efficiency (codegen) |
| v4–v5 | Prefetch & overlap (latency hiding) |
| v6 | Loop cleanup (codegen) |
| v7 | Kernel-level effects |

---

## Performance Analysis

Performance is always measured and explained using:
- Microbenchmarking for throughput
- `rocprofv3` traces for cycle-level analysis
- A custom trace tool to compute **MFMA efficiency**

Reference slides and talks are linked where deeper background is helpful.

---

## Beyond FP16

Although this directory focuses on **FP16 compute-bound GEMM**, the same strategy applies to lower precision:

| Data Type | Tile Size       |
|-----------|-----------------|
| 16-bit    | 256 × 256 × 64  |
| 8-bit     | 256 × 256 × 128 |
| 4-bit     | 256 × 256 × 256 |

The optimization journey remains the same—only the tile shape changes.

---

## How to Read This

Recommended order:
1. Start with `v0_naive`
2. Progress version by version
3. Read code and accompanying explanations together
4. Use traces and layout visualizations when available

If you only want the fastest kernel, jump to the last version.
If you want to understand **why** it is fast, start from the beginning.

---

Happy hacking 🚀
