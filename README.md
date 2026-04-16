# High-Performance Gluon Kernels on AMD GFX9

A hands-on tutorial for writing **high-performance Gluon kernels** on AMD **MI350 / MI355** GPUs (gfx950).

This repo is not a collection of finished kernels — it's a record of **how those kernels were built**. Each version in each series is driven by a measured bottleneck in the previous one, and the old versions are kept so the optimization journey is visible, not just the destination.

The headline result: on `a16w16` (FP16, 4096×4096×8192), the naive baseline runs at **524 TFLOPS**. Nine versions later, it reaches **1634 TFLOPS** — a **3× speedup**, every step motivated by a thread trace or a hardware counter.

---

## Start Here

If this is your first time in the repo, open **[`kernels/gemm/a16w16/README.md`](kernels/gemm/a16w16/README.md)** and follow the Acts I–IV narrative from `v0_naive` to `v9_beyond_hotloop`. Each version isolates one concept — a layout, a pipeline stage, a scheduling decision. Read it alongside the code, then run the kernel:

```bash
cd kernels/gemm/a16w16
python bench.py --version 0 --K 8192 --dtype fp16 --use-rocprof
```

Once you're comfortable there, the FP8 and MXFP4 kernels (`a8w8`, `a4w4`) show how the same design transfers to 8-bit and 4-bit compute with microscaling.

> [!TIP]
> Every version README follows the same shape: **Motivation → Design → Performance Analysis → What Comes Next.** If you only want the fastest FP16 kernel, jump to `v9_beyond_hotloop`. If you want to understand *why* it's fast, start from `v0_naive`.

---

## Repository Layout

```
.
├── kernels/              # The tutorial — step-by-step kernel implementations
│   └── gemm/
│       ├── a16w16/       # FP16 GEMM, v0 → v9 (start here)
│       ├── a8w8/         # BF8 GEMM — same design, adapted for 8-bit
│       └── a4w4/         # MXFP4 GEMM with per-group microscaling
├── docs/                 # Mental models: LDS throughput, memory bandwidth, MFMA efficiency
├── layout_plot/          # LaTeX-based layout visualization (blocked, dot, LDS)
├── scripts/              # Benchmarks, rocprof + ATT automation, counter collection, perf tables
├── profiling/            # Shared helpers used by the profiling scripts
├── experiments/          # Standalone validations referenced from kernel READMEs
├── ROADMAP.md            # What's done, in progress, and planned
└── CHANGELOG.md          # Changes driven by upstream Triton / compiler evolution
```

### Where to go for what

| If you want to…                                          | Look in                     |
|----------------------------------------------------------|-----------------------------|
| Learn the full optimization workflow end to end          | `kernels/gemm/a16w16/`      |
| Apply the same design to FP8                             | `kernels/gemm/a8w8/`        |
| Understand microscaling (MXFP4) and scale pipelines      | `kernels/gemm/a4w4/`        |
| Visualize a blocked / dot operand / LDS layout as a PDF  | `layout_plot/`              |
| Build a mental model for LDS or HBM throughput           | `docs/`                     |
| Automate rocprof, collect counters, generate perf tables | `scripts/`                  |
| See what changed with a Triton or compiler bump          | `CHANGELOG.md`              |
| Know what's planned next                                 | `ROADMAP.md`                |

---

## The Tutorial Progression

| Kernel    | Data Type | Tile Size    | What's New                                                        |
|-----------|-----------|--------------|-------------------------------------------------------------------|
| `a16w16`  | FP16      | 256×256×64   | Foundation — all techniques introduced here across v0 → v9        |
| `a8w8`    | BF8       | 256×256×128  | Same design, larger `BLOCK_K`, 32-cycle scaled MFMA               |
| `a4w4`    | MXFP4     | 256×256×256  | Adds the scale pipeline (GR → LW → LR), 16-cycle MFMA, LDS port contention |

The a16w16 series is the tutorial; a8w8 and a4w4 are what the same design looks like once you change the data type. Each one assumes you've read the previous.

---

## What's Next

Today the tutorial covers GEMM for **16-bit (FP16)**, **8-bit (BF8)**, and **MXFP4** compute. Planned additions: **memory-bound GEMM**, **FAv3 prefill**, **FA decode**, and an **MXFP4 MoE** kernel. See [`ROADMAP.md`](ROADMAP.md) for details.

---

Performance engineering is not magic. It's a process. This repo tries to make that process explicit — one measured bottleneck at a time.
