# FP16 GEMM Kernel Optimization on AMD GFX9 (Gluon)

This directory presents a **step-by-step optimization journey of an FP16 GEMM kernel written in Gluon**, targeting **AMD MI350/355 GPUs** (gfx950).

Rather than presenting a single "final" kernel, this repository documents **how high performance is achieved**—from a naive baseline to a near-optimal design—covering **memory movement, layout design, latency hiding, and instruction scheduling** along the way.

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
├── v7_sliceN/            # N-slicing for register pressure reduction
├── v8_sliceMN/           # M+N slicing, buffer load throughput analysis
└── v9_beyond_hotloop/    # L2 cache locality via XCD-aware PID remapping
```

## 2. How to Run

From the `a16w16` directory:

```bash
python bench.py --version 9 --K 8192 --dtype fp16
```

This runs correctness checks against `torch.matmul` and reports TFLOPS. Use `--version` to select a kernel version (0–9) and `--rocprof` for accurate performance measurement.

## 3. The Optimization Journey

This section tells the story of how we transformed a 525 TFLOPS naive kernel into a 1587 TFLOPS near-optimal implementation—a **~3.0× improvement** through systematic optimization.

| Version | Name | Focus | Key Concept |
|---------|------|-------|-------------|
| v0 | naive | Baseline | Explicit layouts, correctness-first MFMA kernel |
| v1 | buffer_load | Codegen | Hardware OOB checking, branch elimination |
| v2 | async_copy | Codegen | Direct-to-LDS, eliminates register staging |
| v3 | lds | Codegen | LDS layout design: swizzling vs padding |
| v4 | global_prefetch | Latency hiding | 2-stage pipeline, double buffering |
| v5 | local_prefetch | Latency hiding | 3-stage pipeline, LLIR scheduler introduction |
| v6 | loop_unroll | Codegen | Eliminate copy overhead, DIDT/PIT analysis |
| v7 | sliceN | Register pressure | N-slicing, register allocation workarounds |
| v8 | sliceMN | Register pressure, throughput | M+N slicing, buffer load TCP stall analysis |
| v9 | beyond_hotloop | L2 locality | XCD-aware PID remapping with GROUP_SIZE_M |

### Act I: Getting the Basics Right (v0–v3)

**v0 — The Starting Point.** We begin with a kernel that does exactly one thing well: produce correct results. Every layout is explicit, every data movement visible, nothing hidden. Performance? A modest 525 TFLOPS at 22.5% MFMA efficiency. But correctness comes first—this is our foundation.

**v1 — The Branch Problem.** Examining the generated assembly, we find 140 branch instructions. Why? Masked loads generate branches for out-of-bounds checking. The fix is elegant: `buffer_load` handles OOB in hardware. Branches drop from 140 to 4. The lesson: *sometimes the best optimization is choosing the right instruction.*

**v2 — Eliminating the Middleman.** Data flows from HBM → registers → LDS → registers → MFMA. But why stage in registers? With `buffer_load ... lds`, data goes directly from HBM to LDS. We save 100+ VGPRs and eliminate all `ds_write` instructions. Performance jumps to 672 TFLOPS.

**v3 — The Bank Conflict Detective.** LDS has 64 banks. When threads collide on the same bank, throughput drops. We design three layouts—raw, swizzled, and padded—and measure steady-state `ds_read` throughput. Raw layout: 4-way conflicts, 64-cycle issue latency. Padded layout: conflict-free, 16-cycle issue latency. The winner is clear, and we have a methodology for future designs.

### Act II: Hiding Latency (v4–v5)

**v4 — The Pipeline Revolution.** So far, our loop is embarrassingly sequential: load, wait, compute, repeat. Global memory latency (~400 cycles) stalls everything. The solution: *prefetch the next iteration's data while computing on the current iteration's data.* With double buffering and a 2-stage pipeline, we overlap memory latency with compute. Performance leaps to 1123 TFLOPS—a **46% jump** from the previous version.

**v5 — One More Stage.** MFMA still waits for `ds_read`. We add a third pipeline stage: while MFMA computes iteration k, `ds_read` loads iteration k+1, and `buffer_load` prefetches iteration k+2. Now MFMA, `ds_read`, and `buffer_load` can all run concurrently—if only the compiler would schedule them that way.

Enter the **LLIR Scheduler**. The backend wasn't interleaving instructions as we hoped, so we built a custom scheduler operating at LLVM IR level. It interleaves MFMA with memory operations based on hardware throughput models. With LLIR scheduling, MFMA efficiency jumps from 58% to 80%.

### Act III: Taming the Hardware (v6–v8)

**v6 — Loop Unrolling.** The v5 trace reveals copy instructions at iteration boundaries—data moved between register sets for the prefetch. The fix: unroll by 2, alternating register sets naturally. The copies disappear from the assembly and MFMA efficiency climbs to **87%**. But the kernel now presses against the 512-register ceiling: the two register sets are live at once, leaving no headroom. The remaining inefficiency lives inside the loop—the register allocator shuffles MFMA accumulators between AGPRs and VGPRs with `v_accvgpr_*` copies—and with the register file full, there is no room to fix it in place.

**v7 — Fixing the Register Budget by Design.** With 256×256 tiles and prefetching, we need ~512 registers—exactly what gfx950 provides. No headroom. v7 slices along N: instead of loading a full 256-wide B tile, we load two 128-wide halves in sequence. Register pressure drops to 448 by construction. That headroom is the enabler for the real fix. **force-agpr** (`TRITON_FORCE_MFMA_AGPR=1`) pins every MFMA accumulator into an AGPR, so the allocator no longer shuffles accumulators between register files inside the loop—the `v_accvgpr_*` copies vanish and MFMA efficiency reaches **97%**.

The last gap (97% → 98%) is scattered SALU instructions at iteration boundaries. **amdgcnas**—an out-of-tree assembly post-processor—packs them with peephole optimizations. The result: **98% MFMA efficiency**. The hot loop is essentially perfect.

**v8 — Slicing Both Dimensions.** v7 sliced only N. v8 also slices M, splitting A into two 128-row halves. Register pressure drops further to 384, and each K-step now has four regions instead of two. This restructuring has two benefits: the **copy problem from v5 disappears** naturally (the four-region load order eliminates overlapping live ranges without unrolling), and a **buffer load stall at large K is resolved** (v7 clustered 16 buffer loads in ~1000 cycles; v8 distributes them across ~1500 cycles of MFMA, giving HBM enough time to respond even under high contention).

### Act IV: Beyond the Loop (v9)

**v9 — One Problem Outside the Loop.** With 98% MFMA efficiency, where does the remaining performance come from? The answer lies outside the hot loop, in **L2 cache locality**.

**The L2 locality puzzle.** MI350 has 8 XCDs, each with its own L2 cache. By default, adjacent workgroups land on different XCDs, destroying cache reuse. We remap PIDs so adjacent tiles share an XCD, then use **GROUP_SIZE_M** to reshape tile layout within each XCD. A simple math model emerges: minimize GM + ⌈P/GM⌉ where P is workgroups per XCD. For P=32, the optimal GM is 4, 6, or 8. Hardware counters confirm: L2 misses drop from ~5M to ~4M. Lower cache traffic means lower power, higher sustained frequency, and the last few percent of TFLOPS uplift on top of v8.

### The Results

Measured on MI355 with shape 4096×4096×8192, FP16:

![Performance Chart](images/performance_chart.png)

> [!IMPORTANT]
> **The Moral:** Each optimization seemed small in isolation: choose the right instruction, add a pipeline stage, unroll a loop. But together, they compound into a ~2.6× speedup. More importantly, each step taught us something about the hardware—and that knowledge transfers to future kernels.

## 4. Beyond FP16

Although this directory focuses on **FP16 compute-bound GEMM**, the same optimization strategy applies to other data types. After completing this tutorial, the recommended next steps are:

1. **[a8w8/](../a8w8/)** (BF8): Apply the same design to BF8 — the tile shape and MFMA instruction change, but the pipeline, N-slicing, and scheduling are identical.
2. **[a4w4/](../a4w4/)** (MXFP4): A more complex use case with per-group scaling, LDS round-trips for scale layout conversion, and new hardware challenges (LDS port contention).

| Data Type | Tile Size | Key Difference |
|-----------|-----------|----------------|
| FP16      | 256×256×64  | Foundation — all techniques introduced here |
| BF8       | 256×256×128 | Same design, larger BLOCK_K, 32-cycle MFMA |
| MXFP4     | 256×256×256 | Adds scale pipeline (GR → LW → LR), 16-cycle MFMA |

## 5. How to Read This

Recommended order:

1. Start with `v0_naive` to understand the baseline and explicit layouts
2. Progress version by version, reading both code and README
3. Use thread traces and layout visualizations to build intuition
4. Pay attention to bottleneck analysis sections—they motivate the next version

If you only want the fastest kernel, jump to v9 with `llirSched + amdgcnas`. If you want to understand **why** it is fast, start from the beginning.

> [!TIP]
> Each README follows a consistent structure: Motivation → Design → Performance Analysis → What Comes Next. This progression builds understanding incrementally.
