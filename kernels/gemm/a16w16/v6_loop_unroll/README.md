# v6_loop_unroll — Eliminating Copy Overhead via Loop Unrolling

## 1. Directory Structure

```
v6_loop_unroll/
├── matmul_kernel.py    # The kernel implementation
└── README.md           # This file
```

## 2. Motivation

In v5, we identified a bottleneck: copy instructions at the end of each iteration that move `ds_read` results from prefetch registers (`a_next`, `b_next`) to the registers MFMA consumes (`a`, `b`). This overhead is inherent to the prefetch design when using a single-iteration loop body.

> [!IMPORTANT]
> By unrolling the loop by a factor of 2, we alternate between two register sets directly. Odd iterations use `a/b`, even iterations use `a_next/b_next`—no copying required.

## 3. Loop Unrolling Design

### 3.1. Loop Step Size

The loop now iterates with step size 2:

```python
for k in range(0, iterMax - 2, 2):
```

Each unrolled iteration contains two sub-iterations that alternate register sets.

### 3.2. Unrolled Loop Body

```python
for k in range(0, iterMax - 2, 2):
    # --- First sub-iteration: use a/b, prefetch into a_next/b_next ---
    g_idx = 0
    l_idx = 1

    acc = gl.amd.cdna3.mfma(a, b, acc)

    gl.amd.cdna4.async_copy.wait_group(0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets, ...)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets, ...)
    gl.amd.cdna4.async_copy.commit_group()

    a_next = smemA.index(l_idx).load(dotOpLayoutA)
    b_next = smemB.index(l_idx).load(dotOpLayoutB)

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk

    # --- Second sub-iteration: use a_next/b_next, prefetch into a/b ---
    g_idx = 1
    l_idx = 0

    acc = gl.amd.cdna3.mfma(a_next, b_next, acc)

    gl.amd.cdna4.async_copy.wait_group(0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets, ...)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets, ...)
    gl.amd.cdna4.async_copy.commit_group()

    a = smemA.index(l_idx).load(dotOpLayoutA)
    b = smemB.index(l_idx).load(dotOpLayoutB)

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
```

Key observations:
- First sub-iteration: MFMA consumes `a/b`, `ds_read` loads into `a_next/b_next`
- Second sub-iteration: MFMA consumes `a_next/b_next`, `ds_read` loads into `a/b`

The register sets alternate naturally—no copy instructions needed.

### 3.3. Epilogue

With loop unrolling, the epilogue handles the remaining iterations. Since the main loop ends at `iterMax - 2`, the epilogue processes the final two iterations:

```python
## Epilogue
## iterMax - 2
l_idx = 1
acc = gl.amd.cdna3.mfma(a, b, acc)
a_next = smemA.index(l_idx).load(dotOpLayoutA)
b_next = smemB.index(l_idx).load(dotOpLayoutB)

## iterMax - 1
acc = gl.amd.cdna3.mfma(a_next, b_next, acc)
```

Here we assume `iterMax` is even. With an unroll factor of 2 and the main loop ending at `iterMax - 2`, exactly two iterations remain. The epilogue alternates register sets like the main loop: iteration `iterMax - 2` uses `a/b`, iteration `iterMax - 1` uses `a_next/b_next`.

If `iterMax` is odd, only one iteration remains in the epilogue, containing just the final MFMA.

## 4. Performance Analysis

| Version              | TFLOPS | VGPRs | Spills | MFMA Eff. |
|----------------------|--------|-------|--------|-----------|
| v5 + LLIR scheduler  |   1278 |   510 |      0 |       76% |
| v6 + LLIR scheduler  |    346 |   512 |     99 |       19% |

The unroll-by-2 that was supposed to eliminate copy overhead instead produces a **73% TFLOPS regression**. v5 fits its working set in 510 VGPRs with zero spills; v6, structurally identical but unrolled, hits the 512-VGPR ceiling and spills 99 VGPRs to scratch memory. MFMA efficiency collapses from 76% to 19%.

Performance is collected using:
```bash
python scripts/run_perf_table.py --kernel a16w16 --versions 5 6 --configs llir --K 8192 --dtype fp16 --rocprof
```

For an explanation of MFMA efficiency and how to measure it, see [MFMA Efficiency](../../../../docs/mfma_efficiency.md).

### 4.1. Reading the spill count from the assembly

The `.vgpr_spill_count` field in the kernel descriptor at the bottom of the generated `.amdgcn` (in `~/.triton/cache/<hash>/<kernel>.amdgcn`) gives the total number of VGPRs the register allocator had to spill across the kernel. To see *where* they spill, grep for `scratch_load` / `scratch_store` and check whether each one falls inside the inner loop or in the prologue/epilogue.

**Where the spill lives matters far more than how many there are.** Spills outside the hot loop are paid once per kernel launch and amortize over thousands of MFMA cycles — they barely register in the perf number. v5 + LLIR scheduler is the clean baseline at zero spills end-to-end. v6 + LLIR scheduler, by contrast, has 99 spills, of which 33 fall inside the unrolled loop body and 52 in the prologue/epilogue. The 52 outside contribute almost nothing; the 33 inside drive the regression.

### 4.2. What an in-loop spill actually costs

When a register is spilled, the compiler inserts a `scratch_load` to bring the value back from private memory before its next use, followed by `s_waitcnt vmcnt(0)` to ensure the load has completed. `scratch_load` goes through L1 and, on a miss, all the way to HBM — hundreds of cycles in the worst case. The `vmcnt(0)` wait stalls the wave until that load is fence-resolved, which in turn stalls every MFMA instruction whose input depends on the spilled value.

This is why MFMA efficiency drops from 76% to 19% even though v6 schedules the same MFMA + `ds_read` + `buffer_load` pattern as v5: the LLIR scheduler interleaves memory operations to hide their latency, but it cannot hide a `scratch_load` / `vmcnt(0)` pair sitting on the MFMA critical path. Each in-loop spill round-trip serializes the pipeline against HBM, and 33 of them per unrolled iteration is enough to keep the MFMA unit idle most of the time.

### 4.3. Why does v6 spill?

The short answer is that v6's register footprint exceeds the 512-VGPR budget that gfx9 provides per SIMD. Loop unrolling by 2 widens the live ranges of the prefetch staging registers — both `a/b` and `a_next/b_next` pairs are simultaneously live across the unrolled body — and the LLIR scheduler's aggressive interleaving leaves the allocator no room to reuse them. v5 fits in 510 VGPRs without spills; v6 needs 512 + 99 spills to compile.

The closed-form register accounting that quantifies this, and the design change that fixes it, is in [v7 §2.1, Register Usage Analysis](../v7_sliceN/README.md#21-register-usage-analysis). v7 is where the spill problem gets solved by construction.

## 5. What Comes Next

v6 has hit the register-budget ceiling. The remedy is not to coax the allocator into a tighter packing — that strategy plateaus quickly — but to design the kernel so the register footprint fits comfortably under 512 VGPRs *by construction*. v7 introduces slicing along N to halve the B tile's register cost, opening enough headroom to absorb the prefetch buffers and cleanly eliminate the spills.
