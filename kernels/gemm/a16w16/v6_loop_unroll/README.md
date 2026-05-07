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

The unroll-by-2 in v6 eliminates the per-iteration copy as designed — at the IR level, that part of the change works. What it does not eliminate is the register pressure those copies were quietly absorbing. v6 and v5 share exactly the same hot-loop structure under the LLIR scheduler — same MFMA + `ds_read` + `buffer_load` interleaving, same operand layouts, same prefetch pipeline. The difference is in what the LLVM backend can do with the live ranges:

- In **v5**, each iteration ends with a copy `a ← a_next; b ← b_next`. The LLIR scheduler can place that copy in a slot where the backend can reuse VGPRs across iterations. The footprint fits cleanly inside the 512-VGPR budget.
- In **v6**, the unroll removes the copy by alternating buffer roles. In the first sub-iteration, `mfma` reads from `(a, b)` while `ds_read` simultaneously writes into `(a_next, b_next)`; the two operations are concurrent by construction, so their VGPR sets must be disjoint — there is no opportunity for reuse. The footprint blows past 512 and the allocator has to spill 99 VGPRs to scratch.

Two lessons fall out of this:

1. **A solution for one bottleneck can introduce a new one.** v6 cleanly eliminates v5's copy overhead at the IR level; the cost re-emerges one layer down, in register allocation — exactly where the original copies were silently helping.
2. **A regression is not a reason to revert.** The unrolling design is correct; what it surfaces is a real problem that v5 was hiding. The right response is to look deeper, find what actually changed (here, the live-range overlap that the copies had been masking), and fix that — not to throw the unroll away. v7 takes that path.

Performance is collected using:
```bash
python scripts/run_perf_table.py --kernel a16w16 --versions 5 6 --configs llir --K 8192 --dtype fp16 --rocprof
```

For an explanation of MFMA efficiency and how to measure it, see [MFMA Efficiency](../../../../docs/mfma_efficiency.md).

### 4.1. Reading the spill count from the assembly

The `.vgpr_spill_count` field in the kernel descriptor at the bottom of the generated `.amdgcn` (in `~/.triton/cache/<hash>/<kernel>.amdgcn`) gives the total number of VGPRs the register allocator had to spill across the kernel. To see *where* they spill, grep for `scratch_load` / `scratch_store` and check whether each one falls inside the inner loop or in the prologue/epilogue.

**Where the spill lives matters far more than how many there are.** Spills outside the hot loop are paid once per kernel launch and amortize over thousands of MFMA cycles — they barely register in the perf number. v5 + LLIR scheduler is the clean baseline at zero spills end-to-end. v6 + LLIR scheduler distributes its 99 spills across both regions; only the in-loop spills drive the regression while the rest amortize to noise.

### 4.2. What an in-loop spill actually costs

When a register is spilled, the compiler inserts a `scratch_load` to bring the value back from private memory before its next use, followed by `s_waitcnt vmcnt(0)` to ensure the load has completed. `scratch_load` goes through L1 and, on a miss, all the way to HBM — hundreds of cycles in the worst case. The `vmcnt(0)` wait stalls the wave until that load is fence-resolved, which in turn stalls every MFMA instruction whose input depends on the spilled value.

This is why MFMA efficiency drops from 76% to 19% even though v6 schedules the same MFMA + `ds_read` + `buffer_load` pattern as v5: the LLIR scheduler interleaves memory operations to hide their latency, but it cannot hide a `scratch_load` / `vmcnt(0)` pair sitting on the MFMA critical path. Each in-loop spill round-trip serializes the pipeline against HBM, and a handful of them per unrolled iteration is enough to keep the MFMA unit idle most of the time.

### 4.3. Where this gets fixed

The closed-form register accounting that quantifies the spill — and the design change that resolves it — is in [v7 §2.1, Register Usage Analysis](../v7_sliceN/README.md#21-register-usage-analysis). v7 fixes the problem by construction.

## 5. What Comes Next

v6 has hit the register-budget ceiling. The remedy is not to coax the allocator into a tighter packing — that strategy plateaus quickly — but to design the kernel so the register footprint fits comfortably under 512 VGPRs *by construction*. v7 introduces slicing along N to halve the B tile's register cost, opening enough headroom to absorb the prefetch buffers and cleanly eliminate the spills.
