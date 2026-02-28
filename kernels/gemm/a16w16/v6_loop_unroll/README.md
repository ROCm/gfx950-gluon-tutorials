# v6_loop_unroll — Eliminating Copy Overhead via Loop Unrolling

## 1. Directory Structure

```
v6_loop_unroll/
├── matmul_kernel.py                    # The kernel implementation
├── README.md                           # This file
├── ir_dump_K8192_fp16/                 # IR dumps for analysis
├── ir_dump_K8192_fp16_llirSched/       # IR dumps with LLIR scheduler enabled
└── ir_dump_K8192_fp16_llirSched_amdgcnas/  # IR dumps with LLIR scheduler + AMDGCN AS
```

## 2. Motivation

In v5, we identified the bottleneck: copy instructions at the end of each iteration that move `ds_read` results from prefetch registers (`a_next`, `b_next`) to the registers MFMA will consume (`a`, `b`). This overhead is inherent to the prefetch design when using a single-iteration loop body.

> [!IMPORTANT]
> By unrolling the loop by a factor of 2, we can alternate between two register sets directly. Odd iterations use `a/b`, even iterations use `a_next/b_next` — no copying required.

## 3. Loop Unrolling Design

### 3.1. Key Change: Loop Step Size

The loop now iterates with step size 2:

```python
for k in range(0, iterMax - 2, 2):
```

Each unrolled iteration contains two complete sub-iterations that alternate register sets.

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

    a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

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

    a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
```

Notice that:
- First sub-iteration: MFMA consumes `a/b`, `ds_read` loads into `a_next/b_next`
- Second sub-iteration: MFMA consumes `a_next/b_next`, `ds_read` loads into `a/b`

The register sets naturally alternate — no copy instructions needed.

### 3.3. Epilogue

With loop unrolling, the epilogue must handle the remaining iterations. Since the main loop ends at `iterMax - 2`, the epilogue processes the final two iterations:

```python
## Epilogue
## iterMax - 2
l_idx = 1
acc = gl.amd.cdna3.mfma(a, b, acc)
a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
b_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

## iterMax - 1
acc = gl.amd.cdna3.mfma(a_next, b_next, acc)
```

Unrolling makes it tricky to determine how many iterations remain for the epilogue. Here we assume the total number of loop iterations (`iterMax`) is even. With an unroll factor of 2 and a main loop ending at `iterMax - 2`, exactly two iterations are left. The epilogue alternates register sets just like the main loop: iteration `iterMax - 2` uses `a/b`, iteration `iterMax - 1` uses `a_next/b_next`.

If the loop has an odd number of iterations, there will be only 1 iteration in the epilogue containing just the final MFMA.

## 4. Performance Analysis

| Version              | TFLOPS | VGPRs | MFMA Eff. |
|----------------------|--------|-------|-----------|
| v5 + LLIR scheduler  |   1283 |   510 |       76% |
| v6 + LLIR scheduler  |   1260 |   500 |       88% |

With the LLIR scheduler enabled, MFMA efficiency improves from 76% to 88% by eliminating the copy overhead.

Performance is collected using:
```bash
python scripts/run_perf_table.py --kernel a16w16 --versions 5 6 --configs llir --K 8192 --dtype fp16 --use-rocprof
```
This command can be run from anywhere in the repository. See [run_perf_table.py](../../../../scripts/README.md#run_perf_tablepy) for more details.

For an explanation of MFMA efficiency and how to measure it, see [MFMA Efficiency](../../../../docs/mfma_efficiency.md).

### 4.1. What Changed (v5 + LLIR scheduler → v6 + LLIR scheduler)

Comparing the thread traces of v5 and v6 with the LLIR scheduler enabled. Each screenshot shows one iteration of the main loop:

![v5 with LLIR scheduler trace](../images/v5-llirSched_bottleneck_zoomin.png)

![v6 with LLIR scheduler trace](../images/v6-llirSched_bottleneck_zoomin.png)

In v5 (top), copy instructions are visible at the end of each iteration. In v6 (bottom), the copy instructions are eliminated — the register sets alternate naturally without explicit copying.

### 4.2. Bottleneck Analysis

Even with 88% MFMA efficiency, there is still room for improvement. Examining the v6 trace above, we observe MFMA gaps scattered throughout the iteration.

**Bottleneck 1: DIDT control overhead**

The area marked by the **purple circle** shows instructions across iteration boundaries. The VALU instructions in this region take significantly longer than the expected 4 cycles to execute.

This is caused by **DIDT control** (di/dt — rate of change of current). If the voltage rises too fast, the chip may produce incorrect results. The hardware limits how fast the current (and thus voltage) can rise to prevent this instability.

Looking at the trace, there is a long period of non-VALU, non-MFMA instructions across the iteration boundary (scalar operations, waits, etc.). These low-power instructions cause the voltage to drop. Then at the beginning of the next loop iteration, there is a high density of VALU and MFMA instructions. The voltage must rise to support these high-power operations, but DIDT protection prevents it from rising too quickly. The hardware inserts delays to smooth out the current ramp rate.

This is observable in the trace as VALU instructions taking 40-80 cycles instead of the expected 4 cycles.

**Bottleneck 2: AGPR ↔ VGPR copies**

Examining the generated assembly, there are still copy instructions between AGPRs and VGPRs. While we eliminated the `a/b` ↔ `a_next/b_next` copies, the hardware requires data movement between accumulator registers (AGPRs) and vector registers (VGPRs) for certain operations.

## 5. What Comes Next

In the next versions, we focus on addressing the two bottlenecks identified above:

- **Register pressure optimization** — to eliminate the AGPR ↔ VGPR copy overhead
- **Fine-grained peephole optimizations** — to mitigate the DIDT control delays at iteration boundaries
