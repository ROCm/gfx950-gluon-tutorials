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

**Bottleneck 1: VALU stalls from power management**

The area marked by the **purple circle** shows instructions across iteration boundaries. The VALU instructions in this region take significantly longer than the expected 4 cycles to execute—often 40-80 cycles. This is caused by power management mechanisms designed to prevent voltage droops.

**Voltage Droops and Clock Stretching**

When power consumption spikes, the power delivery network cannot react quickly enough, causing a voltage droop. Small droops are acceptable, but large droops can cause incorrect results. To maintain stability, the hardware uses **clock stretching**—extending the clock period to effectively lower frequency until the power delivery adapts and voltage stabilizes.

On MI350, this is handled by **DIDT (Digital Integrated Droop Tracking)**, a closed-loop control system that monitors supply voltage and activity indicators to predict and mitigate droops. Unlike traditional sensors that may be too slow to react, DIDT is integrated with the Digital Frequency-Locked Loop (DFLL) to provide sub-nanosecond response. When a droop is detected, DIDT communicates with the DFLL to stretch the clock, reducing instantaneous power demand:

$$P = C \cdot V^2 \cdot f$$

By reducing frequency ($f$) for a few cycles, DIDT arrests the downward trajectory of voltage. In extreme cases, the DFLL may gate the clock entirely for a few cycles to allow decoupling capacitors to recharge.

**PIT: Proactive Power Spike Mitigation**

To avoid triggering DIDT in the first place, MI350 firmware includes **PIT (Power Instruction Throttling)**. PIT looks ahead in time and analyzes the expected power signature of upcoming instructions. If it detects a large power change (in either direction), it inserts stalls—either across the entire SE or on a per-WGP basis—to spread high-power instructions (e.g., VALU) over time, smoothing the power consumption curve.

**What We Observe in the Trace**

Looking at the trace, there is a long period of low-power instructions (scalar operations, waits) across the iteration boundary. This causes voltage to drop. At the beginning of the next iteration, there is a high density of VALU and MFMA instructions. PIT detects this upcoming power spike and inserts stalls to prevent a droop that would trigger DIDT clock stretching.

The result is VALU instructions taking 40-80 cycles instead of 4 cycles.

**The Solution: Increase MFMA Efficiency**

From a kernel and compiler perspective, the solution is to increase MFMA efficiency—keeping the MFMA unit continuously busy with minimal gaps. This maintains a stable, high power draw without the sudden spikes and drops that trigger PIT stalls or DIDT clock stretching. A steady power profile leads to stable voltage and sustained high frequency.

**Bottleneck 2: AGPR ↔ VGPR copies**

Examining the generated assembly, there are still copy instructions between AGPRs and VGPRs. While we eliminated the `a/b` ↔ `a_next/b_next` copies, the hardware requires data movement between accumulator registers (AGPRs) and vector registers (VGPRs) for certain operations.

## 5. What Comes Next

In the next versions, we focus on addressing the two bottlenecks identified above:

- **Register pressure optimization** — to eliminate the AGPR ↔ VGPR copy overhead
- **Increased MFMA efficiency** — to maintain stable power draw and avoid PIT stalls at iteration boundaries
