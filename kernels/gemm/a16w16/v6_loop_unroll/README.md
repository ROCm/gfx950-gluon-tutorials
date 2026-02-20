# v6_loop_unroll — Eliminating Copy Overhead via Loop Unrolling

## 1. Directory Structure

```
v6_loop_unroll/
├── matmul_kernel.py              # The kernel implementation
├── README.md                     # This file
└── ir_dump_K4096_fp16_llirSched/ # IR dumps with llirSched enabled
```

## 2. Motivation

In v5, we identified the bottleneck: copy instructions at the end of each iteration that move `ds_read` results from prefetch registers (`a_next`, `b_next`) to the registers MFMA will consume (`a`, `b`). This overhead is inherent to the prefetch design when using a single-iteration loop body.

> [!IMPORTANT]
> By unrolling the loop by a factor of 2, we can alternate between two register sets directly. Odd iterations use `a/b`, even iterations use `a_next/b_next`—no copying required.

## 3. Loop Unrolling Design

### 3.1 Key Change: Loop Step Size

The loop now iterates with step size 2:

```python
for k in range(0, iterMax - 1, 2):
```

Each unrolled iteration contains two complete sub-iterations that alternate register sets.

### 3.2 Unrolled Loop Body

```python
for k in range(0, iterMax - 1, 2):
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
- First sub-iteration: MFMA consumes `a/b`, ds_read loads into `a_next/b_next`
- Second sub-iteration: MFMA consumes `a_next/b_next`, ds_read loads into `a/b`

The register sets naturally alternate—no copy instructions needed.

### 3.3 Why This Works

In v5, each iteration ended with:
```python
a = a_next
b = b_next
```

This assignment generated copy instructions because `a` and `a_next` are different register allocations.

With unrolling, we eliminate this assignment entirely. Each sub-iteration uses its designated register set, and the loop structure ensures the correct set is ready for the next sub-iteration.

## 4. Performance Analysis

| Version | TFLOPS | MFMA Eff. |
|---------|--------|-----------|
| v5      |   1000 |       59% |
| v5 + llirSched |   1123 |       76% |
| v6      |   1025 |       61% |
| v6 + llirSched |   1105 |       84% |

Without llirSched, v6 (1025 TFLOPS) is slightly slower than v5 + llirSched (1123 TFLOPS) because the baseline scheduler doesn't fully exploit the unrolled structure. However, with llirSched enabled, MFMA efficiency improves from 76% to 84%.

Performance is collected using:
```bash
python bench.py --K 8192 --dtype fp16
```

For an explanation of MFMA efficiency and how to measure it, see [MFMA Efficiency](../../../../docs/mfma_efficiency.md).

## 5. What Comes Next

In `v7_slice`, we explore K-slicing to further improve parallelism and reduce memory pressure.
