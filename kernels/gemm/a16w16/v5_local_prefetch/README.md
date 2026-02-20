# v5_local_prefetch — 3-Stage Pipeline with Local Prefetch

## 1. Directory Structure

```
v5_local_prefetch/
├── matmul_kernel.py              # The kernel implementation
├── README.md                     # This file
├── ir_dump_K4096_fp16/           # IR dumps for analysis
└── ir_dump_K4096_fp16_llirSched/ # IR dumps with llirSched enabled
```

## 2. Motivation

In v4, we identified the bottleneck: MFMA must wait for `ds_read` to complete because it depends on data loaded from LDS to registers. This dependency prevents MFMA from starting at the beginning of each iteration.

> [!IMPORTANT]
> By issuing `ds_read` for the **next** iteration while the **current** iteration's MFMA is executing, we break the dependency between MFMA and `ds_read` within the same iteration.

This transforms the pipeline from 2 stages to 3 stages:
- Stage 0: Global memory → LDS (async copy)
- Stage 1: LDS → registers (ds_read / local load)
- Stage 2: MFMA compute

## 3. Pipeline Design

This kernel implements a 3-stage software pipeline:

```
Prologue:
    AC A0, B0 → buffer 0
    AC A1, B1 → buffer 1
    wait buffer 0
    local_load A0, B0 ← buffer 0

Main Loop (g_idx = k % 2, l_idx = 1 - g_idx):
    DOT(A[k], B[k])                        (consume prefetched data)
    wait buffer l_idx
    AC A[k+2], B[k+2] → buffer g_idx       (prefetch k+2)
    local_load A[k+1], B[k+1] ← buffer l_idx   (prefetch next for registers)

Epilogue:
    DOT(A[last-1], B[last-1])
    DOT(A[last], B[last])
    store(acc)
```

### 3.1 Key Difference from v4

In v4, each iteration looked like:
```
async_copy (next) → wait → ds_read → mfma
```

In v5, the structure becomes:
```
mfma (current) → wait → async_copy (next+1) → ds_read (next)
```

The critical change is that **MFMA executes first** using data that was prefetched in the previous iteration. Meanwhile, `ds_read` loads data for the *next* iteration. This decouples MFMA from the `ds_read` in the same iteration.

### 3.2 Prologue

The prologue now issues two async copies and one local load to prime the pipeline:

```python
## Prologue
## AC A0, B0 --> buffer 0
g_idx = 0
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
gl.amd.cdna4.async_copy.commit_group()

## AC A1, B1 --> buffer 1
g_idx = 1
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
gl.amd.cdna4.async_copy.commit_group()

## wait buffer 0, local_load A0, B0
gl.amd.cdna4.async_copy.wait_group(1)
l_idx = 0
a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)
```

### 3.3 Main Loop

In the main loop, three independent operations happen in parallel:
1. **MFMA** computes with data from buffer `g_idx` (loaded in previous iteration)
2. **Async copy** prefetches data for iteration `k+2` into buffer `g_idx`
3. **Local load** prefetches data for iteration `k+1` from buffer `l_idx`

```python
for k in range(0, iterMax - 1):
    g_idx = k % 2
    l_idx = 1 - g_idx

    acc = gl.amd.cdna3.mfma(a, b, acc)  # Use prefetched data

    gl.amd.cdna4.async_copy.wait_group(0)

    # Async copy for k+2 (masked on last iteration)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA.index(g_idx), a_base, a_offsets, mask=(k != (iterMax - 2))
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB.index(g_idx), b_base, b_offsets, mask=(k != (iterMax - 2))
    )
    gl.amd.cdna4.async_copy.commit_group()

    # Local load for k+1
    a_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b_next = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)

    a = a_next
    b = b_next
```

> [!NOTE]
> The `mask=(k != (iterMax - 2))` prevents issuing async copy on the last iteration when there's no more data to prefetch.

### 3.4 Epilogue

The epilogue processes the final tile:

```python
## Epilogue
acc = gl.amd.cdna3.mfma(a, b, acc)
```

## 4. Performance Analysis

| Version | TFLOPS | MFMA Eff. |
|---------|--------|-----------|
| v4      |    984 |       57% |
| v5      |   1000 |       59% |
| v5 + llirSched |   1123 |       76% |

The 3-stage pipeline provides a modest improvement in the baseline case (57% → 59%). However, when combined with `llirSched` (LLIR-level instruction scheduling), MFMA efficiency jumps to 76%.

Performance is collected using:
```bash
python bench.py --K 8192 --dtype fp16
```

For an explanation of MFMA efficiency and how to measure it, see [MFMA Efficiency](../../../../docs/mfma_efficiency.md).

### Bottleneck Analysis

Even with llirSched, MFMA efficiency is 76%—there is still room for improvement. Examining the generated assembly in [`ir_dump_K4096_fp16_llirSched/v5_local_prefetch.s`](./ir_dump_K4096_fp16_llirSched/v5_local_prefetch.s) (lines 814–907), we see a block of copy instructions at the end of each iteration:

```asm
v_accvgpr_mov_b32 a120, a128
v_accvgpr_mov_b32 a121, a129
...
v_accvgpr_mov_b32 a56, a204
v_accvgpr_mov_b32 a57, a205
...
```

These instructions copy the `ds_read` results (which landed in registers like `a[128:131]`, `a[204:207]`) to the registers that MFMA will consume in the next iteration (like `a[120:123]`, `a[56:59]`). This overhead is inherent to the prefetch design: since `a_next` and `b_next` are loaded into different registers than `a` and `b`, the data must be copied before the next iteration can use it.

This copy overhead is unavoidable in a single-iteration loop body—unless we unroll the loop.

## 5. What Comes Next

In `v6_loop_unroll`, we unroll the loop to eliminate this copy overhead. With unrolling, alternating iterations can use different register sets directly, avoiding the need to copy between them.
