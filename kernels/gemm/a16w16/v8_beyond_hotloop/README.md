# v8_beyond_hotloop — Kernel-Level Optimizations

## 1. Directory Structure

```
v8_beyond_hotloop/
├── matmul_kernel.py                  # The kernel implementation
├── README.md                         # This file
├── ir_dump_K4096_fp16/               # IR dumps for analysis
├── ir_dump_K4096_fp16_llirSched/     # IR dumps with llirSched
└── ir_dump_K4096_fp16_llirSched_amdgcnas/  # IR dumps with llirSched + amdgcnas
```

## 2. Motivation

Previous versions focused on optimizing the hot loop—pipelining, prefetching, register pressure. But a complete kernel has more than just the main loop:
- **Prologue**: Initial data loads before the loop
- **Epilogue**: Final computations and stores after the loop
- **Workgroup scheduling**: How thread blocks are mapped to hardware

> [!IMPORTANT]
> This version addresses optimizations beyond the hot loop: XCD-aware PID remapping, workgroup swizzling, and an interleaved epilogue that overlaps final MFMA with stores.

## 3. Key Optimizations

### 3.1 XCD-Aware PID Remapping

gfx942 and gfx950 have 8 XCDs (Accelerator Compute Dies), each with its own L2 cache. The hardware distributes workgroups across XCDs in round-robin order: workgroup 0 to XCD 0, workgroup 1 to XCD 1, and so on.

This creates a problem for cache reuse. Adjacent output tiles often share the same A or B input tiles. With naive PID assignment, adjacent tiles go to consecutive workgroup IDs—and thus different XCDs. Since each XCD has a separate L2 cache, the shared input data must be reloaded from HBM on each XCD, wasting memory bandwidth.

The `get_pids` function solves this by remapping PIDs so that adjacent tiles land on the **same XCD**:

```python
@gluon.jit
def get_pids(M, N, BM, BN, GRID_MN, NUM_XCDS, GROUP_SIZE_M):
    pid = gl.program_id(axis=0)

    if NUM_XCDS != 1:
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS

        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid
    ...
```

By grouping adjacent tiles onto the same XCD, shared input tiles stay in L2 cache and can be reused without reloading from HBM.

### 3.2 Workgroup Swizzling (GROUP_SIZE_M)

In addition to XCD-aware remapping, the kernel uses `GROUP_SIZE_M` for further L2 cache locality:

```python
if GROUP_SIZE_M == 1:
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
else:
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
```

With `GROUP_SIZE_M=4`, workgroups are scheduled in groups that share rows, improving L2 cache reuse for the A matrix.

### 3.3 Choosing Optimal GROUP_SIZE_M

XCD remapping and GROUP_SIZE_M work together to minimize L2 cache traffic. Given the total number of workgroups:

```
#wgs = M × N / BLOCK_M / BLOCK_N
```

Each XCD receives `P = #wgs / 8` workgroups. These P workgroups are arranged in a `GM × (P/GM)` grid, where `GM` is GROUP_SIZE_M.

The P workgroups on each XCD read:
- From A: GM row-strips, each of size K
- From B: P/GM column-strips, each of size K

Total data per XCD is proportional to `K × (GM + P/GM)`.

**Optimization Problem**: Given integer P, find integer GM that minimizes `GM + P/GM`.

For the continuous relaxation:
```
f(x) = x + P/x
f'(x) = 1 - P/x² = 0  →  x = √P
```

**Solution**: GM should be the divisor of P closest to √P.

```python
import math

def optimal_group_m(P):
    """Find GM that minimizes (GM + P/GM)."""
    sqrt_p = math.sqrt(P)
    divisors = []
    for i in range(1, int(sqrt_p) + 1):
        if P % i == 0:
            divisors.append(i)
            if i != P // i:
                divisors.append(P // i)
    return min(divisors, key=lambda d: abs(d - sqrt_p))
```

**Example**: For shape 4096×4096 with BLOCK_M=BLOCK_N=256:
- Total workgroups: 16 × 16 = 256
- Per XCD: P = 256 / 8 = 32
- √32 ≈ 5.66
- Divisors of 32: {1, 2, 4, 8, 16, 32}
- Closest to 5.66: 4 or 8 (both give f(GM) = 12)

### 3.5 Interleaved Epilogue with extract_slice

The epilogue is restructured to overlap MFMA with stores using `extract_slice`:

```python
## slice 0 m[0:64]n[0:128]
a0 = extract_slice(a, [64, 64], [0, 0])
acc00 = extract_slice(acc0, [64, 128], [0, 0])
acc00 = gl.amd.cdna3.mfma(a0, b0, acc00)
c00 = acc00.to(a_ptr.dtype.element_ty)
gl.amd.cdna3.buffer_store(stored_value=c00, ptr=c00_base, offsets=c_slice_offsets)

## slice 1 m[64:128]n[0:128]
a1 = extract_slice(a, [64, 64], [64, 0])
acc01 = extract_slice(acc0, [64, 128], [64, 0])
acc01 = gl.amd.cdna3.mfma(a1, b0, acc01)
...
```

Instead of computing all MFMAs then storing all results, the epilogue interleaves:
1. Compute MFMA for slice 0
2. Store slice 0
3. Compute MFMA for slice 1
4. Store slice 1
5. ...

This allows stores to overlap with subsequent MFMA computations, hiding store latency.

### 3.6 M-Dimension Slicing in Epilogue

The epilogue slices the 256×256 output into 8 pieces (4 M-slices × 2 N-slices):
- `acc00`, `acc01`, `acc02`, `acc03` for N=[0:128]
- `acc10`, `acc11`, `acc12`, `acc13` for N=[128:256]

Each slice is 64×128, small enough to process and store efficiently while maintaining overlap.

## 4. Performance Analysis

| Version                        | TFLOPS | MFMA Eff. |
|--------------------------------|--------|-----------|
| v7 + LLIR scheduler + amdgcnas |   1523 |       98% |
| v8 + LLIR scheduler + amdgcnas |   1610 |       99% |

With full scheduling optimization, v8 achieves **1610 TFLOPS** at 99% MFMA efficiency — the highest performance in this tutorial series.

Performance is collected using:
```bash
python scripts/run_perf_table.py --kernel a16w16 --versions 7 8 --configs llir+amdgcnas --K 8192 --dtype fp16 --use-rocprof
```

For an explanation of MFMA efficiency and how to measure it, see [MFMA Efficiency](../../../../docs/mfma_efficiency.md).

## 5. Summary

This version demonstrates that high-performance GEMM requires attention to the entire kernel, not just the hot loop:

- **XCD-aware PID remapping** enables L2 cache reuse for adjacent tiles
- **Workgroup swizzling** improves L2 cache utilization
- **Interleaved epilogue** overlaps final computations with stores

Combined with the optimizations from previous versions (pipelining, local prefetch, loop unrolling, N-slicing), this kernel achieves near-theoretical peak performance.
