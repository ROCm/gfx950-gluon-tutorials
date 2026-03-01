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

### 3.1 L2 Cache Locality on Multi-XCD GPUs

gfx942 and gfx950 have 8 XCDs (Accelerator Compute Dies), each with its own L2 cache. The hardware distributes workgroups across XCDs in round-robin order: workgroup 0 to XCD 0, workgroup 1 to XCD 1, and so on.

This creates a problem for cache reuse. Adjacent output tiles often share the same A or B input tiles. With naive PID assignment, adjacent tiles go to consecutive workgroup IDs—and thus different XCDs. Since each XCD has a separate L2 cache, the shared input data must be reloaded from HBM on each XCD, wasting memory bandwidth.

Consider v7 as an example with shape 4096×4096 and BLOCK_M=BLOCK_N=256. The output is divided into 16×16 = 256 tiles. With row-major PID assignment and round-robin XCD distribution, workgroups on XCD 0 are spread across all rows:

![v7 Grid Layout](../images/v7_grid.png)

As shown, XCD 0 receives workgroups from every row of the output grid. This means XCD 0 must load the **entire A matrix** (all 16 row-strips) but only 1/8 of the B matrix (2 column-strips). The same pattern applies to all XCDs—each requires the full A matrix, resulting in poor L2 cache reuse.

### 3.2 XCD-Aware PID Remapping

The `get_pids` function remaps PIDs so that consecutive tiles land on the **same XCD**:

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

With XCD remapping alone (GROUP_M=1), each XCD receives a contiguous block of 32 tiles arranged in 2 rows × 16 columns:

![v8 Grid with XCD Remapping, GM=1](../images/v8_grid_xcd_GM1.png)

However, this layout still requires the **entire B matrix** (all 16 column-strips) and only 1/8 of the A matrix (2 row-strips) per XCD. The total data footprint per XCD is unchanged—we've simply swapped which matrix is fully loaded.

### 3.3 Workgroup Swizzling (GROUP_SIZE_M)

To reduce the data footprint per XCD, we reshape the workgroup layout using `GROUP_SIZE_M`:

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

With `GROUP_SIZE_M=4`, the 32 workgroups per XCD are arranged in a 4×8 grid:

![v8 Grid with XCD Remapping, GM=4](../images/v8_grid_xcd_GM4.png)

Now each XCD only requires **1/4 of the A matrix** (4 row-strips) and **1/2 of the B matrix** (8 column-strips). The total data footprint per XCD is significantly reduced compared to either v7 or v8 with GM=1.

### 3.4 Math Model for Optimal GROUP_SIZE_M

The three configurations can be analyzed mathematically. Given:
- Total workgroups: 256
- Workgroups per XCD: P = 32
- Workgroup layout per XCD: GM × (P/GM)

Each XCD loads:
- From A: GM row-strips
- From B: ⌈P/GM⌉ column-strips

Total data per XCD is proportional to GM + ⌈P/GM⌉.

**Optimization Problem**: Find integer GM that minimizes $f(\text{GM}) = \text{GM} + \lceil P/\text{GM} \rceil$.

For the continuous relaxation where GM divides P:

$$f(x) = x + \frac{P}{x}$$

$$f'(x) = 1 - \frac{P}{x^2} = 0 \quad \Rightarrow \quad x = \sqrt{P}$$

For P = 32, $\sqrt{32} \approx 5.66$.

Evaluating f(GM) for different values:

| GM | ⌈P/GM⌉ | f(GM) = GM + ⌈P/GM⌉ | Configuration |
|----|--------|---------------------|---------------|
| 16 |      2 |                  18 | v7 (no XCD remapping) |
|  2 |     16 |                  18 | v8 GM=1 (XCD remapping only) |
|  4 |      8 |                  12 | v8 GM=4 |
|  6 |      6 |                  12 | v8 GM=6 |
|  8 |      4 |                  12 | v8 GM=8 |

The optimal values are GM = 4, 6, or 8, all achieving f(GM) = 12. The math model confirms:
- v7 (GM=16) and v8 GM=1 (GM=2) have the same suboptimal data footprint
- GM=4, 6, or 8 all achieve the minimum data footprint

### 3.5 L2 Cache Measurements

To validate the math model, we collected hardware counters for different configurations:

| Configuration | TCC_EA0_RDREQ_DRAM_sum | TCP_TCC_READ_REQ_sum |
|---------------|------------------------|----------------------|
| v7            |              4,754,139 |           16,777,216 |
| v8 GM=1       |              4,870,303 |           16,777,216 |
| v8 GM=4       |              3,148,321 |           16,777,216 |
| v8 GM=6       |              3,147,765 |           16,777,216 |

- **TCP_TCC_READ_REQ_sum**: Requests from L1 to L2. All configurations have identical values since all workgroups perform the same computation.
- **TCC_EA0_RDREQ_DRAM_sum**: Requests from L2 to MALL (memory-side last-level cache), indicating L2 cache misses.

As expected:
- v7 and v8 GM=1 have similar L2 miss counts (~4.8M), confirming their equivalent data footprints
- v8 GM=4 and GM=6 have similar and lower L2 miss counts (~3.1M), matching the math model prediction

### 3.6 Interleaved Epilogue with extract_slice

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

### 3.7 M-Dimension Slicing in Epilogue

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
