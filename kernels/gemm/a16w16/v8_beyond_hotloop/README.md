# v8_beyond_hotloop — Kernel-Level Optimizations

## 1. Directory Structure

```
v8_beyond_hotloop/
├── matmul_kernel.py                        # The kernel implementation
├── README.md                               # This file
├── ir_dump_K8192_fp16/                     # IR dumps for analysis
├── ir_dump_K8192_fp16_llirSched/           # IR dumps with llirSched
└── ir_dump_K8192_fp16_llirSched_amdgcnas/  # IR dumps with llirSched + amdgcnas
```

## 2. Motivation

Previous versions focused on improving MFMA efficiency inside the loop—maximizing compute utilization on each CU through pipelining, prefetching, and register pressure management. MFMA efficiency measures cycle-wise performance on each SIMD.

However, final performance reported as TFLOPS depends on both cycles and frequency. Frequency is governed by power management: higher power consumption leads to lower sustained frequency. As discussed in v7 regarding DIDT protection, power-aware optimization is essential for achieving peak TFLOPS.

To reduce power consumption, we consider three strategies:

- **Use power-efficient instructions**: 16×16 MFMA instructions are more power-efficient than 32×32 MFMA instructions. Our kernel already uses the efficient variant.
- **Use large tile sizes**: Workgroups computing different output tiles may request the same input data from A or B. Although these requests hit in cache and pipelining hides latency, every memory request consumes power. Larger tiles reduce redundant requests across workgroups. Our kernel uses 256×256 tiles, the largest feasible size given the register budget per SIMD and Triton's constraint that tile dimensions must be powers of 2.
- **Optimize tile-to-workgroup mapping for L2 locality**: Reduce memory requests from L2 to MALL by ensuring workgroups on the same XCD share input data in L2 cache.

This version focuses on the third strategy: XCD-aware PID remapping and workgroup swizzling to improve L2 cache locality and reduce power consumption.

## 3. L2 Cache Locality

Throughout this section, we use **GM** as shorthand for **GROUP_SIZE_M**.

### 3.1 The Problem: Poor L2 Reuse Across XCDs

gfx942 and gfx950 have 8 XCDs (Accelerator Compute Dies), each with its own L2 cache. The hardware distributes workgroups across XCDs in round-robin order: workgroup 0 goes to XCD 0, workgroup 1 to XCD 1, and so on.

This creates a problem for cache reuse. Adjacent output tiles often share input tiles from A or B. With naive PID assignment, adjacent tiles map to consecutive workgroup IDs—and thus to different XCDs. Since each XCD has a separate L2 cache, shared input data must be reloaded from HBM on each XCD, wasting memory bandwidth.

Consider v7 with shape 4096×4096 and BLOCK_M=BLOCK_N=256. The output is divided into 16×16 = 256 tiles. With row-major PID assignment and round-robin XCD distribution, workgroups on XCD 0 are spread across all rows:

<img src="../images/v7_grid.png" width="600">

XCD 0 receives workgroups from every row of the output grid. As a result, XCD 0 must load the **entire A matrix** (all 16 row-strips) but only 1/8 of B (2 column-strips). The same pattern applies to all XCDs—each requires the full A matrix, resulting in poor L2 cache reuse.

### 3.2 XCD-Aware PID Remapping

The first optimization remaps PIDs so that consecutive tiles land on the **same XCD**:

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

With XCD remapping alone (GM=1), each XCD receives a contiguous block of 32 tiles arranged in 2 rows × 16 columns:

<img src="../images/v8_grid_xcd_GM1.png" width="600">

However, this layout still requires the **entire B matrix** (all 16 column-strips) and only 1/8 of A (2 row-strips) per XCD. The total data footprint per XCD is unchanged—we have simply swapped which matrix is fully loaded.

### 3.3 Workgroup Swizzling with GROUP_SIZE_M

To reduce the data footprint per XCD, we reshape the workgroup layout using GROUP_SIZE_M:

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

With GM=4, the 32 workgroups per XCD are arranged in a 4×8 grid:

<img src="../images/v8_grid_xcd_GM4.png" width="600">

Now each XCD only requires **1/4 of A** (4 row-strips) and **1/2 of B** (8 column-strips). The total data footprint per XCD is significantly reduced compared to either v7 or v8 with GM=1.

### 3.4 Math Model for Optimal GM

The configurations above can be analyzed mathematically. Given:
- Total workgroups: 256
- Workgroups per XCD: P = 32
- Workgroup layout per XCD: GM × ⌈P/GM⌉

Each XCD loads:
- From A: GM row-strips
- From B: ⌈P/GM⌉ column-strips

Total data per XCD is proportional to GM + ⌈P/GM⌉.

**Optimization Problem**: Find integer GM that minimizes $f(\text{GM}) = \text{GM} + \lceil P/\text{GM} \rceil$.

For the continuous relaxation:

$$f(x) = x + \frac{P}{x}$$

$$f'(x) = 1 - \frac{P}{x^2} = 0 \quad \Rightarrow \quad x = \sqrt{P}$$

For P = 32, $\sqrt{32} \approx 5.66$.

Evaluating f(GM) for different values:

| GM | ⌈P/GM⌉ | f(GM) | Configuration |
|----|--------|-------|---------------|
| 16 |      2 |    18 | v7 (no XCD remapping) |
|  2 |     16 |    18 | v8 GM=1 (XCD remapping only) |
|  4 |      8 |    12 | v8 GM=4 |
|  6 |      6 |    12 | v8 GM=6 |
|  8 |      4 |    12 | v8 GM=8 |

The optimal values are GM = 4, 6, or 8, all achieving f(GM) = 12. The math model confirms:
- v7 (effectively GM=16) and v8 GM=1 (effectively GM=2) have the same suboptimal data footprint
- GM = 4, 6, or 8 all achieve the minimum data footprint

### 3.5 Validation

The math model predicts that GM = 4, 6, or 8 should achieve better L2 locality than v7 or v8 with GM=1. Hardware counter measurements in Section 4 confirm this:

- v7 and v8 GM=1 have ~4.7–5.0M L2 misses (f(GM) = 18)
- GM = 4, 6, and 8 all achieve ~3.1M L2 misses (f(GM) = 12)

The reduced L2 traffic translates directly to higher sustained TFLOPS through lower power consumption.

## 4. Performance Analysis

The following table shows performance, MFMA efficiency, and L2 cache behavior for different configurations:

| Configuration                  | TFLOPS | MFMA Eff. | TCC_EA0_RDREQ_DRAM_sum | TCP_TCC_READ_REQ_sum |
|--------------------------------|--------|-----------|------------------------|----------------------|
| v7 + llirSched + amdgcnas      |   1538 |     98.5% |              5,043,636 |           16,777,216 |
| v8 + llirSched + amdgcnas GM=1 |   1556 |     98.6% |              4,737,950 |           16,777,216 |
| v8 + llirSched + amdgcnas GM=4 |   1605 |     98.9% |              3,147,709 |           16,777,216 |
| v8 + llirSched + amdgcnas GM=6 |   1634 |     98.4% |              3,147,743 |           16,777,216 |
| v8 + llirSched + amdgcnas GM=8 |   1608 |     98.9% |              3,147,721 |           16,777,216 |

**Counter Definitions**:
- **TCP_TCC_READ_REQ_sum**: Requests from L1 to L2. All configurations have identical values since all workgroups perform the same computation.
- **TCC_EA0_RDREQ_DRAM_sum**: Requests from L2 to MALL (memory-side last-level cache), indicating L2 cache misses.

**Observations**:
- v7 and v8 GM=1 have similar L2 miss counts (~4.7–5.0M), confirming their equivalent data footprints as predicted by the math model.
- GM = 4, 6, and 8 all achieve ~3.1M L2 misses, matching the optimal f(GM) = 12 prediction.
- Despite similar MFMA efficiency (~98%), the reduced L2 traffic from GM = 4/6/8 leads to higher sustained TFLOPS due to lower power consumption and higher frequency.

Performance is collected using:
```bash
python scripts/run_perf_table.py --kernel a16w16 --versions 7 8 --configs llir+amdgcnas --K 8192 --dtype fp16 --use-rocprof
```

Counters are collected using:
```bash
python scripts/run_counter_collection.py --kernel a16w16 --versions 8 --configs llir+amdgcnas --K 8192 --dtype fp16 --counters TCC_EA0_RDREQ_DRAM_sum,TCP_TCC_READ_REQ_sum
```

For an explanation of MFMA efficiency and how to measure it, see [MFMA Efficiency](../../../../docs/mfma_efficiency.md).

## 5. Summary

This version demonstrates that achieving peak TFLOPS requires attention beyond MFMA efficiency:

- **XCD-aware PID remapping** groups adjacent tiles onto the same XCD
- **Workgroup swizzling (GROUP_SIZE_M)** reshapes tile layout to minimize data footprint per XCD
- **Math model** provides a principled approach to choosing optimal GM

Combined with optimizations from previous versions (pipelining, local prefetch, loop unrolling, N-slicing), this kernel achieves **1634 TFLOPS** with optimal L2 locality.

### Beyond This Version

With near-100% MFMA efficiency inside the loop and optimal L2 locality, the remaining optimization opportunity is the **epilogue**. When K is large, the epilogue is a small fraction of total kernel execution time, so we have not prioritized its optimization.

However, as K decreases, the epilogue takes a larger portion of total execution time. Epilogue optimization (e.g., interleaving MFMA with stores, sliced output writes) will be addressed in future versions if small-K kernels become common use cases.
