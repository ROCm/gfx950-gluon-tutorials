# v9_beyond_hotloop — Optimizations Beyond the Hot Loop

## 1. Directory Structure

```
v9_beyond_hotloop/
├── matmul_kernel.py                        # The kernel implementation
├── README.md                               # This file
├── ir_dump_K8192_fp16/                     # IR dumps for analysis
├── ir_dump_K8192_fp16_llirSched/           # IR dumps with llirSched
└── ir_dump_K8192_fp16_llirSched_amdgcnas/  # IR dumps with llirSched + amdgcnas
```

## 2. Motivation

Previous versions focused on improving MFMA efficiency inside the loop—maximizing compute utilization on each CU through pipelining, prefetching, and register pressure management.

However, TFLOPS depends on more than per-SIMD cycle efficiency. Two additional factors matter:

1. **Frequency**: Higher power consumption leads to lower sustained clock frequency. As discussed in v7 regarding DIDT protection, reducing power consumption is essential for achieving peak TFLOPS.

2. **Epilogue overhead**: After the loop completes, the epilogue converts accumulators to the output type and writes results to global memory. When K is large, the epilogue is a small fraction of total execution time. As K decreases, the epilogue takes a larger share and its inefficiencies directly reduce TFLOPS.

This version addresses both factors with two optimizations:

- **L2 cache locality** (Section 3): XCD-aware PID remapping and workgroup swizzling reduce L2 misses, lowering power consumption and sustaining higher frequency.
- **Interleaved epilogue** (Section 4): `extract_slice` breaks each accumulator half into sub-tiles along the M dimension and interleaves MFMA computation with stores, reducing `buffer_store` contention when CUs reach the epilogue simultaneously. This is distinct from v7's N-slicing (splitting the output into left/right halves along the N dimension for register pressure), which is present in both epilogue strategies discussed here.

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

The math model predicts that GM = 4, 6, or 8 should achieve better L2 locality than v7 or v8 with GM=1. Hardware counter measurements confirm this:

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

## 4. Epilogue Optimization

### 4.1 The Problem: Store Contention at Small K

After the loop, the epilogue converts accumulators to the output type and writes results to global memory. A straightforward approach completes all remaining MFMAs, then issues all stores in a burst. This works well when K is large—CUs finish the loop at different times due to natural variation, so their stores are staggered. But when K is small, all CUs complete the loop nearly simultaneously and issue stores at the same time, saturating the L2/HBM write path.

### 4.2 Two Epilogue Strategies

Both strategies build on v7's N-slicing, which splits the output tile into left (N/2) and right (N/2) halves to reduce register pressure. The difference is how the final MFMAs and stores are organized within each half.

**v7-style epilogue (full-tile stores)**: Completes all remaining MFMAs, then converts and stores the full 256×128 `acc_left` tile, followed by the full 256×128 `acc_right` tile:

```python
## Region 2: finish MFMAs, then store left tile
acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left)
b_right = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_right.index(g_idx), dotOpLayoutB)

c_left = acc_left.to(a_ptr.dtype.element_ty)
c_left = gl.convert_layout(c_left, layout=gStoreLayoutC)
gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_left_offsets, stored_value=c_left)

## Region 3: finish MFMAs, then store right tile
acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)

c_right = acc_right.to(a_ptr.dtype.element_ty)
c_right = gl.convert_layout(c_right, layout=gStoreLayoutC)
gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_right_offsets, stored_value=c_right)
```

**Interleaved epilogue (sub-tiled stores)**: Uses [`extract_slice`](https://github.com/ROCm/triton/tree/matmul_4waves) to break each 256×128 half into four 64×128 sub-tiles along the M dimension. `extract_slice` is not yet upstreamed and is available on the `matmul_4waves` development branch. MFMAs and stores are pipelined: while sub-tile `i+1` is being computed by MFMA, sub-tile `i` is being stored. This follows the same design principle as the main loop—keeping adjacent operations independent so MFMA and memory operations can execute in parallel:

```python
## sub-tile 0 m[0:64]n[0:128] — compute acc00
a0 = extract_slice(a, [64, 64], [0, 0])
acc00 = extract_slice(acc_left, [64, 128], [0, 0])
acc00 = gl.amd.cdna3.mfma(a0, b_left, acc00)
b_right = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB_right.index(g_idx), dotOpLayoutB)

## sub-tile 1 m[64:128]n[0:128] — compute acc01 while storing acc00
a1 = extract_slice(a, [64, 64], [64, 0])
acc01 = extract_slice(acc_left, [64, 128], [64, 0])
acc01 = gl.amd.cdna3.mfma(a1, b_left, acc01)
c00 = acc00.to(a_ptr.dtype.element_ty)
c00 = gl.convert_layout(c00, layout=gStoreLayoutC)
gl.amd.cdna3.buffer_store(stored_value=c00, ptr=c00_base, offsets=c_slice_offsets)

## sub-tiles 2-3 for acc_left, then 4 sub-tiles for acc_right (same pattern)
```

### 4.3 Performance Comparison

We measured both epilogue strategies across three scheduler configs and two K values:

- **`base`**: Default LLVM instruction scheduler.
- **`llir`**: LLIR scheduler, which reorders instructions in the **loop body only**. The epilogue still uses the default LLVM scheduler.
- **`llir+amdgcnas`**: Same as `llir`, with additional AMD GCN assembly-level scheduling passes (also loop-only).

**K = 8192 (M=N=4096, fp16)**

| Config | TFLOPS | VGPRs | Spills | Epi Cycles | v_accvgpr_ Count |
|---|---|---|---|---|---|
| | interleaved / v7 | interleaved / v7 | interleaved / v7 | interleaved / v7 | interleaved / v7 |
| base | 1339 / 1335 | 452 / 444 | 0 / 0 | 12,108 / 13,072 | 154 / 110 |
| llir | 1424 / 1390 | 444 / 444 | 4 / 0 | 14,744 / 12,552 | 208 / 128 |
| llir+amdgcnas | 1629 / 1590 | 444 / 444 | 0 / 0 | 12,460 / 13,124 | 280 / 256 |

**K = 512 (M=N=4096, fp16)**

| Config | TFLOPS | VGPRs | Spills | Epi Cycles | v_accvgpr_ Count |
|---|---|---|---|---|---|
| | interleaved / v7 | interleaved / v7 | interleaved / v7 | interleaved / v7 | interleaved / v7 |
| base | 715 / 674 | 452 / 444 | 0 / 0 | 12,076 / 14,564 | 154 / 110 |
| llir | 721 / 719 | 444 / 444 | 4 / 0 | 14,652 / 15,004 | 208 / 128 |
| llir+amdgcnas | 790 / 743 | 444 / 444 | 0 / 0 | 12,868 / 14,292 | 280 / 256 |

**Epilogue cycle difference (interleaved minus v7)**

| Config | K=8192 | K=512 |
|---|---|---|
| base | -964 | -2,488 |
| llir | +2,192 | -352 |
| llir+amdgcnas | -664 | -1,424 |

Negative means the interleaved epilogue is faster.

### 4.4 Key Observations

**The interleaved epilogue is faster at small K.** Under `base` and `llir+amdgcnas`, the interleaved epilogue consistently outperforms the v7-style across both K values. The advantage is larger at K=512—for example, `base` saves 2,488 epilogue cycles and delivers 41 more TFLOPS (+6.1%)—because the store contention effect described in Section 4.1 is more pronounced at small K.

**v_accvgpr_ instruction count.** The interleaved epilogue generates more `v_accvgpr_read` / `v_accvgpr_write` instructions because `extract_slice` requires moving data between ACCVGPRs and VGPRs. However, this data movement overlaps with stores and does not increase epilogue latency in most configs.

**VGPR spills under `llir`.** With the interleaved epilogue, the `llir` config produces 4 VGPR spills. These cause `scratch_load_dword` instructions in the epilogue, each immediately followed by `s_waitcnt vmcnt(0)`, which serializes the scratch load and exposes its full latency. This is why `llir` is the only config where the v7-style epilogue has lower epilogue cycles at K=8192. The spills are a side effect of how `llir` allocates registers for the loop, not an epilogue scheduling decision—`llir` only schedules the loop body; the epilogue uses the default LLVM scheduler regardless.

### 4.5 ATT Analysis: Why the v7-Style Epilogue Suffers at Small K

The epilogue executes the same instructions regardless of K—the same MFMAs, type conversions, and stores. Epilogue cycles should therefore be independent of K. This holds for the interleaved epilogue but not for the v7-style:

**Epilogue cycles vs K (llir+amdgcnas config)**

| Epilogue | K=8192 | K=512 | Delta |
|---|---|---|---|
| Interleaved | 12,392 | 12,344 | +48 |
| v7-style | 14,484 | 15,224 | +740 |

The interleaved epilogue is stable (48-cycle variation), while the v7-style takes 740 extra cycles at K=512.

To understand where these extra cycles come from, we collected per-instruction ATT (Advanced Thread Trace) timing for the v7-style epilogue at both K values. The cycles are broken down by instruction category:

**v7-style epilogue cycle breakdown (llir+amdgcnas config)**

| Category | K=8192 | K=512 | Extra at K=512 |
|---|---|---|---|
| `buffer_store_dwordx4` | 3,232 | 3,904 | +672 |
| `s_endpgm` (store drain) | 552 | 1,328 | +776 |
| `s_waitcnt` | 1,432 | 1,456 | +24 |
| `s_barrier` | 680 | 604 | -76 |
| Other | 8,668 | 8,764 | +96 |
| **Total** | **14,564** | **16,056** | **+1,492** |

The `s_waitcnt` delta is negligible (+24 cycles) because the progressive `wait_group(2)` → `wait_group(1)` → `wait_group(0)` allows MFMA computation to overlap with async copy draining. Without progressive waits, a single `wait_group(0)` at the epilogue entry would stall for hundreds of extra cycles at small K, waiting for the last loop iteration's async copies to complete.

The penalty is dominated by two sources:

1. **`buffer_store` back-pressure (+672 cycles)**: At K=512, all CUs complete the loop at nearly the same time and issue stores simultaneously. Stores from all 256 CUs saturate the L2/HBM write path, causing individual `buffer_store_dwordx4` instructions to stall while waiting for write slots.

2. **`s_endpgm` store drain (+776 cycles)**: `s_endpgm` waits for all outstanding stores to complete before the wave terminates. At K=512, store congestion means more writes are still in flight when `s_endpgm` is reached, requiring a longer drain.

The interleaved epilogue avoids both problems. The MFMAs between store groups (computing sub-tile `i+1` while storing sub-tile `i`) create natural gaps in the store stream, preventing the burst that causes contention.

### 4.6 Summary

| Strategy | Pros | Cons |
|---|---|---|
| v7-style (full-tile) | Fewer v_accvgpr_ instructions; simpler code | Store burst causes write contention at small K |
| Interleaved (sub-tiled) | Stores pipelined with compute; stable across K | More v_accvgpr_ data movement; can cause spills under `llir` |

The interleaved epilogue is used in this kernel because it provides consistent performance across K values and the best TFLOPS under `base` and `llir+amdgcnas` configs.

## 5. Summary

This version demonstrates that achieving peak TFLOPS requires optimization beyond the hot loop:

- **L2 cache locality**: XCD-aware PID remapping and workgroup swizzling (GROUP_SIZE_M) reduce L2 misses by ~40%, lowering power consumption and sustaining higher frequency.
- **Interleaved epilogue**: `extract_slice` breaks accumulators into sub-tiles and pipelines stores with MFMA computation, avoiding the `buffer_store` contention that occurs when all CUs reach the epilogue simultaneously at small K.

Combined with optimizations from previous versions (pipelining, local prefetch, loop unrolling, N-slicing), this kernel achieves **1634 TFLOPS** at K=8192 and **798 TFLOPS** at K=512 with optimal L2 locality and efficient epilogue execution.
