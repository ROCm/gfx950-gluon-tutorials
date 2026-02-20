# v4_global_prefetch — Software Pipelining with Global Data Prefetch

## 1. Directory Structure

```
v4_global_prefetch/
├── matmul_kernel.py      # The kernel implementation
├── README.md             # This file
└── ir_dump_K4096_fp16/   # IR dumps for analysis
```

## 2. Motivation

In all previous versions, the main loop follows a simple sequential pattern:

```
for k in range(0, K // BLOCK_K):
    load A_tile, B_tile from global memory → LDS
    wait for load to complete
    load from LDS → registers
    compute MFMA
```

This structure has a fundamental inefficiency: **MFMA execution cannot begin until data arrives from global memory**. Global memory latency (hundreds of cycles) directly stalls compute.

> [!IMPORTANT]
> **Software pipelining** restructures the loop to overlap memory latency with computation. The key idea is to **prefetch the next iteration's data while computing on the current iteration's data**.

The simplest form is a 2-stage pipeline:
- Stage 0: Global memory → LDS (async copy)
- Stage 1: LDS → registers + MFMA compute

We use **double buffering** to implement this: while MFMA consumes data from one LDS buffer, async copy fills the other buffer with the next iteration's data.

## 3. Pipeline Design

This kernel implements a 2-stage software pipeline with the following structure:

```
Prologue:
    AC A0, B0 → buffer 0

Main Loop (iterMax - 1 iterations):
    AC A[k+1], B[k+1] → buffer g_idx      (prefetch next iteration)
    wait for buffer l_idx                  (wait for previous iteration's data)
    local_load A[k], B[k] ← buffer l_idx
    DOT(A[k], B[k])

Epilogue:
    wait for final buffer
    local_load A[last], B[last]
    DOT(A[last], B[last])
    store(acc)
```

Where `g_idx` and `l_idx` alternate between 0 and 1:
- `l_idx = k % 2` — buffer to consume (local load)
- `g_idx = 1 - l_idx` — buffer to fill (global prefetch)

### 3.1 Double Buffering

The key change from previous versions is allocating **two LDS buffers** instead of one:

```python
nBuffers: gl.constexpr = 2
smemA = gl.allocate_shared_memory(
    a_ptr.dtype.element_ty, [nBuffers, BLOCK_M, BLOCK_K], sharedLayoutA
)
smemB = gl.allocate_shared_memory(
    b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N], sharedLayoutB
)
```

The buffer dimension is added as the outermost axis, so `smemA.index(0)` and `smemA.index(1)` access the two separate buffers.

### 3.2 Prologue

Before entering the main loop, we issue the first async copy to fill buffer 0:

```python
## Prologue
g_idx = 0
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
gl.amd.cdna4.async_copy.commit_group()
a_base += BLOCK_K * stride_ak
b_base += BLOCK_K * stride_bk
```

### 3.3 Main Loop

Each iteration of the main loop:
1. Issues async copy for the **next** iteration's data
2. Waits for the **previous** iteration's async copy to complete
3. Loads from LDS and computes MFMA

```python
for k in range(0, iterMax - 1):
    l_idx = k % 2      # buffer to consume
    g_idx = 1 - l_idx  # buffer to fill

    # Prefetch next iteration's data
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemA.index(g_idx), a_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(smemB.index(g_idx), b_base, b_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    # Wait for previous iteration's data (1 group in flight)
    gl.amd.cdna4.async_copy.wait_group(1)

    # Consume from buffer l_idx
    a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
    b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)
    acc = gl.amd.cdna3.mfma(a, b, acc)

    a_base += BLOCK_K * stride_ak
    b_base += BLOCK_K * stride_bk
```

> [!NOTE]
> `wait_group(1)` means "wait until at most 1 async copy group is still in flight." Since we just issued a new group for the next iteration, this effectively waits for all previous groups to complete.

### 3.4 Epilogue

After the loop, we process the final tile that was prefetched in the last loop iteration:

```python
## Epilogue
gl.amd.cdna4.async_copy.wait_group(0)  # Wait for all async copies
l_idx = (iterMax - 1) % 2
a = gl.amd.cdna4.async_copy.load_shared_relaxed(smemA.index(l_idx), dotOpLayoutA)
b = gl.amd.cdna4.async_copy.load_shared_relaxed(smemB.index(l_idx), dotOpLayoutB)
acc = gl.amd.cdna3.mfma(a, b, acc)
```

## 4. Impact on Generated Code

### 4.1 Performance

| Version | TFLOPS | VGPRs | MFMA Eff. |
|---------|--------|-------|-----------|
| v3      |    700 |   420 |       43% |
| v4      |    984 |   446 |       57% |

Software pipelining delivers a **40% performance improvement** (700 → 984 TFLOPS) by overlapping global memory latency with compute.

Performance is collected using:
```bash
python bench.py --K 8192 --dtype fp16
```

### 4.2 MFMA Efficiency

**What is MFMA efficiency?**

In the main loop, MFMA efficiency is the ratio of total cycles spent executing MFMA instructions over the total cycles of the loop per SIMD. The higher the MFMA efficiency, the better the utilization of the matrix core. An ideal compute-bound kernel should have MFMA efficiency very close to 100%.

**Why measure MFMA efficiency?**

TFLOPS is the typical end-to-end performance metric, but it can be affected by factors outside the kernel developer's control: temperature, cooling, system state, etc. MFMA efficiency is cycle-based and does not factor in frequency, making it more stable across runs. Since this metric is directly related to instruction scheduling and co-execution, it serves as a better guide for kernel developers and compilers.

**How to measure it?**

Collect thread traces via `rocprofv3`. Steps can be found at: [Triton Profiling with ATT](https://amd.atlassian.net/wiki/spaces/MLSE/pages/744188574/Triton+Profiling+with+ATT)

Then use the `process_json.py` script to process the generated `ui_` directory. Example output:

```json
{
  "loop_first_index": 773,
  "epilogue_first_index": 1038,
  "mfma_count_in_loop": 128,
  "total_mfma_cycles_in_loop": 2048,
  "loop_hitcount": 1016,
  "epilogue_hitcount": 8,
  "num_iterations": 127.0,
  "wave_durations": {
    "se0_sm0_sl0_wv0.json": 456316
  },
  "average_loop_duration": 456316.0,
  "average_prologue_duration": 2708.0,
  "average_epilogue_duration": 18552.0,
  "pro_ratio": "0.57%",
  "loop_ratio": "95.55%",
  "epi_ratio": "3.88%",
  "average_iteration_duration": 3593.0393700787404,
  "mfma efficiency": "57.00%"
}
```

This shows the cycle ratio of prologue, loop, and epilogue, along with MFMA efficiency in the loop. For deeper analysis, ATT Viewer can visualize thread traces to zoom into specific code regions. For MFMA efficiency as a metric, the script suffices.

### 4.3 Assembly Structure

The generated assembly shows clear separation between prefetch and compute phases. Looking at the main loop, we see:

1. **buffer_load_dwordx4 with lds flag** — async copy to LDS
2. **s_waitcnt vmcnt/lgkmcnt** — synchronization
3. **ds_read_b128** — local load from LDS
4. **v_mfma_f32_16x16x32_f16** — matrix multiply-accumulate

The `s_waitcnt lgkmcnt(N)` instructions appear interleaved with MFMA, allowing the scheduler to overlap memory operations with compute.

### 4.4 VGPR Usage

VGPR count increases slightly from 420 to 446. This is expected because:
- Double buffering requires tracking two sets of LDS buffer indices
- Additional loop state variables for buffer alternation

This modest increase is a good trade-off for the significant performance gain.

## 5. What Comes Next

In `v5_local_prefetch`, we extend the pipeline to 3 stages by also prefetching the LDS → register transfer, further improving compute/memory overlap.
