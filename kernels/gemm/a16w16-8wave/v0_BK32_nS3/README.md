# v0_BK32_nS3 — 8-wave warp-pipeline baseline (BLOCK_K=32, 3-buffer ring)

## 1. Directory Structure

```
v0_BK32_nS3/
├── matmul_kernel.py    # The kernel implementation
└── README.md           # This file
```

## 2. What this kernel is

The tuned 8-wave baseline. Unlike the 4-wave `a16w16` tutorial kernels — which
schedule the hot loop with the `llir+amdgcnas` toolchain — the 8-wave kernels
schedule themselves at the **wave level** with `warp_pipeline_stage`, and run in
plain "base" mode with **no AGPRs** (`TRITON_HIP_AGPR_ALLOC="0,0"`). See the
[parent README](../README.md) for why the `llir+amdgcnas` path can't be used here
(it is built around the 4-wave register/schedule model and fails register
allocation at 8 waves).

| | value |
|---|---|
| Warps / CTA | 8 (`warpsPerCTA=[2,4]`) |
| Tile M×N×K | 256×256×**32** |
| LDS buffers | **3** (triple-buffered ring) |
| K-unroll | **3×** (= ring size) |
| Scheduling | `warp_pipeline_stage` (mfma / mem regions) |
| LDS allocation | one `smemA[3]` / `smemB[3]` ring |

Each K-step the loop issues, per region: a **DOT** (MFMA), an **LR** (local read,
`ds_read` LDS→regs) for the next step's operand, and an **AC** (async copy,
`buffer_load_to_lds` HBM→LDS) refilling the ring slot just consumed.

## 3. Key design points

### 3.1 Relaxed `local_load` to drop a redundant LDS barrier

Inside a mem region the kernel does `smem.index(k+1).load()` (LR) followed by
`buffer_load_to_shared(smem.index(k))` (AC) — a read then a write of the **same
allocation** at **different** ring indices. Triton's membar analysis
(`lib/Analysis/Membar.cpp`) cannot disambiguate `MemDescIndexOp` sub-buffers (it
only reads static offsets from `MemDescSubsliceOp`), so it conservatively assumes
the LR and AC overlap and inserts a redundant `s_waitcnt lgkmcnt(0)` + `s_barrier`
between them.

Switching the LR to `cdna4_async.load_shared_relaxed` makes the load carry an
async-wait token (`isSyncedViaAsyncWait`), which the AMD `membarFilter`
(`MembarUtility.cpp`) recognizes and skips the barrier for. This drops
`s_barrier` 22→17 and lifts per-SIMD MFMA efficiency from ~79% to ~85% at the same
VGPR count.

> [!NOTE]
> The 4-wave [`v9`](../../a16w16/v9_beyond_hotloop) sidesteps this entirely by
> using **separate** per-tile allocations, so the membar sees distinct buffer IDs
> and never inserts the barrier. v1 in this directory takes that approach instead
> — see [`v1_sliceMN_BK64_nS2`](../v1_sliceMN_BK64_nS2/README.md).

### 3.2 3× unroll → constant ring indices

The ring has 3 buffers, so unrolling the K loop by 3 makes the buffer indices
compile-time constants (0, 1, 2) instead of a runtime `tile % 3`. That removes the
loop-carried div-by-3, ring-span multiply, and ~15 wrap-around subtracts the 2×
schedule needed.

### 3.3 Pointer-walk addressing

The three unrolled regions share a single `a_base`/`b_base` and use three
**precomputed** offset sets; the base advances by `3 * BLOCK_K * stride` once per
unrolled iteration. This keeps the `wait_group` naturally at the head of the loop
body (no loop-index arithmetic ahead of it), which matters because
`WarpPipeliner` rejects pipelining if an async-wait is reached while a stage
cluster is non-empty.

### 3.4 Beyond-the-loop: v9 XCD remap

`get_pids` (in `../common.py`) is copied verbatim from the 4-wave
[`v9`](../../a16w16/v9_beyond_hotloop/README.md#3-l2-cache-locality): XCD-aware PID
remapping + `GROUP_SIZE_M=4` swizzle for L2 locality. This is active at the 256-tile
4096² grid and cut measured VMEM latency substantially.

## 4. Performance

MI350X, gfx950, 4096×4096×8192, fp16, no-AGPR:

| Metric | Value |
|---|---|
| Correctness vs torch | ✅ PASS (K 512…16384, fp16 + bf16) |
| rocprof TFLOPS (cold, rotating) | ~912 |
| MFMA efficiency (per-SIMD) | ~85% warm / ~83% cold |
| VGPRs / spills | 188 / 0 |

```bash
# correctness + do_bench TFLOPS
TRITON_HIP_AGPR_ALLOC="0,0" python ../bench.py --version 0 --K 8192 --dtype fp16

# rocprof cold-rotating TFLOPS + MFMA efficiency (ATT) + VGPR/spill
TRITON_HIP_AGPR_ALLOC="0,0" python ../collect_perf.py --version 0 --K 8192 --dtype fp16
```

## 5. What comes next

[`v1_sliceMN_BK64_nS2`](../v1_sliceMN_BK64_nS2/README.md) restructures the tile into
four M/N quadrants with separate per-quadrant LDS allocations (so the loads stay
non-relaxed with no membar barrier) at BLOCK_K=64 / 2-buffer, raising cold-rotating
throughput to ~1039 TFLOPS.
