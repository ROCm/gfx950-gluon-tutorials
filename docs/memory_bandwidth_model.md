# Memory Bandwidth Model for AMD GFX9 Kernels

This document describes a structured way to reason about **memory-bound kernels** on AMD GPUs.
Rather than jumping directly to a single bandwidth equation, we decompose the problem into
a small number of interacting components that reflect how memory requests are actually generated,
queued, and serviced by the hardware.

The goal is to provide a **mental model** that helps answer questions such as:
- Why does increasing buffering sometimes stop helping?
- Why does higher occupancy not always increase memory bandwidth?
- Which kernel parameters matter most for latency hiding?

In this model, memory performance is not viewed primarily as a latency-hiding problem.
Instead, we focus on whether a kernel can issue memory requests at a sufficient rate
to saturate the available memory bandwidth.

Memory latency matters only insofar as it limits how many requests can be outstanding at a given request issue rate. 
If requests are not issued frequently enough, the memory system remains underutilized regardless of how many waves are resident.

---

## 1. High-Level View

There are two complementary perspectives for analyzing the achieved memory bandwidth of a kernel:

### 1.1 Steady-State Analysis

Focus on the main loop of the kernel — the steady state.
Given the kernel's memory access pattern, compute resources, and buffering strategy,
how much bandwidth can the kernel sustain per cycle?

This is the approach developed in detail below. It decomposes the achieved bandwidth
into three interacting factors:

1. **How many memory requests each wave can keep in flight**
2. **How large each memory request is**
3. **How many waves can issue memory requests concurrently on a CU**

These factors are not independent. In particular, increasing the number of concurrent waves
can reduce the request issue rate of each wave due to shared compute resources.

### 1.2 End-to-End (E2E) Analysis

Focus on the entire kernel execution — not just the loop.
From the problem size, we know exactly how many bytes the kernel must transfer.
The achieved bandwidth is then:

```
BW_achieved = total_bytes_transferred / total_execution_cycles
```

The total execution cycles include not only the steady-state loop,
but also the prologue (pipeline fill) and epilogue (drain and store).
For large problems, the prologue and epilogue are amortized and the steady-state
analysis dominates. For smaller problems or kernels with expensive epilogues,
the E2E view can reveal bottlenecks that steady-state analysis alone would miss.

Both perspectives are useful. The steady-state model tells you the *peak sustainable bandwidth*
of your loop design. The E2E model tells you the *actual bandwidth* you observe on a given problem size,
and helps identify whether non-loop overhead is significant.

---

## 2. Steady-State Analysis

The steady-state analysis focuses on the main loop of the kernel.
Given the kernel's memory access pattern, compute resources, and buffering strategy,
how much bandwidth can the kernel sustain per cycle?

We decompose this into three factors — in-flight request count, request size, and
wave concurrency — then combine them to estimate the total bandwidth.

### 2.1 In-Flight Memory Requests per Wave

Each wave issues one memory request per loop iteration. The number of requests it can
keep simultaneously in flight depends on how quickly it issues new requests relative to
how long each request takes to complete:

```
num_req_per_wave = min(hbm_latency / compute_latency,
                       num_stages * hbm_latency / (hbm_latency + compute_latency))
```

Note that `num_req_per_wave` is the *average* number of in-flight memory requests over
a period of `hbm_latency`. During that window, the wave issues
`hbm_latency / compute_latency` requests (if not buffer-limited), and each request
remains outstanding for `hbm_latency` cycles. The average occupancy of the memory
pipeline is therefore this ratio, capped by the available buffers.

The first term (`hbm_latency / compute_latency`) is the issue-rate limit: the wave
can issue one request every `compute_latency` cycles, and each request stays in flight
for `hbm_latency` cycles.

The second term is the buffer limit. Each of the `num_stages` buffers cycles through
two phases: being filled by an in-flight memory request (`hbm_latency` cycles) and
being consumed by compute (`compute_latency` cycles). The total cycle time per buffer
is `hbm_latency + compute_latency`. The fraction of time each buffer has an in-flight
request is `hbm_latency / (hbm_latency + compute_latency)`. With `num_stages` buffers,
the average number that can hold in-flight requests at any moment is:

```
buffer_limit = num_stages * hbm_latency / (hbm_latency + compute_latency)
```

Note that when `compute_latency` is small relative to `hbm_latency`, the buffer limit
approaches `num_stages` — almost all buffers are being filled at any given time because
consumption is nearly instantaneous. Conversely, when `compute_latency` is comparable to
`hbm_latency`, the buffer limit approaches `num_stages / 2` — each buffer spends roughly
half its time being consumed and half being filled.

The three parameters here are:

- **`hbm_latency`** — how long a memory request remains outstanding (hardware property).
- **`compute_latency`** — cycles required to execute the compute work in one iteration,
  including LDS reads, MFMA, and any other non-memory instructions. This is not the same
  as the total loop iteration time, which may be longer if the loop is HBM-latency bound.
  `compute_latency` determines the fastest rate at which a wave *could* issue new memory
  requests.
- **`num_stages`** — number of software pipeline buffers (e.g., LDS buffers).

When `compute_latency` is large relative to `hbm_latency`, the wave issues requests too
infrequently to fill the memory pipeline. When `compute_latency` is small, the wave can
issue at a high rate but becomes limited by the number of available buffers.

### 2.2 Request Size

Although memory requests are issued by individual waves, the data being fetched is
defined at the workgroup level: all waves in a workgroup collectively load one or more
blocks per iteration. We define `block_size` as the total size of all blocks that one
workgroup loads per iteration. Each wave handles a disjoint fraction:

```
data_per_request_per_wave = block_size / num_waves_per_workgroup
```

This per-wave accounting is valid because all waves in a workgroup are co-scheduled
on the same CU, so their memory requests contribute collectively to the same CU-level
traffic. The split is purely a bookkeeping convenience — the total data per iteration
remains `block_size`.

### 2.3 Concurrent Waves and SIMD Sharing

The number of resident waves on a CU (occupancy) is constrained by LDS and VGPR usage.
However, occupancy alone does not determine bandwidth — waves sharing a SIMD must
time-share its compute resources, which slows down each wave's issue rate.

If `waves_per_simd` waves share a SIMD, each wave's effective compute time becomes:

```
effective_compute_latency = waves_per_simd × compute_latency
```

This feeds back into the in-flight count:

```
num_req_per_wave = min(hbm_latency / effective_compute_latency,
                       num_stages * hbm_latency / (hbm_latency + effective_compute_latency))
```

This creates a tension: increasing occupancy adds more waves but slows each one down.
Beyond a certain point, higher occupancy can actually *reduce* the total number of
in-flight requests. This is why maximizing occupancy is not always optimal for
memory-bound kernels.

### 2.4 Combining the Pieces

The total in-flight memory per CU depends on how many waves are actively resident.
The number of active waves per CU, `num_active_waves_per_CU`, is determined by the
resource usage of each workgroup — registers (VGPRs) and LDS — together with the
number of waves per workgroup and the CU's hardware limits.

Using `num_req_per_wave` with `effective_compute_latency` from Section 2.3:

```
inflight_bytes_per_CU =
  num_req_per_wave × data_per_request_per_wave × num_active_waves_per_CU
```

This is capped by a hardware limit. On GFX9, all memory requests pass through the
L1 cache, also known as TCP (Texture Cache per Pipe), which is 32 KB per CU.
This means at most 32 KB of memory requests can be in flight at any moment:

```
CU_inflight_limit = 32 KB  (TCP size on GFX9)
effective_inflight_bytes_per_CU = min(inflight_bytes_per_CU, CU_inflight_limit)
```

The bandwidth requested by the kernel follows from Little's Law — the in-flight bytes
divided by the time each request spends in the memory system:

```
BW_per_CU = effective_inflight_bytes_per_CU / hbm_latency
```

The kernel is launched with a grid of workgroups, which are distributed to CUs in a
round-robin manner. On GFX950, there are 256 CUs. The number of active CUs is simply:

```
num_active_CUs = min(num_workgroups, num_CUs)    (num_CUs = 256 on GFX950)
BW_total       = BW_per_CU × num_active_CUs
```

If this total exceeds the available memory bandwidth, the kernel saturates the memory
system. If it falls short, the kernel is underutilizing memory and the gap points to
which factor — issue rate, buffering, or concurrency — is the bottleneck.

---

## 3. End-to-End (E2E) Analysis

The E2E analysis calculates the achieved bandwidth from the total bytes transferred
and the total execution cycles of the waves on a CU:

```
BW_achieved = total_bytes / total_cycles
```

`total_bytes` is determined by the problem size and how work is partitioned among
workgroups and waves — it is a property of the kernel launch configuration.

`num_active_waves_per_CU` is determined by resource constraints as discussed in
Section 2.4.

The key question is: how many cycles does it take for `num_active_waves_per_CU` to
finish execution?

### 3.1 Total Execution Cycles

The total execution cycles consist of three stages:

**Prologue** — The wave must issue the first memory load to fill the pipeline.
The data is not available until it returns from HBM, so the prologue costs:

```
prologue_cycles = hbm_latency
```

**Steady state** — Each iteration, the wave performs compute and waits for
memory data that was requested several iterations ahead. The per-iteration
latency is:

```
iter_latency = max(hbm_latency / effective_pipeline_depth, effective_compute_latency)
```

where `effective_pipeline_depth` accounts for the TCP size limit (Section 2.4).
The kernel has `num_stages * hbm_latency / (hbm_latency + effective_compute_latency)`
buffers available on average for in-flight requests per wave (Section 2.1),
but the total in-flight bytes across all waves on a CU cannot exceed the TCP capacity:

```
effective_pipeline_depth = min(
    num_stages * hbm_latency / (hbm_latency + effective_compute_latency),
    floor(CU_inflight_limit / (num_active_waves_per_CU × data_per_request_per_wave))
)
```

The two terms in `iter_latency` reflect two possible bottlenecks. Each iteration,
the wave must do its computation, which takes at least `effective_compute_latency`
cycles. But the wave is also waiting for memory data that was issued
`effective_pipeline_depth` iterations ago. If that data has not yet returned —
i.e., if `hbm_latency` is large relative to the effective pipeline depth — then
the wave stalls until the data arrives. The iteration cannot begin until the
awaited buffer is ready.

For `num_iters` loop iterations, the steady-state cost is:

```
steady_state_cycles = num_iters × iter_latency
```

**Epilogue** — The wave stores the final result back to HBM. This store must
complete before the wave finishes, so the epilogue costs:

```
epilogue_cycles = hbm_latency
```

### 3.2 Putting It Together

The total execution cycles per CU are:

```
total_cycles = prologue_cycles + steady_state_cycles + epilogue_cycles
             = hbm_latency + num_iters × iter_latency + hbm_latency
```

And the achieved bandwidth:

```
BW_achieved = total_bytes / total_cycles
```

For large problems (`num_iters` is large), the prologue and epilogue are
amortized and `BW_achieved` converges to the steady-state bandwidth from
Section 2. To see why, note that per CU:

```
total_bytes_per_CU = num_iters × data_per_request_per_wave × num_active_waves_per_CU
```

As `num_iters → ∞`:

```
BW_achieved ≈ total_bytes_per_CU / (num_iters × iter_latency)
            = data_per_request_per_wave × num_active_waves_per_CU / iter_latency
```

In the compute-bound case (`iter_latency = effective_compute_latency`), this equals
`effective_inflight_bytes_per_CU / hbm_latency` from Section 2.4, since
`num_req_per_wave = hbm_latency / effective_compute_latency`. In the memory-bound
case (`iter_latency = hbm_latency / effective_pipeline_depth`), it likewise reduces
to the same steady-state formula with `num_req_per_wave = effective_pipeline_depth`.

For smaller problems or kernels with deep pipelines, the fixed overhead of prologue
and epilogue becomes significant and the E2E view reveals the actual performance gap.

---

## 4. TCP Utilization Efficiency and Redundant Data

The analysis in Sections 2 and 3 treats all in-flight bytes equally, but not all
bytes contribute equally to useful work. In many kernels — particularly GEMMs with
skinny M — some of the data loaded per iteration is **redundant** across workgroups,
while other data is **unique**. The TCP capacity is shared between both, so redundant
data reduces the fraction of TCP available for useful work.

### 4.1 Redundant vs. Unique Data in GEMM

In a GEMM C = A × B with shape [M, N] = [M, K] × [K, N], each workgroup computes
a tile of C of size [BLOCK_M, BLOCK_N]. Per K-loop iteration, it loads:

- **A tile** [BLOCK_M, BLOCK_K]: shared across all workgroups along the N dimension.
  Every workgroup loads the same A data, making it redundant.
- **B tile** [BLOCK_K, BLOCK_N]: unique per workgroup. Each workgroup loads a
  distinct slice of B.

The total HBM traffic is:

```
actual_hbm_bytes = num_workgroups × (A_tile_bytes + B_tile_bytes) × num_iters
                 + num_workgroups × C_tile_bytes
```

But the unique problem bytes are:

```
unique_bytes = M × K × elem_bytes + K × N × elem_bytes + M × N × elem_bytes
```

The ratio between these defines the **data reuse efficiency**:

```
reuse_efficiency = unique_bytes / actual_hbm_bytes
```

When reporting achieved bandwidth as `unique_bytes / time`, the kernel can never
exceed `peak_hbm_bandwidth × reuse_efficiency`.

### 4.2 TCP Utilization Efficiency

Since the TCP (32 KB on GFX9) limits the total in-flight bytes per CU, and both
redundant and unique data compete for this capacity, only a fraction of TCP
is doing useful work:

```
tcp_efficiency = B_tile_bytes / (A_tile_bytes + B_tile_bytes)
```

This creates a direct relationship between tile shape and achievable bandwidth.
Increasing `BLOCK_N` (relative to `BLOCK_K`) shifts bytes from the redundant A tile
to the unique B tile, improving TCP efficiency:

| BLOCK_K | BLOCK_N | A tile | B tile | TCP efficiency |
|---------|---------|--------|--------|----------------|
| 128     | 32      | 8 KB   | 8 KB   | 50%            |
| 128     | 64      | 8 KB   | 16 KB  | 67%            |
| 64      | 128     | 4 KB   | 16 KB  | 80%            |
| 64      | 256     | 4 KB   | 32 KB  | 89%            |

However, increasing `BLOCK_N` has a cost: it reduces the number of workgroups
for a given problem size (`num_workgroups = N / BLOCK_N`), which may leave CUs
idle. This creates a **three-way tradeoff**:

1. **TCP efficiency** — wants large BLOCK_N / small BLOCK_K
2. **CU utilization** — wants many workgroups (small BLOCK_N or large N)
3. **Minimum load width** — BLOCK_K must be large enough for each thread to
   issue at least one full-width memory load (e.g., `buffer_load_dwordx4`
   requires 8 elements per thread, so `BLOCK_M × BLOCK_K / num_threads ≥ 8`)

### 4.3 Strategies for Improving TCP Efficiency Without Reducing CU Utilization

The tradeoff in Section 4.2 arises because each workgroup processes exactly one
output tile. Several strategies can break this coupling:

**A-reuse via inner N-tile loop.** Within each K iteration, load the A tile once
into registers, then process multiple B tiles sequentially:

```
for k in range(K / BLOCK_K):
    a = load_A(k)                        # load once
    for t in range(N_TILES_PER_WG):
        b = load_B(k, base_n + t)        # load each B tile
        acc[t] = mfma(a, b, acc[t])      # reuse a registers
```

This amortizes the A tile load over `N_TILES_PER_WG` B tile loads, giving an
effective TCP efficiency of:

```
tcp_efficiency = (N_TILES × B_tile) / (A_tile + N_TILES × B_tile)
```

which approaches 100% as `N_TILES` grows. The cost is additional accumulator
registers — each N-tile requires `BLOCK_M × BLOCK_N × 4` bytes of fp32 VGPRs.
The number of workgroups also decreases by `N_TILES_PER_WG`, requiring
proportionally larger N to fill all CUs.

**Split-K.** Partition the K dimension across multiple workgroups. Each workgroup
computes a partial sum over a subset of K, and a final reduction combines them.
This increases the workgroup count without changing BLOCK_N, allowing both high
TCP efficiency (large BLOCK_N) and high CU utilization. The cost is the reduction
overhead and additional global memory traffic for partial sums.

### 4.4 Integrating TCP Efficiency into the Bandwidth Model

The steady-state bandwidth from Section 2.4 gives the raw HBM bandwidth the kernel
can sustain. To predict the *useful* (unique-bytes) bandwidth, multiply by the
data reuse efficiency:

```
BW_useful = BW_raw × reuse_efficiency
```

where `reuse_efficiency` accounts for redundant A loads across workgroups.
Equivalently, in the E2E model, `total_bytes` should be the unique problem bytes
(A + B + C counted once), while `total_cycles` reflects the actual execution time
which includes cycles spent loading redundant A data. The gap between raw HBM
bandwidth and useful bandwidth is entirely determined by the A/B tile ratio and
the number of workgroups sharing the same A data.
