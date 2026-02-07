# Mental Model for Analyzing Memory-Bound Kernels on AMD GPUs

This document describes a structured way to reason about **memory-bound kernels** on AMD GPUs.
Rather than jumping directly to a single bandwidth equation, we decompose the problem into
a small number of interacting components that reflect how memory requests are actually generated,
queued, and serviced by the hardware.

The goal is to provide a **mental model** that helps answer questions such as:
- Why does increasing buffering sometimes stop helping?
- Why does higher occupancy not always increase memory bandwidth?
- Which kernel parameters matter most for latency hiding?

In this model, memory performance is not viewed primarily as a latency-hiding problem.
Instead, we focus on whether a kernel can issue memory requests at a sufficient rate to saturate the available memory bandwidth.

Memory latency matters only insofar as it limits how many requests can be outstanding at a given request issue rate. 
If requests are not issued frequently enough, the memory system remains underutilized regardless of how many waves are resident.

---

## High-Level View

At a high level, the achieved memory bandwidth is determined by three factors:

1. **How many memory requests each wave can keep in flight**
2. **How large each memory request is**
3. **How many waves can issue memory requests concurrently on a CU**

These factors are not independent. In particular, increasing the number of concurrent waves
can reduce the request issue rate of each wave due to shared compute resources.

---

## Part 1: In-Flight Memory Requests per Wave

### Concept

Each wave attempts to issue memory requests as aggressively as possible in order to
utilize available memory bandwidth. Whether this succeeds depends on how frequently
the wave can issue new requests relative to how long previous requests remain in flight.

The number of in-flight memory requests per wave is limited by:

- The latency of HBM memory (how long each request occupies the memory system)
- The request issue rate of the wave
- The number of explicitly buffered requests (software pipelining depth)

We model this as:

```
num_req_per_wave = min(hbm_latency / iter_latency, num_stages)
```


### Key Parameters

- **`hbm_latency`**
  - Hardware property obtained from the GPU specification
  - Determines how long a memory request remains outstanding

- **`iter_latency`**
  - Cycles per loop iteration for a wave
  - Determines how frequently the wave can issue memory requests
  - Depends on the amount of work assigned to the wave
  - Influenced by:
    - block size
    - number of warps per workgroup

- **`num_stages`**
  - Number of software pipeline stages (e.g., number of LDS buffers)
  - Upper bound on the number of outstanding requests per wave

### Interpretation

- If `iter_latency` is large, the wave issues memory requests too infrequently to
  fully utilize the memory system.
- If `iter_latency` is small, the wave can issue requests at a high rate and becomes
  limited by buffering (`num_stages`) or hardware request queues.
- This component answers:
  > *Can a single wave generate enough memory traffic to saturate bandwidth?*

---

## Part 2: Size of Each Memory Request

### Concept

Although memory requests are issued by individual waves, the data being requested is
naturally defined at the **workgroup level**.

All waves in a workgroup collectively request the data corresponding to a block (or tensor).
Each wave typically operates on a disjoint portion of that block. As a result, the total
amount of data requested per iteration is best viewed as a **block-level quantity**, not a
per-wave one.

### Block-Level View

At the workgroup level:

```
data_per_request_per_workgroup = block_size (or tensor size)
```


This represents the total amount of data that must be fetched from memory for one logical
request of the block.

### Per-Wave Accounting

If a workgroup contains `n` waves, then each wave is responsible for a fraction of the block.
In that case, the effective data size per request per wave is:

```
data_per_request_per_wave = block_size / n
```


This per-wave view is a convenient accounting choice that allows us to combine request size
with per-wave request counts.

### Why This Is Valid

- Waves belonging to the same workgroup are guaranteed to be scheduled on the same CU.
- Memory requests issued by these waves contribute collectively to the same CU-level
  memory traffic.
- Dividing the block size evenly across waves preserves the total amount of data requested
  while enabling a per-wave formulation.

As a result, we can reason about memory traffic either at the workgroup level or at the wave
level without changing the final bandwidth calculation.

Request size is fundamentally a workgroup-level property; 
the per-wave request size is simply a convenient way to distribute this cost across waves.

---

## Part 3: Number of Concurrent Waves per CU

### Static Resource Constraints

The number of waves that can be resident on a CU is constrained by per-wave resource usage:

- **LDS usage**
  - Depends on block size and number of buffering stages (`num_stages`)
- **VGPR usage**
  - Depends on block size and number of warps

Given hardware limits, these determine the **occupancy**: `num_waves_per_CU`

This is the familiar static occupancy calculation.

---

### The Subtle Part: SIMD Sharing and Feedback

Waves resident on the same CU do not run independently.

If multiple waves are scheduled on the same SIMD, they must time-share compute resources.
As a result, the rate at which each wave can issue memory requests is reduced.

If `x` waves share a SIMD: `effective_iter_latency = x * iter_latency`

This directly reduces the number of in-flight memory requests per wave:
```
num_req_per_wave = min(hbm_latency / effective_iter_latency, num_stages)
```

### Why This Matters

This creates a **feedback loop**:

- Increasing occupancy:
  - Increases the number of waves that could issue memory requests
  - But reduces the issue rate of each wave
- Beyond a point, higher occupancy can *reduce* total in-flight memory requests

This explains why maximizing occupancy is not always optimal for memory-bound kernels.

---

## Putting It All Together

After defining:
1. the number of in-flight memory requests,
2. the size of each request, and
3. the available concurrency,

we can now combine these pieces to estimate the memory bandwidth requested by the kernel.

---

### Workgroups per CU

Given the kernel configuration and how work is partitioned, we can determine how many
workgroups may reside on a single CU.

For memory-bound kernels, the total amount of work is often limited, and kernels typically
do not launch a large number of workgroups. As a result, it is common that:

- each CU hosts **at most one workgroup**, and
- some CUs may be idle if the total number of workgroups is small.

This simplifies the analysis, since all waves contributing to memory traffic on a CU
belong to the same workgroup.

---

### In-Flight Memory per Workgroup

From the previous sections, we already have:

- the number of in-flight memory requests per wave,
- the size of each request (accounted at the workgroup level), and
- the number of waves in the workgroup.

Combining these gives the total in-flight memory request size per workgroup:

```
inflight_bytes_per_WG =
  (in-flight requests per wave)
  × (request size per wave)
  × (number of waves per workgroup)
```


This value represents how much memory traffic the workgroup attempts to keep outstanding
at any given time.

---

### Per-CU Hardware Limit

The actual amount of in-flight memory that can be sustained by a CU is limited by hardware,
such as:

- per-CU cache capacity,
- request queue depth,
- internal buffering limits.

Therefore, the effective in-flight memory per CU is:

```
effective_inflight_bytes_per_CU = min(inflight_bytes_per_WG, CU_inflight_limit)
```


This captures the fact that even if the kernel attempts to issue more memory requests,
the hardware may not be able to accept them.

---

### Bandwidth Requested per CU

Once the effective in-flight memory size is known, the bandwidth requested by a single CU
follows directly.

The model assumes that a CU issues its in-flight memory requests every iteration cycle:

```
BW_per_CU = effective_inflight_bytes_per_CU / iter_latency
```


In other words, the CU attempts to send the in-flight amount of memory every iteration.
If this requested bandwidth exceeds what the memory system can deliver, it will be
clamped by hardware limits.

---

### Total Requested Bandwidth

Finally, the total bandwidth requested by the kernel is obtained by multiplying the
per-CU bandwidth by the number of active CUs:

```
BW_total = BW_per_CU × num_active_CU
```

This value represents the total memory bandwidth demand generated by the kernel across
the GPU.

---

### Interpretation

This formulation makes the data flow explicit:

- Kernel parameters determine how much memory a workgroup attempts to keep in flight.
- Hardware limits cap how much of that demand can be realized per CU.
- The number of active CUs scales the total bandwidth demand.

As a result, overall memory performance is determined by whether the aggregate bandwidth
requested by all active CUs can saturate the available memory system.
