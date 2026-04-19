# Understanding `ds_read` Throughput

When discussing `ds_read` instructions, 
it is tempting to focus on a single number: how many cycles a load takes. 
In practice, this leads to confusion, 
because two different performance questions are being mixed together:

1. How long does it take for the data to become available?
2. How often should we issue a ds_read instruction?

These correspond to **latency** and **throughput**, respectively. 
Both are important, 
but they are solved in different ways and by different parts of the kernel design.

In this tutorial, we focus on building an accurate mental model of `ds_read` throughput. 
Latency will reappear in another doc, when we discuss prefetching and pipeline depth.

## 1. The basic mental model for throughput

A `ds_read` instruction moves data through several stages before it becomes usable by the compute instructions. 
Conceptually, the data flows in the following order:

1. The instruction is issued by SQ
2. Addresses are transferred from SIMD to LDS
3. LDS services the request
4. Data is transferred from LDS back to the SIMD

Each stage has its own performance characteristics. 
The slowest stage determines the **steady-state throughput** of `ds_read`.

For concreteness, we focus on `ds_read_b128`.


### 1.1. Instruction issue: SQ is not the bottleneck

The first stage is instruction issue. 
A `ds_read` is issued by the SQ to a wave executing on a SIMD. 
Because a wave has 64 threads and each SIMD is 16-wide, 
issuing a ds_read instruction takes 4 cycles.

This is fast. 
In fact, it is fast enough that the SQ can issue `ds_read` instructions much more frequently 
than the rest of the system can handle. 
As a result, instruction issue itself is not a limiting factor for throughput. 
If nothing else applied back-pressure, the SQ would happily over-issue LDS loads.

This observation is important: any sustained throughput limit must come from downstream stages.


### 1.2. Address transfer: SIMD to LDS

Before LDS can service a `ds_read` request, the addresses must be delivered from the SIMDs to LDS. 
This address transfer is an explicit part of the dataflow and is worth understanding, 
even though it does not ultimately limit throughput in this case.

Each thread participating in a `ds_read` provides an address.
For a wave of 64 threads, a single `ds_read` instruction generates 256 bytes of address data (64 threads × 4 bytes per address).
SIMDs are grouped into SIMD pairs (SPs),
and each SP has a dedicated 128 B/cycle bus to LDS.
Each SP must send 512 bytes of address data to LDS for a single `ds_read` instruction,
which takes 4 cycles.

This address transfer happens before LDS can service the request, 
but it is fast relative to the rest of the pipeline.


### 1.3. LDS service: the first real limit

LDS is organized as 64 banks, each 4 bytes wide, 
giving a total service bandwidth of 256 bytes per cycle.

For `ds_read_b128`, each thread requests 16 bytes.
A single `ds_read_b128` instruction therefore requests 1024 bytes (64 threads × 16 bytes),
and when all four SIMDs in a compute unit issue a `ds_read_b128` instruction, LDS must serve a total of 4096 bytes.

At 256 bytes per cycle, LDS requires 16 cycles to service these requests. 
This immediately tells us that LDS cannot support issuing `ds_read_b128` instructions 
at a rate faster than one per 16 cycles in steady state.

At this point, LDS already looks like a throughput limiter.


### 1.4. Data return: bus bandwidth matches LDS

After LDS services the request, data must be delivered back to the SIMDs. 
The bus connecting each SP and LDS is bi-directional, with a bandwidth of 128 bytes per cycle in each direction. 
Address traffic travels from SP to LDS, while data traffic travels from LDS back to SP, 
and the two directions do not contend with each other.

Each SP serves two SIMDs, so for `ds_read_b128`, an SP must receive 2048 bytes. 
With a bandwidth of 128 bytes per cycle, this transfer also takes 16 cycles.

This matches the LDS service rate exactly. 
In other words, LDS and the LDS-to-SIMD bus are balanced, 
and together they define the steady-state throughput of `ds_read_b128`.


### 1.5. Steady-state throughput of `ds_read_b128`

Putting all stages together, we arrive at a clear picture:

- SQ can issue `ds_read` instructions quickly and is not limiting
- Addresses take 4 cycles to transfer from SIMD to LDS
- LDS can service one `ds_read_b128` per SIMD every 16 cycles
- The LDS-to-SIMD bus can deliver data at the same rate

Therefore, in steady state:

> [!IMPORTANT]
> A SIMD should expect to issue one `ds_read_b128` every 16 cycles.

This number describes **throughput**, not latency.


### 1.6. Transient behavior: the SP-to-LDS FIFO

The derivation above describes steady state, but not the warmup.
In thread traces, the first several `ds_read` instructions appear to issue in 4–8 cycles,
and only afterward do subsequent `ds_read`s settle into the 16-cycle-per-instruction rate predicted by the throughput model.
This prefix is the visible signature of a FIFO queue sitting between instruction issue and LDS service.

Each SIMD pair has its own 8-entry FIFO that buffers outstanding `ds_read` requests to LDS.
Because a CU contains two SIMD pairs, the CU-level capacity is up to 16 `ds_read`s in flight at once.
While there is room in the FIFO, SQ issues at its own 4-cycle rate and the SIMD does not wait for LDS to complete the previous request.
Only when the FIFO fills does back-pressure force the issue rate down to the LDS service rate.

The same FIFO also buffers `ds_write` requests, not just `ds_read`s.
A slow `ds_write` at the head of the queue can block subsequent `ds_read`s behind it —
a detail that matters for kernels that mix the two,
see [a4w4 §3.5–3.6](../kernels/gemm/a4w4/README.md#35-where-to-place-lwlr-for-scales) for a worked example where this dynamic drives the pipeline design.

With SQ issuing every 4 cycles and LDS completing one `ds_read_b128` every 16 cycles,
the FIFO grows by one slot per 4 cycles and drains by one slot per 16 cycles — a net fill rate of three slots per 16 cycles.
Starting from empty, the FIFO reaches capacity after roughly ten instructions,
after which the steady-state model from §1.5 takes over.
The prefix observed in traces is a handful of instructions, not a sharp cutoff at instruction 9.

This has a practical consequence for clustered LDS accesses:
the first handful of `ds_read`s in a cluster are effectively free relative to the steady-state model,
but layouts that enlarge a cluster beyond what the FIFO absorbs start paying the throughput cost.
Interleaving non-LDS work into a cluster lets the FIFO drain and extends the free-burst regime further.

> [!NOTE]
> The FIFO depth is a hardware parameter, not a fundamental limit.
> Deeper FIFOs would absorb larger bursts and shrink the prefix regime —
> one of the expected directions for future LDS designs (see §6).


## 2. Why this matters for instruction scheduling

Once the throughput of `ds_read_b128` is understood, 
instruction scheduling becomes much more systematic. 
If a `ds_read_b128` can be issued once every 16 cycles, 
then issuing it more frequently does not help. 
Those cycles are better spent on computation.

This naturally leads to interleaving patterns such as pairing one `ds_read_b128` with one 16-cycle MFMA instruction, or two `ds_read_b128` with one 32-cycle MFMA instruction.
While LDS is busy transferring data, the SIMD performs useful compute. 
When done correctly, latency is hidden and throughput is fully utilized.

This idea - separating latency hiding from throughput saturation - will reappear throughout the kernel optimization journey and is central to building high-performance LDS-heavy kernels.


## 3. `ds_read_b64` vs `ds_read_b128`: same bandwidth, different trade-offs

So far, we have used `ds_read_b128` as the primary example. 
The same mental model, however, also applies to `ds_read_b64`, 
and comparing the two helps clarify what really matters for LDS access.

For `ds_read_b64`, each thread loads 64 bits, or 8 bytes.
A single `ds_read_b64` instruction therefore requests 512 bytes (64 threads × 8 bytes),
and when all four SIMDs in a compute unit issue a `ds_read_b64` instruction, LDS must serve a total of 2048 bytes.
With an LDS service rate of 256 bytes per cycle, the LDS can service these requests in 8 cycles. 
The LDS-to-SIMD bus bandwidth leads to the same result.

As a consequence, in steady state:

- `ds_read_b128` has a throughput of **one instruction every 16 cycles**
- `ds_read_b64` has a throughput of **one instruction every 8 cycles**

From a pure throughput perspective, these two options are equivalent in an important sense:
Issuing one `ds_read_b128` is exactly equivalent to issuing two `ds_read_b64` instructions in steady state.
Both transfer the same total amount of data from LDS, 
and both fully saturate the LDS system when issued at their respective rates.

### 3.1. Why the choice still matters

Even though the bandwidth is the same, 
`ds_read_b64` and `ds_read_b128` differ in how they interact with the rest of the kernel.

From the instruction issue perspective, `ds_read_b128` has a clear advantage: 
it moves more data per instruction. 
Fewer LDS instructions means less pressure on the SQ and fewer instructions to schedule, 
which is generally beneficial for code generation.

From the latency perspective, however, `ds_read_b64` completes sooner. 
In steady state, its data returns 8 cycles earlier than `ds_read_b128`. 
This shorter latency can make it easier to hide LDS access behind computation if the kernel does not explicitly prefetch LDS data.

This leads to an important practical guideline:

> [!TIP]
> If the kernel implements LDS prefetching and the data is requested well before it is used, 
> then latency is no longer the dominant concern. 
> In that case, `ds_read_b128` is usually the better choice, 
> because it reduces instruction count while achieving the same throughput.
>
> On the other hand, if LDS data is consumed shortly after it is requested and no explicit prefetching is in place, 
> `ds_read_b64` may provide more flexibility. 
> Its shorter latency gives the scheduler more opportunities to overlap the load with computation, 
> even though it requires issuing more instructions overall.

This trade-off is not theoretical -
it shows up directly when tuning kernels that are transitioning from naive LDS usage to pipelined designs.


### 3.2. `ds_read_tr`: transposed reads for layout conversion

`ds_read_tr` is a third variant worth naming here.
It reads from LDS exactly like `ds_read`, but the hardware transposes the data during the transfer so that the VGPR result is in a different thread layout than the LDS source.
Its throughput profile is identical to `ds_read_b*` — the same SP-to-LDS pipeline, the same bank-conflict rules, and the same 8-entry FIFO — so everything in this doc applies unchanged.
The choice to use it is a layout-conversion decision, not a throughput decision.
The MXFP4 kernel uses `ds_read_tr` for the LR step of its scale pipeline;
see [a4w4 §2.5](../kernels/gemm/a4w4/README.md#25-ds_read_tr-hardware-assisted-layout-conversion-for-scales) for the full treatment.


## 4. LDS bank conflicts and their impact on throughput

The LDS service rates discussed so far describe the peak capability of the hardware.
Achieving this peak requires that LDS accesses be bank-conflict free.
In practice, this assumption does not always hold, and when it does not,
the effect on performance can be dramatic.

For a practical demonstration of how layout design affects bank conflicts,
see the [v3_lds kernel tutorial](../kernels/gemm/a16w16/v3_lds/README.md).

LDS is organized into 64 banks, each servicing one 4-byte access per cycle. 
When a `ds_read` instruction reaches LDS, the 64 threads in a wave are not serviced individually. 
Instead, threads are grouped, and LDS services requests group by group in multiple phases. 
Threads within the same group access LDS simultaneously, 
and if multiple threads in a group target the same bank, those accesses must be serialized.

The exact grouping of threads and the phasing behavior of LDS accesses are hardware-defined 
and somewhat subtle. 
A detailed description can be found in the [AMD GPU kernel optimization guide](https://github.com/nod-ai/amd-shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md).
What matters for performance is the consequence: 
bank conflicts reduce the effective LDS service rate.

In the ideal case, each thread in a group accesses a different bank. 
LDS can then sustain its peak service bandwidth, 
and the steady-state throughput of `ds_read_b128` is one instruction every 16 cycles, 
as derived earlier.

When bank conflicts are present, LDS must serialize conflicting accesses. 
For example, in the case of a two-way bank conflict, 
each conflicting access requires two phases instead of one. 
As a result, the LDS service time doubles from 16 cycles to 32 cycles. 
The entire `ds_read_b128` pipeline slows down accordingly.

From the perspective of instruction throughput, this has a direct and observable effect:

- Without bank conflicts, `ds_read_b128` can be issued once every 16 cycles in steady state.
- With a two-way bank conflict, the steady-state issue interval increases to 32 cycles.

This leads to an important diagnostic rule of thumb:

> [!TIP]
> If a `ds_read_b128` instruction takes longer than 16 cycles to issue in steady state, 
> bank conflicts must be present.

Avoiding bank conflicts is therefore a prerequisite for achieving the theoretical throughput discussed earlier. 
This is one of the main reasons why LDS layout design is so critical in high-performance kernels. 
It is also why simply "using LDS" is not enough. 
How data is laid out and accessed matters just as much as where it is stored.

In the context of Gluon kernels, this observation reinforces a central theme of this tutorial: 
learning Gluon is learning layouts. 
Tools that visualize layouts and access patterns are invaluable, 
not only for understanding correctness, 
but also for ensuring that LDS accesses are bank-conflict free and capable of reaching peak throughput.

> [!NOTE]
> We validate this steady-state throughput model using a microbenchmark kernel that
> issues back-to-back `ds_read_b128` instructions with controlled bank conflicts.
> The experiment and ATTViewer traces are available [here](../experiments/lds_throughput_validation).


## 5. Latency is real — but it is solved differently

None of the above implies that `ds_read` latency is unimportant. 
On the contrary, a `ds_read_b128` has a long latency, 
and if data is requested only at the moment it is needed, the kernel will stall.

The key distinction is that **latency and throughput are solved by different techniques**:

- **Latency** is addressed by prefetching: 
  issuing `ds_read` instructions early so that data is ready when it is needed. 
  This is the focus of `v5_local_prefetch`.
- **Throughput** is addressed by interleaving: 
  issuing `ds_read` instructions at the correct rate and filling the remaining cycles with other useful work.

Confusing these two leads to common mistakes, 
such as over-issuing LDS loads in an attempt to hide latency, 
which only creates back-pressure without increasing bandwidth.


## 6. A note on future LDS designs

The throughput limits discussed so far are properties of a specific hardware implementation, not of LDS as a concept.
As hardware evolves, the numbers change,
but the core reasoning — that steady-state throughput is determined by the slowest stage, and that bursts below the FIFO depth evade that limit — will keep applying as long as the pipeline structure itself is preserved.
Understanding where the slowest stages are today also makes it possible to anticipate which future-hardware changes are *parametric* (same mental model, different numbers) versus *architectural* (different mental model).

### 6.1. Parametric improvements

Most plausible near-term evolution falls into this category: same pipeline stages, better numbers.

- **LDS-to-SIMD bus bandwidth** can grow through wider buses or higher frequencies, at known area, power, and wire-delay costs.
  This is not a free lever, but it is the most direct way to raise the per-SIMD service rate.
- **Number of LDS banks** sets aggregate service bandwidth.
  More banks is not automatically fewer conflicts, though:
  conflict patterns depend on the interaction between bank count and the software's access stride,
  and without additional address hashing in the LDS controller, doubling the bank count may just move the conflict pattern rather than eliminate it.
- **SP-to-LDS FIFO depth** (§1.6) determines how long a burst of `ds_read`s can run at SQ issue rate before the steady-state throughput model kicks in.
  Deeper FIFOs help kernels with naturally bursty LDS access — unrolled loops that emit clusters of reads followed by compute — and we expect FIFO depth to grow in future generations.
- **Latency-side knobs** — shorter pipelines, hardware prefetch into LDS, asynchronous LDS — reduce the cycle cost of the stages already in the mental model without restructuring them.

Under all of these, the mental model in §§1–5 applies unchanged;
only the specific cycle counts, bank counts, burst lengths, and latencies shift.

### 6.2. Architectural tensions

Two directions are sometimes suggested as obvious wins but involve real trade-offs, not just "more is better."

- **Servicing SIMD pairs more independently.** Giving each SP a dedicated path into LDS (or its own partition) could raise throughput by reducing contention.
  But LDS exists specifically to be a cross-wave shared scratchpad inside a CU — full per-SP partitioning collapses that property and effectively turns LDS into four independent scratchpads.
  The design question is how to parallelize LDS service *without* losing the shared-scratchpad semantic:
  tiered designs, per-SP read buffers in front of a shared pool, or software-visible partitions are all candidate answers.
- **Bank-conflict remapping in hardware.** Rather than relying on the programmer to design conflict-free layouts, the LDS controller could apply address hashing so that common access patterns are automatically spread across banks.
  This is attractive but changes the contract software relies on — this repo's layout design work (v3_lds and elsewhere) assumes deterministic bank mapping, and a hashed LDS would rewrite that entire discipline.

### 6.3. The most interesting direction: format-aware LDS

The most significant architectural change worth naming is already visible in a single instruction on MI350: `ds_read_tr` (§3.2).
By transposing data during the LDS→VGPR transfer,
`ds_read_tr` collapses a layout-conversion step into an LDS read.
The MXFP4 kernel's GR → LW → LR scale round-trip ([a4w4 §2.3–§2.5](../kernels/gemm/a4w4/README.md#23-why-scales-need-an-lds-round-trip)) exists specifically to do layout conversion;
an LDS that natively handled common conversions — transpose, scale-format reshape, bank-swizzle permutation — could eliminate the round-trip entirely.
On MXFP4-class workloads, a format-aware LDS would be a bigger win than any amount of "more banks."

Unlike the parametric improvements in §6.1, format-aware LDS *does* change the mental model:
the stages no longer map one-to-one onto the pipeline this doc describes.
The software patterns this repo develops — prefetching for latency, interleaving for throughput, layout design for conflict avoidance — transfer under the parametric changes.
Only a structural shift in what LDS is *for* — from a byte-addressable scratchpad to a layout-transforming store — would require a fundamentally new set of patterns.


## 7. See Also

- [v3_lds kernel tutorial](../kernels/gemm/a16w16/v3_lds/README.md) — Practical application of the throughput model to evaluate LDS layout designs (raw, swizzling, padding).
