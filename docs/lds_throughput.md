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

## The basic mental model for throughput

A `ds_read` instruction moves data through several stages before it becomes usable by the compute instructions. 
Conceptually, the data flows in the following order:

1. The instruction is issued by SQ
2. Addresses are transferred from SIMD to LDS
3. LDS services the request
4. Data is transferred from LDS back to the SIMD

Each stage has its own performance characteristics. 
The slowest stage determines the **steady-state throughput** of `ds_read`.

For concreteness, we focus on `ds_read_b128`.


### Instruction issue: SQ is not the bottleneck

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


### Address transfer: SIMD to LDS

Before LDS can service a `ds_read` request, the addresses must be delivered from the SIMDs to LDS. 
This address transfer is an explicit part of the dataflow and is worth understanding, 
even though it does not ultimately limit throughput in this case.

Each thread participating in a `ds_read` provides an address. 
For a wave of 64 threads, this results in 256 bytes of address data per SIMD. 
SIMDs are grouped into SIMD pairs (SPs), 
and each SP has a dedicated 128 B/cycle bus to LDS.
Each SP must send 512 bytes of address data to LDS for a single `ds_read` instruction,
which takes 4 cycles.

This address transfer happens before LDS can service the request, 
but it is fast relative to the rest of the pipeline.


### LDS service: the first real limit

LDS is organized as 64 banks, each 4 bytes wide, 
giving a total service bandwidth of 256 bytes per cycle.

For `ds_read_b128`, each thread requests 16 bytes. 
A single SIMD therefore requests 1024 bytes, 
and when all four SIMDs in a compute unit participate, LDS must serve a total of 4096 bytes.

At 256 bytes per cycle, LDS requires 16 cycles to service these requests. 
This immediately tells us that LDS cannot support issuing `ds_read_b128` instructions 
at a rate faster than one per 16 cycles in steady state.

At this point, LDS already looks like a throughput limiter.


### Data return: bus bandwidth matches LDS

After LDS services the request, data must be delivered back to the SIMDs. 
The bus connecting each SP and LDS is bi-directional, with a bandwidth of 128 bytes per cycle in each direction. 
Address traffic travels from SP to LDS, while data traffic travels from LDS back to SP, 
and the two directions do not contend with each other.

Each SP serves two SIMDs, so for `ds_read_b128`, an SP must receive 2048 bytes. 
With a bandwidth of 128 bytes per cycle, this transfer also takes 16 cycles.

This matches the LDS service rate exactly. 
In other words, LDS and the LDS-to-SIMD bus are balanced, 
and together they define the steady-state throughput of `ds_read_b128`.


### Steady-state throughput of `ds_read_b128`

Putting all stages together, we arrive at a clear picture:

- SQ can issue `ds_read` instructions quickly and is not limiting
- Addresses take 4 cycles to transfer from SIMD to LDS
- LDS can service one `ds_read_b128` per SIMD every 16 cycles
- The LDS-to-SIMD bus can deliver data at the same rate

Therefore, in steady state:

> [!IMPORTANT]
> A SIMD should expect to issue one `ds_read_b128` every 16 cycles.

This number describes **throughput**, not latency.


## Why this matters for instruction scheduling

Once the throughput of `ds_read_b128` is understood, 
instruction scheduling becomes much more systematic. 
If a `ds_read_b128` can be issued once every 16 cycles, 
then issuing it more frequently does not help. 
Those cycles are better spent on computation.

This naturally leads to interleaving patterns such as pairing one `ds_read_b128` with one 16-cycle MFMA instruction, or two `ds_read_b128` with one 32-cycle MFMA instruction.
While LDS is busy transferring data, the SIMD performs useful compute. 
When done correctly, latency is hidden and throughput is fully utilized.

This idea - separating latency hiding from throughput saturation - will reappear throughout the kernel optimization journey and is central to building high-performance LDS-heavy kernels.


## `ds_read_b64` vs `ds_read_b128`: same bandwidth, different trade-offs

So far, we have used `ds_read_b128` as the primary example. 
The same mental model, however, also applies to `ds_read_b64`, 
and comparing the two helps clarify what really matters for LDS access.

For `ds_read_b64`, each thread loads 64 bits, or 8 bytes. 
A single SIMD therefore requests 512 bytes, 
and all four SIMDs together request 2048 bytes. 
With an LDS service rate of 256 bytes per cycle, the LDS can service these requests in 8 cycles. 
The LDS-to-SIMD bus bandwidth leads to the same result.

As a consequence, in steady state:

- `ds_read_b128` has a throughput of **one instruction every 16 cycles**
- `ds_read_b64` has a throughput of **one instruction every 8 cycles**

From a pure throughput perspective, these two options are equivalent in an important sense:
Issuing one `ds_read_b128` is exactly equivalent to issuing two `ds_read_b64` instructions in steady state.
Both transfer the same total amount of data from LDS, 
and both fully saturate the LDS system when issued at their respective rates.

### Why the choice still matters

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


## LDS bank conflicts and their impact on throughput

The LDS service rates discussed so far describe the peak capability of the hardware. 
Achieving this peak requires that LDS accesses be bank-conflict free. 
In practice, this assumption does not always hold, and when it does not, 
the effect on performance can be dramatic.

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


## Latency is real — but it is solved differently

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


## A note on future LDS designs

The throughput limits discussed so far are not fundamental properties of LDS as a concept; 
they are properties of a specific hardware implementation.

In the current model, increasing `ds_read` throughput requires improvements in two areas simultaneously: 
the bandwidth of the LDS-to-SIMD buses and the service rate of LDS itself. 
Bus bandwidth is relatively straightforward to scale over time through wider buses or higher frequencies.

LDS service rate is more subtle. 
There are two obvious directions for improvement. 
One is to increase the number of banks, which increases the total service bandwidth. 
Another is to allow LDS to service requests from multiple SIMD pairs more independently, 
reducing contention between them.

How these ideas are realized, or whether different approaches are taken entirely,
is a design choice for future GPU generations. 
As hardware evolves, the exact numbers will change, 
but the mental model remains the same: 
steady-state throughput is determined by the slowest stage in the data path.

Understanding that model today makes it much easier to reason about performance tomorrow.
