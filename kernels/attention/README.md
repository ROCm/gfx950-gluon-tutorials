# Flash Attention on gfx950 (CDNA4) in Gluon

Forward flash-attention kernels for gfx950 (MI350 / MI355X), written in Triton Experimental
**Gluon**. Two kernels, `fav3` and `fav4`, share one pipeline architecture and differ in how
they handle the softmax rescale.

| `B=32, S=8192, H=8, D=128`, bf16, non-causal, MI355X | TFLOPS | MFMA eff / SIMD |
|---|---:|---:|
| **`fav4`** — lazy rescale, tuned | **1318** | **94.5%** |
| `fav3` — eager rescale, tuned | 1243 | 86.2% |
| *ROCm/FlyDSL* at its own published config | *1320* | *84.7%* |
| `fav4` — stock LLVM, no plugin, no env | 1198 | 68.5% |

`fav4` ties the fastest published kernel for this shape while needing about 10% fewer cycles to
do it. The distance between the first row and the last — **+10.0% of throughput, +26 points of
efficiency** — is what the design work of §5 and the compiler work of §6 are worth together.
§8 is the full table with its measurement protocol, the FlyDSL comparison worked through, and
why that ranking is shape-dependent.

**Before you start.** Read [`../gemm/README.md`](../gemm/README.md) first: §3 below uses its
intra-wave / inter-wave taxonomy, and the two-wave ping-pong of `gemm/inter_wave/` is the
structure these kernels are built on. This also assumes you know the flash-attention algorithm
— the streaming softmax that carries a running max `m`, a running sum `l` and an unnormalized
accumulator `acc`, and rebases them with `alpha = exp2(m − m_new)` as the max moves. The term
to keep in mind is **`acc·alpha`**: `acc` is the largest live value in the kernel, so rescaling
it every tile is 64 vector instructions that are pure overhead whenever the row max did not
actually move. §5 is the story of removing them.

**To build and run**, from this directory:

```bash
FA_MODULE=fav4 DISABLE_LLVM_OPT=disable-machine-sink \
LLVM_PASS_PLUGIN_PATH=$PWD/../../plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
python bench.py --seqlen 16320
```

Those three variables are not tuning knobs — they are what §6 is about, and dropping them
measures the stock-LLVM row above instead.

---

The GEMM tutorial in [`../gemm/`](../gemm/README.md) asks where scheduling intelligence should
live, and answers it for a kernel with **two** kinds of instruction competing for a SIMD:
`mfma` that computes, and `buffer_load`/`ds_read` that prepare operands. Put them in different
`warp_pipeline_stage`s, run two waves phase-offset, and the matrix pipe never idles.

Attention has **three**. Between its two MFMA chains sits a softmax — row max, `exp2`, row sum,
and a rescale of the accumulator — that is neither memory nor matrix work, and that the next
MFMA chain depends on. So:

> **Where does the third one go?**

That question generates this entire document. Answering it needs the SIMD's issue rules (§1),
gives a cycle budget to spend (§2), places attention in the GEMM tutorial's taxonomy (§3), and
then determines the loop structure (§4), the difference between the two kernels (§5), and what
the compiler has to do for them (§6). §7 is an appendix for one conflict too subtle to belong in
the main line; §8 measures the result and takes the comparison above apart, and §9 is where to
read further.

---

## 1. Three competitors, two issue ports

Start from how a CDNA SIMD issues. Two rules, and everything below is a corollary:

1. A wave issues **at most one instruction per cycle**.
2. The **VALU** and the **memory pipe** are separate issue ports, so the SIMD can issue one of
   each in the same cycle — but they must come from **different waves**, by rule 1. Note that
   LDS and VMEM *share* the memory port: a `ds_read` and a `buffer_load` cannot pair with each
   other, only with a VALU.

While an MFMA runs, those ports are free. Work issued there is free too. Work that does not fit
adds directly to the loop's cycle count. So the question is an allocation question: *the
MFMA's shadow is the resource, and there are two kinds of non-matrix work competing for it.*

There are three places the softmax could go.

### In the mem stage, with the loads

Then the VALU and the `ds_read` issue from the **same wave**, and by rule 1 a wave issues one
instruction per cycle — so they take turns.

![VALU and ds_read in one wave, taking turns](images/issue_mem_stage.svg)

Six issue slots buy three loads and three VALU. The memory port sits idle on the VALU cycles
and the VALU port sits idle on the load cycles, even though the hardware was willing to run
both at once. Half the shadow is wasted.

### In a stage of its own

Now three categories are live at the same time, which needs **three waves per SIMD**: one in
the mem stage, one in the VALU stage, one in the MFMA stage.

![three waves per SIMD, drawn from three different workgroups](images/three_waves.svg)

It is reachable, and the pairing works — the memory and the VALU do come from different waves. It
is the *third* wave that is the problem, because it has to come from a different workgroup, and
nothing can keep three workgroups in step.

### In the dot stage, with the MFMA

The VALU now issues from the **same wave as the MFMA**, while the *other* wave supplies the
memory traffic.

![VALU riding with the MFMA in one wave, memory in the other](images/issue_dot_stage.svg)

Different waves, different ports — **the VALU and the `ds_read` pair up in the same cycle.**
Six slots now buy six loads *and* six VALU: twice the work of the first option, out of the same
shadow.

That is the answer, and it is why these kernels look the way they do: **the softmax rides with
the MFMA, and has to be interleaved into it carefully enough to actually fit.**

## 2. The budget: what fits before the next MFMA can issue

The useful mental model is not "how long does an MFMA take" — nothing waits for it to finish —
but **when can the SIMD issue the next instruction**.

The public [CDNA4 ISA][isa] gives the two numbers this rests on: `PASS = 4 clock cycles`
(§7.6), and `V_MFMA_F32_32X32X16_F16` "performs 8 passes". So issue one at cycle 0 and the
**next MFMA of that shape cannot issue before cycle 32**. Meanwhile a plain VALU can issue from
cycle 8. Cycles 8–31 are therefore free real estate: **24 cycles of issue opportunity that cost
nothing**, because the matrix pipe was not going to accept anything until 32 regardless.

What can be spent there, and at what price:

| | issue cost | consequence |
|---|---|---|
| VALU (`v_fma`, `v_add`, `v_max3`, `v_cvt`…) | **4 cycles** | 6 fit in one MFMA's window |
| TRANS (`v_exp_f32`) | **8 cycles** | 3 fit — a transcendental is not un-hideable, just twice the price |
| packed f32 (`v_pk_*`) | **does not fit** | cannot be placed in the window at all; issuing one pushes the next MFMA past its 32-cycle interval |

A dot cluster of 16 MFMAs therefore has **16 × 24 = 384 cycles** to spend.

> The 32-cycle interval is derived from the public ISA as above. The per-class issue costs are
> the cost model the scheduler uses and that these kernels were measured against; the
> microarchitectural reasons behind them are not in the public document, so they are stated
> here as behaviour rather than mechanism. You do not have to take them on faith either — an
> ATT instruction trace timestamps every issue, so the cost of each class, and whether a given
> op landed inside a shadow or outside it, can be read straight off a trace of your own kernel
> ([`note.md`](note.md) has the recipe).

The packed-math row has a consequence worth pausing on, because it is counter-intuitive.
`v_pk_mul` retires two elements in one issue slot, so it is *exactly* what you want for work
that has to be exposed — and it is unusable for work you were hoping to hide. Whether to emit
packed or scalar is therefore a per-instruction decision that depends on whether that
instruction won a window slot. §6 is about making that decision.

[isa]: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf

## 3. Where attention sits in the taxonomy — a hybrid, per region

The GEMM tutorial classifies kernels as **intra-wave** (one wave per SIMD, the compiler weaves
memory into the MFMA stream) or **inter-wave** (two waves ping-pong, the overlap is structural
and needs almost no compiler help). Attention does not fit either box, and the reason is
instructive: **the taxonomy's real unit is not the kernel, it is the region.**

The discriminator is one question — *does this region need two instruction categories to issue
from the same wave?*

- **No**, one category per wave: the overlap is supplied by the ping-pong. Nothing to schedule.
- **Yes**, two categories in one wave: the overlap has to be manufactured by instruction
  ordering, and only the compiler can do it.

Read that way, the GEMM table falls out as a consequence rather than an assertion:

| | waves / SIMD | regions | region kind | scheduling model |
|---|---|---|---|---|
| `gemm/intra_wave` | 1 | `mfma` + `mem`, one wave | all intra-wave | throughput |
| `gemm/inter_wave` | 2 | `mem` \| `mfma`, split by stage | all inter-wave | none needed |
| **attention** | 2 | `mem` \| **`mfma` + VALU** | mem: inter-wave, **dot: intra-wave** | **co-execution** |

Attention is **inter-wave between memory and compute, and intra-wave inside each compute
cluster.** §1 forced both halves of that: memory has to live in the other wave so it can pair
with a VALU in one cycle, and the VALU has to live with the MFMA, which puts two categories in
one wave and hands the dot clusters to the compiler.

This also sharpens what "compiler involvement" means. It is not how *many* intra-wave regions a
kernel has — it is how hard the model for those regions is. `gemm/intra_wave` interleaves
`mfma` with memory, which is a **throughput** problem: keep the memory pipe fed far enough
ahead and the matrix pipe never starves. Attention's dot clusters interleave `mfma` with VALU,
which is a **co-execution** problem: every vector op has to be assigned to a specific MFMA's
window, in the right form, or it falls outside and costs cycles. Same "intra-wave" label, a
markedly harder question.

That distinction is exactly how the scheduler is built. The
[llirSched](../../plugins/llir_scheduler/llir_scheduler.html) plugin classifies every region
and routes it: `mfma` + `mem` to the throughput model, `mfma` + VALU to the co-execution model
of §6. Regions mixing all three do not arise here, and on gfx950 they should not: §1 showed that
putting VALU and memory in one wave wastes shadow cycles, so an FA kernel has no reason to
build such a region. It is a question for future parts, where a different issue rule could make
a three-category region worth scheduling — at which point the two models would have to be
reconciled rather than selected between.

## 4. Designing the loop

One workgroup owns a `BLOCK_M`=256-row slab of the output for one head and walks all of K/V;
the grid is `(HQ, ceil(S/BLOCK_M), B)`. Per tile there are eight things to do.

![the eight per-tile operations and their dependencies](images/tile_deps.svg)

### 4.1 Four pipeline stages, four clusters

Run them in dependency order and the matrix core idles through every copy and every LDS read,
and the memory pipe idles through all the math. The fix is the standard one — software-pipeline
the loop so that the memory for a *future* tile overlaps the matrix work of the current one.

![the four stages, filling and draining](images/pipeline.svg)

Each row is a pipeline stage and each column one loop iteration. Read down a column and the wave
is working on **four different tiles at once**: copying tile *j+3* into LDS, reading tile *j+2*
back out, running the QK chain of *j+1* and the PV chain of *j*. That column is the loop body,
and each op's tile index says how far ahead of the current tile that stage runs.

The two ends are the cost of the arrangement. Three columns at the start have stages with no tile
to work on yet, and three at the end are finishing tiles with no copies left to issue; both are
straight-line code outside the loop, which is why a trace reports them separately as the prologue
and epilogue. A four-stage pipeline costs three of each.

§1 then fixes how the slice is subdivided: every group must pair matrix work with memory so
that a wave in one kind of group always faces a wave in the other. Four clusters do it.

![the loop body split into four alternating clusters](images/clusters.svg)

`warp_pipeline_stage("…")` marks each cluster and the compiler lowers each boundary to a
barrier. With `waves_per_eu=2` the two waves on a SIMD run **one cluster apart**:

![two waves running the same four clusters, offset by one](images/pingpong.svg)

### 4.2 How the softmax is split

The softmax is cut into two groups that land in *different* dot clusters, so that neither cluster
has to hide all of it and both MFMA chains have independent vector work beside them. Where the cut
can go is not a free choice — the softmax is a single dependency chain:

![the softmax dependency chain and where it can be cut](images/softmax_dag.svg)

| | work | lands in |
|---|---|---|
| **VEC1** | the new row max, then `fma` (subtract the max) and `exp2` on the score tile | `dot2`, beside the PV MFMA |
| **VEC2** | the row sum, the downcast of `p` to fp16 for the next PV MFMA, and the accumulator rescale | `dot1`, beside the QK MFMA |

Two properties of that chain decide everything about the balance. The row max is a **reduction**,
so nothing downstream of it exists until the entire tile has been reduced — it cannot be moved,
and the subtract depends on it. And both consumers of `exp2`, the row sum and the downcast, sit on
the far side of the cut. So the chain admits one natural cut point, after `exp2`, and the split
above is that cut applied to the whole tile.

What makes the balance tunable is that the tile is a set of independent **column slices**, and each
slice can be cut in a different place. A slice cut after `exp2` leaves its subtract and exponential
in VEC1; a slice cut *before* the subtract is carried across raw and has both computed in VEC2.
§5 tunes how many slices go each way — which is balancing two clusters without ever breaking the
chain.

The consequence to hold on to while reading the code: `VEC1` runs in `dot2` on tile *j+1* while
the PV MFMA there is still working on tile *j*, so the `p` it produces is consumed a full turn
of the pipeline later. Every buffer in the kernel is sized around that skew.

### 4.3 Register budget, and why the loop is unrolled 2×

With `warps_per_cta=[8,1]` each of the 8 waves owns 32 of the 256 rows, so the row-wise softmax
reductions are per-wave — one cross-lane shuffle each, no cross-workgroup reduction. That also
sets what each wave has to hold.

![the two MFMA chains and the tile shapes one wave works on](images/tile_shapes.svg)

Each count below is one of those shapes divided by the 64 lanes of a wave, and halved again for
fp16 since two elements share a register.

| tile | shape, dtype | VGPRs / lane | buffer |
|---|---|---:|---|
| `Q` | 32 × 128 fp16 | 32×128 / 64 / 2 = **32** | live-in; loaded once, never reloaded |
| `acc` (O) | 32 × 128 fp32 | 32×128 / 64 = **64** | live-in; the running output |
| `K` via `LRK` | 128 × 64 fp16 | 128×64 / 64 / 2 = **64** | `regBuf1` |
| `V` via `LRV` | 64 × 128 fp16 | 64×128 / 64 / 2 = **64** | `regBuf1` — reuses `LRK`'s, they are never live together |
| score tile, `VEC1` | 32 × 64 fp32 | 32×64 / 64 = **32** | `regBuf2` |
| score tile, `VEC2` | 32 × 64 fp32 | 32×64 / 64 = **32** | `regBuf0` |

96 VGPRs are live for the whole loop and 128 more are the working set: **224 of 256**, with
LDS holding 2 × K + 2 × V = 64 KB. `LRK` and `DOT1` are producer and consumer of the same tile,
so they share a buffer. `VEC1` and `VEC2` are not, so they need two — and that is what forces
the unroll.

![the score buffers swap each tile, and unrolling twice puts them back](images/unroll.svg)

Over one tile the two score buffers exchange places, so a 1× loop would copy a 32-VGPR tile
every iteration just to restore the naming. Unrolling by two returns each buffer to where it
started. This is the same motivation as
[`gemm/intra_wave/a16w16/v6_loop_unroll`](../gemm/intra_wave/a16w16/v6_loop_unroll) — in both
kernels the unroll exists to make loop-carried buffers land in the same registers each time
around, not to reduce loop overhead.

## 5. `fav3` → `fav4`: getting under the budget

Start by pricing the softmax against §2's budget. Each Gluon op below expands to a fixed number
of machine instructions per wave per tile — the score tile is 32 × 64 over 64 lanes, so 32
registers, and the accumulator is 32 × 128, so 64:

| Gluon op | instructions | class | cycles | packed | cluster |
|---|---:|---|---:|---:|---|
| row max | 16 × `v_max3` | VALU | 64 | — | PV |
| subtract the max | 32 × `v_fma` | VALU | 128 | 16 × `v_pk_fma` = **64** | PV |
| `exp2` | 32 × `v_exp` | TRANS | **256** | — | PV |
| row sum | 32 × `v_add` | VALU | 128 | 16 × `v_pk_add` = **64** | QK |
| downcast `p` to fp16 | 16 × `v_cvt_pk` | VALU | 64 | already packed | QK |
| rescale the accumulator | 64 × `v_mul` | VALU | **256** | 32 × `v_pk_mul` = **128** | QK |

The packed column is the same work at half the issue cost — and it is a trap here, because a packed
op cannot go in an MFMA's shadow at all (§2). Halving the cost only helps for work that was never
going to be hidden, which is why the column matters for `fav3` and not for `fav4`, and why §6 has
to decide the two forms per instruction rather than globally. Read the cycles column as the price
of work you intend to hide.

Two items dominate, and they are the two that scale with a *whole tile* rather than with a row:
the `exp2` burst and the accumulator rescale. Totalled per cluster against the 384 cycles each
one has to spend:

![fav3's softmax priced against the shadow](images/budget_fav3.svg)

These are the Gluon-level counts. The compiled kernels land near but not exactly on them — the
backend adds address arithmetic, and `fav3`'s scheduler leaves some work packed, which halves
its instruction count — so §8's ceilings are computed from the compiled inventory rather than
from this table. The shape of the argument is the same either way.

**`fav3` is over capacity in both clusters** — 448 against 384, twice. Whatever does not fit is
issued in the open and lands directly on the loop's cycle count. Note this is a budget
statement, not a scheduling one: no ordering of these instructions can help, because there are
simply more cycles of vector work than there is shadow to hide it in.

**Lazy rescaling** is what removes the larger of the two. Softmax is shift-invariant, so the
running max does not have to be *tight*; it only has to keep `exp2`'s argument in range. `fav4`
lets it **lag**: the max is bumped, and `acc` rescaled, only when a tile's max exceeds the
running max by more than a log2 threshold of 8 — a 256× safety margin, trivially inside fp32's
range. When the max is stable, which is the common case after the first few tiles, `p` is
allowed to rise as high as 256 and the correction is skipped entirely. `acc` and `l` are carried
in the same lagging frame, so the result is unchanged.

Skipping needs a branch, and the useful granularity is the wave: each wave owns 32 rows and can
decide independently. `gl.warp_predicate` expresses exactly that — it lowers to
`s_and_saveexec` + `s_cbranch_execz`, with no cross-wave reduction and no barrier. Rows that did
not advance carry `alpha == 1`, which leaves their values unchanged — but the multiply still
issues and still costs its four cycles, which is exactly why the skip has to be a real branch
rather than a multiply by one.

That empties the QK cluster to 192 of 384 — and leaves PV untouched at 448. The kernel is now
badly unbalanced rather than uniformly over: 192 cycles of QK shadow go unused while PV pays 64
cycles for its overflow.

**Balancing them is the final piece.** Only the totals matter, so move work from PV to QK until
both fit. The candidates are the three elementwise items in `VEC1` — the row max, the subtract,
and the `exp2` — and the row max is not one of them: it is a *reduction*, so the whole tile has
to be reduced before `m_new` exists, and it is what the subtract depends on.

That leaves the subtract and the `exp2`, and **they have to move as a pair.** Moving the `exp2`
alone was tried first, and it fails in an instructive way. The subtract stays behind in PV while
its only consumer is now in the other cluster, and something then drags it out of PV into a mem
stage — the work leaves the cluster it was meant to leave, but does not arrive in QK either.

The something is **ISel's pre-RA list scheduler**, not `MachineSink`, which is worth being precise
about because `MachineSink` is already disabled (§6) and disabling it does not help. ISel's *first*
MIR dump already shows 16 of the 32 subtracts sitting in the mem region, before `MachineSink` ever
runs; re-running the same IR with `-pre-RA-sched=linearize` keeps all of them in the dot stage. The
underlying reason is the one from §6's third bullet: a pure `fsub` has no chain edge, so neither
`s_barrier` nor `sched.barrier` orders it, and a scheduler that sees its consumer downstream is
free to move it there.

Moving both ops instead removes the opportunity. The **raw** slice is carried across and subtracted
where it is exponentiated, so the subtract is born in the cluster that consumes it and there is
nothing left downstream to be pulled toward.

So the score tile is sliced along N and the subtract + `exp2` for some fraction of the slices is
computed in `dot1` instead. The fraction was found by bisection, not derived — and from here the
numbers are the **compiled** ones rather than the table's, because the margins are what the decision
turns on and the backend's own address arithmetic is part of them. Measured, the two clusters start
at 204 and 484 rather than the table's 192 and 448:

![sweeping the fraction moved from PV to QK](images/budget_balance.svg)

A half overshoots — QK goes over while PV is left with slack, the same imbalance with the sides
swapped. A quarter does not move enough: PV is still over, and QK is still holding cycles it could
have taken. **Three eighths** puts both inside with the two clusters within 8 cycles of each other,
and that is why the tile is sliced into eighths rather than halves — the granularity exists to make
the ratio adjustable.

### Design rules the final kernel embodies

| rule | why the hardware demands it |
|---|---|
| **Keep control flow out of the dot clusters.** The rescale's `warp_predicate` block lives in the `mem2` cluster, not in `VEC2` where the arithmetic belongs. | Control-flow instructions are scheduled ahead of everything else in their region, so a branch inside a dot cluster is issued *before* the first MFMA — the matrix core waits on it. Moving the block to a mem cluster puts that cost against memory latency instead, and lets `dot1` start with the QK MFMA. |
| **Balance the two dot clusters.** The score tile is sliced along N and the subtract + `exp2` for 3/8 of the slices is computed in the *other* cluster, bringing PV to 340 and QK to 348. | 204 against 484 wastes the whole of QK's slack while PV pays for the overflow. Only the totals matter, and both are made of the same elementwise work, so it can be moved. |
| **Rebalancing must cost nothing.** `gl.amd.slice` takes a register-only view of a distributed tensor: the slice keeps the source layout, so it is a partition of each lane's own registers and emits **no instructions at all**. | A distributed-tensor slice normally costs a shuffle. Free slicing is what makes the previous rule affordable, and it is a Gluon technique worth knowing well beyond attention. |

## 6. Making the compiler co-operate

The budget in §5 says the work *fits*. Getting it actually issued inside the shadow is the
compiler's job, and by §3 the dot clusters are intra-wave regions, so it needs help in three
places.

### Keeping each op in the cluster it was assigned to

§5's whole argument is an assignment of vector ops to clusters — this `exp2` belongs beside the
PV MFMA, that subtract beside the QK MFMA. `MachineSink` runs on MIR, long after any IR pass,
and undoes it: it moves an op toward its consumer, and since `VEC1` of one tile feeds `VEC2` of
the next, "toward its consumer" means *out of its cluster and into the following one*. An
`s_barrier` is no obstacle — it is `IntrNoMem`, so it is not a code-motion fence for pure ALU
ops, and a chain-free `fsub` has no dependency edge on it at all.

The result is a cluster that was carefully balanced arriving at the scheduler with its work
somewhere else, and most of a cluster's shadow left empty. Hence
`DISABLE_LLVM_OPT=disable-machine-sink`. Note this is not a scheduling decision being overridden —
it is a placement decision being *preserved* so that scheduling has something to work with.

### Packed or scalar: whose job is it?

§2's rule makes this a real decision. A packed op cannot go in the shadow, but retires two
elements per issue when it is outside. So the ideal is precise: **work that will be covered should
be scalar, and work that will be left over should be packed** — and which is which is not known
until the assignment is done.

Both kernels therefore start from **packed** math, which is what Gluon emits anyway, and the
decision is made where the budget is known. `fav3` has genuine leftovers: its over-capacity path
splits only the ops it managed to cover and leaves the remainder packed, halving their issue cost
since they are going to be exposed either way. `fav4` has no leftovers, so everything it declares
gets covered and everything gets split.

The split itself is performed by LLVM's `SIPreEmitPeephole`, which finds a packed op sitting in an
MFMA's shadow, recognizes that it cannot co-execute there, and breaks it into scalars — the
correct local decision, made at the one point where the answer is known, after scheduling has
settled what sits where.

What the scheduler has to get right for that to work is the **unit** it declares in.
`sched_group_barrier` takes a class and a count of *instructions*, so a window holding three packed
ops must be declared as three, not as the six elements they will become. Declared in instructions
the group is satisfiable whether or not the ops are still packed, and the peephole takes care of
the rest:

![a declared group of packed ops becoming issued scalars](images/packed_formation.svg)

Which is also why one kernel no longer needs a different environment from the other: nothing
upstream has to normalize the packing first.

The third panel is a detail that belongs to the kernel rather than the compiler. A packed op
cannot follow an MFMA back to back, so the *first* uncovered packed op in a cluster pays a hazard
on top of being exposed. Ordering the uncovered work ahead of the cluster's first MFMA removes
that stall — same instructions, same count, only the order differs. Nothing in the toolchain will
do it for you, because only the kernel knows which work was never going to be covered.

### Declaring the interleave

The out-of-tree **llirSched** plugin does the assignment itself. It classifies each region (§3),
and for a dot cluster it walks the vector ops against the MFMA windows, then *declares* the
result with `sched_group_barrier` — a sequence of "N instructions of this class, then M of that"
which AMDGPU's IGroupLP builds in the machine scheduler. When the region fits it spreads the work
evenly; when it does not, it covers the ops that cannot be packed first, spends what window is
left on packable ops split into scalars, and leaves the remainder packed — which is what gets
`fav3` close to its 91.4% ceiling despite being over capacity in both clusters.

Declaring rather than reordering is the load-bearing choice, and `sched_barrier` is the
alternative that does not work:

- **It is advisory.** `sched_barrier(0)` asks the machine scheduler not to move instructions
  across a point; it does not oblige it. We measured codegen consolidating a stage's last two
  sub-regions anyway. `sched_group_barrier` does not ask the scheduler to preserve an order — it
  tells IGroupLP to *construct* one.
- **It does not fence what we care about.** A barrier is a chain node, so it orders memory
  operations. Chain-free arithmetic — a pure `fsub` — has no edge to it and drifts across freely.
  This is the same property that lets `MachineSink` move `exp2` over an `s_barrier` above.
- **Pinning an order throws away the scheduler's knowledge.** Physically reordering the IR and
  fencing it fixes one order for good, and a fixed order cannot respond to the latencies and
  register pressure the machine scheduler can see. Declaring says *what* the pipeline should look
  like and leaves the scheduler to choose which instruction fills each slot.

One detail worth knowing if you read the plugin: the declaration has to be emitted **after**
every real instruction of the region, because IGroupLP forms its groups scanning upward. A
declaration at the top of a region yields empty groups and silently does nothing. The algorithm,
the region classifier and the cost model are in
[`llir_scheduler.html`](../../plugins/llir_scheduler/llir_scheduler.html).

## 7. Advanced: the LDS burst and the head of a dot cluster

Two settings in these kernels are worth about a percent each and are easy to mistake for tuning
noise. They are not — they attack the same hardware conflict from opposite ends, and the numbers
behave the way the explanation predicts.

By §1's design, a wave in a dot cluster always faces a wave in a mem cluster, and its VALU shares
each cycle with that wave's `ds_read`. That pairing is the whole point. But not every VALU is
equally cheap to pair: a **3-source** op needs more register-file read ports in its cycle than a
1- or 2-source one, and an LDS return needs the register file too. Where the two coincide, they
contend.

The dot clusters put their 3-source work exactly where the collision is worst. Read the head of a
compiled PV cluster and the first two MFMA shadows are filled entirely with `v_maximum3` — the row
max is first in dependency order, so the scheduler has nowhere else to put it — and with
`SCALE_ON_Q` off, the subtract that follows is an `fma`, also 3-source.

![the LDS burst meeting the head of a dot cluster, and the two fixes](images/lds_conflict.svg)

**`MEMNOP` moves the burst.** A couple of `s_nop`s at the head of each mem cluster delay that
wave's `ds_read`s just enough that they arrive past the `max3` block and land on the `exp2` and
sum work instead, which is 1- and 2-source. The kernel is unchanged; only the phase relationship
between the two waves moves. `MEMNOP=2` is the optimum for both kernels at either `SCALE_ON_Q`
setting — and the sweep is not smooth, which is what you would expect from a phase effect rather
than a quantity.

**`SCALE_ON_Q` removes the ops.** Folding `qk_scale` into `Q` before the loop turns every
`fma(qk, qk_scale, −m_new)` into a plain `sub`, so those stop competing for the register file at
all. It is visible in a count of 3-source VALU in `fav4`'s loop body: **98 with the fold off, 34
with it on** — and 98 − 34 = 64 is exactly the 32 subtracts of each of the two unrolled tiles. The
34 that remain are the `max3` reduction, which is the part only `MEMNOP` can help.

Measured at `B=1, S=16320, H=64, fp16` on GPU[0] — TFLOPS and in-loop MFMA efficiency per SIMD.
(A different shape and dtype from §8, so read the two tables separately; the *deltas* are what
matter here.)

| | `fav3` | `fav4` |
|---|---|---|
| no `s_nop`, no fold | 1165 / 83.9% | 1223 / 89.5% |
| `MEMNOP=2` | 1169 / 85.3% | 1233 / 92.4% |
| `MEMNOP=2` + `SCALE_ON_Q` | **1177 / 86.2%** | **1243 / 93.8%** |

Each step is worth well under a percent of throughput, and together about 1% on `fav3` and 1.6% on
`fav4`. Both kernels gain the same ~0.8% from the fold, which is the check that matters: it is the
same op count leaving the same cluster in both, so a mechanism tied to that op count should pay the
same, and it does. The efficiency column moves further than the throughput column for the reason
§8 gives — a denser MFMA stream costs clock on a power-capped part.

`SCALE_ON_Q` is not free: pre-scaling rounds `q · scale` back to fp16 before the loop, so max error
goes from 3.05e-05 to 1.22e-04 on `fav3`, and from 6.10e-05 to the same 1.22e-04 on `fav4`. All
are far inside the 1e-3 tolerance, and `--scale-on-q 0` restores the tighter numerics on either
kernel.

## 8. Results

`B=32, S=8192, H=8, D=128, bf16`, non-causal, MI355X, `rocm-smi` GPU[0] — ROCm/FlyDSL's published
benchmark shape. TFLOPS is the mean of three runs of `rocprofv3 --kernel-trace` with
`AMD_SERIALIZE_KERNEL=3`, averaging the last 100 of 1000 dispatches; MFMA efficiency is the
in-loop per-SIMD figure from an ATT instruction trace. The five configurations were run
**interleaved** — one of each, three times round — so any drift in the board hits every row
equally. Every row here is stable to about 2 TFLOPS.

| | TFLOPS | MFMA eff / SIMD |
|---|---:|---:|
| *ROCm/FlyDSL* — its own tuned config | *1320* | 84.7% |
| **`fav4`** — llirSched, `SCALE_ON_Q=1`, `MEMNOP=2` | **1318** | **94.5%** |
| **`fav3`** — llirSched, `SCALE_ON_Q=1`, `MEMNOP=2` | **1243** | 86.2% |
| `fav4` — stock LLVM, no plugin, no env | 1198 | 68.5% |
| `fav3` — stock LLVM, no plugin, no env | 1141 | 67.8% |

**What the compiler work is worth** is the distance between the tuned rows and the stock ones:
**+10.0%** of throughput on `fav4` and **+8.9%** on `fav3`, and in efficiency terms **+26.0** and
**+18.4 points**. Everything in §5 and §6 lives in that gap.

**Stock LLVM cannot tell the two kernels apart** where it counts — 68.5% against 67.8%. Without a
scheduler, `fav4`'s lazy rescale barely moves the loop at all; the 8.3-point efficiency gap
between the tuned rows is the scheduler exploiting budget headroom that lazy rescaling created,
not the algorithm on its own. A design that creates headroom only pays if something downstream
spends it.

**The ceilings from §5 still frame the tuned rows.** `fav4`'s demand fits its window, so its
ceiling is 100% and it reaches 94.5%. `fav3` leaves 4 × 48 = 192 cycles exposed per loop body
against 2048 of MFMA, so its ceiling is 2048/2240 = **91.4%** and it reaches 86.2%. The two are
8.3 points apart while their ceilings are 8.6 apart — so the whole difference is work `fav3`'s
budget cannot absorb rather than a worse schedule, and each lands within about 5.5 points of its
own ceiling.

**On the FlyDSL row.** This is its published configuration, and its published figure of 1320
TFLOPS reproduces exactly (1319.4 / 1319.6 / 1320.2). `fav4` ties it — 0.2% apart, well inside
any spread — and `fav3` is 5.8% behind.

The interesting part is *how* they tie, because the two kernels are strong in different places.
`fav4` needs about 10% fewer cycles for the same work (94.5% against 84.7%). FlyDSL spends more of
its time in the loop (94.1% against our 88.3% — its pipeline fill and drain are cheaper than our
four-stage one) and clocks **5.2% higher** at comparable power. Those cancel:

```
fav4 cycle advantage    (0.945 x 0.8833) / (0.847 x 0.9413)  = +4.8%
FlyDSL clock advantage   1650.9 MHz / 1569.8 MHz             = +5.2%
```

**And the ranking is shape-dependent, so treat one number with care.** At `B=1, S=16384` the board
runs pinned to its ~1400 W cap; there cycles are what matter and `fav4` leads FlyDSL by 9.3%. At
the shape above there is power headroom, FlyDSL's clock advantage cashes in, and the two are level.
Neither shape is *the* answer — that one flatters us, this one flatters them. The quantity that
does not move between them is the in-loop MFMA efficiency, which is the only thing in this table
the scheduler actually controls. `note.md` works the comparison through in full.

**The launch geometry is identical on both sides**, which is worth checking before trusting any
of the above — a difference in grid or occupancy would confound the whole table. FlyDSL builds its
grid as `(NUM_HEADS_Q, ceil(S/BLOCK_M), batch)` from `BLOCK_M=256`, `BLOCK_N=64`, 8 waves per
workgroup, 32 rows per wave and `waves_per_eu=2` — the same three axes in the same order, and the
same numbers, as ours. At this shape both launch `(8, 32, 32)` = 8192 workgroups of 8 waves, one
workgroup resident per CU, 256 at a time, 32 rounds with no tail imbalance. So the differences in
the table are what happens inside the loop, and power — not how the work was handed out.

Neither side remaps workgroups across XCDs here, though for different reasons. FlyDSL's GEMM and
MoE kernels do (`xcd_remap_bx_by`, behind an `xcd_swizzle` knob) but its attention kernels do not.
Ours calls `remap_xcd` on the head index — and at `HQ=8` against 8 XCDs that map is exactly the
identity, so it does nothing at this shape. It *is* active at `B=1, HQ=64`, which is one more
reason the two shapes are not interchangeable.

One caveat runs against the Gluon rows rather than for them: `S=8192` gives `n_blocks = 128`, so
`(n_blocks − 3)` is odd and both kernels run the `ODD_TAIL` path — one unpipelined tile in the
drain. FlyDSL has no such constraint.

Two things about the metrics themselves, worth knowing before quoting either:

**MFMA efficiency is derived from the cycle count**, not measured independently: it is
`mfma_count × 32 / loop_cycles` per wave, doubled for the per-SIMD figure because two waves share
the matrix unit. It is the right metric for judging a *scheduling* change, since that is exactly
what a scheduling change moves — but it carries no information a cycle count does not.

**Cycles and TFLOPS disagree, and both are honest about different things.** A denser MFMA stream
draws more power, and against a power cap that buys back a lower clock — so a change can be worth
several percent of cycles and under one percent of wall time. bf16 is the cleanest example: it
measures 3–7% faster than fp16 at *identical* cycle counts and identical in-loop efficiency, with
the whole difference coming from clock, because its narrower mantissa toggles less of the
multiplier array. Judge a scheduling change by cycles; judge a kernel by both.

**Reproducing the Gluon rows:**

```bash
cd <repo>
SHAPE="--batch 32 --seqlen 8192 --hq 8 --hk 8 --d 128"   # bf16 is the default

# tuned
HIP_VISIBLE_DEVICES=<gpu> FA_MODULE=fav4 LLIRSCHED_WP_MEMNOP=2 \
DISABLE_LLVM_OPT=disable-machine-sink \
LLVM_PASS_PLUGIN_PATH=$PWD/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
python scripts/fa_kernel_time.py $SHAPE --iters 1000 --last-n 100 --scale-on-q 1

# stock LLVM: the same command with every variable above removed
HIP_VISIBLE_DEVICES=<gpu> FA_MODULE=fav4 \
python scripts/fa_kernel_time.py $SHAPE --iters 1000 --last-n 100 --scale-on-q 1
```

**Reproducing the FlyDSL row** needs a checkout of [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL);
these numbers were taken at commit **`63eb891`** (`v0.2.4-26-g63eb891`):

```bash
git clone https://github.com/ROCm/FlyDSL && cd FlyDSL && git checkout 63eb891
pip install -e .            # per FlyDSL's own README

cd <this repo>
HIP_VISIBLE_DEVICES=<gpu> FLYDSL_ROOT=/path/to/FlyDSL \
python scripts/fly_kernel_time.py --batch 32 --seqlen 8192 --hq 8 --d 128 \
  --iters 1000 --last-n 100
```

`scripts/fly_kernel_time.py` builds `flash_attn_dualwave_swp` through FlyDSL's own
`build_flash_attn_dualwave_swp_module`, then times it exactly as `fa_kernel_time.py` times ours —
same rocprofv3 invocation, same serialization, same rotating-buffer rule, same averaging window —
which is the only reason the two rows can be put in one table. It checks the output against
`scaled_dot_product_attention` before timing. The window depth matters: at `B=1` FlyDSL's own
shallower window leaves the kernel in its thermal transient and six consecutive runs of one config
read 1236.9, 1242.8, 1165.9, 1167.7, 1159.3, 1157.5. At the shape above both harnesses are steady.

Its defaults are FlyDSL's own tuned configuration (`FLASH_ATTN_FUNC_KERNEL_CONFIG` in that
project's `tests/kernels/test_flash_attn_fwd.py`): lazy rescale, `setprio`, stagger, and
`waves_per_eu=2`. Sweeping those confirms the defaults are its optimum — disabling stagger costs
15%, disabling `setprio` 3%, `waves_per_eu=1` is indistinguishable, and `--eager-rescale` (the
`fav3`-equivalent path) costs far more. Note the builder defaults to `causal=True` while this
script passes `causal=False` to match the Gluon kernels; `--causal 1` measures the causal path
against half the FLOPs, which comes out lower still.

## 9. Where to go deeper

- [`../gemm/README.md`](../gemm/README.md) — read this **first** if you have not. §3 above
  assumes its intra-wave / inter-wave taxonomy, and `gemm/inter_wave/` is the two-wave
  ping-pong that attention builds on.
- [`note.md`](note.md) — the optimization notebook: every step with its measurement, the
  per-stage instruction inventories, the environment-variable reference, the measurement
  protocol, and the comparison methodology. Its last section is the file map, what the two
  kernels do and do not support, and the odd-tail tile.
- [`mfma_coissue_scheduling.md`](mfma_coissue_scheduling.md) — §2's budget as a formal
  scheduling problem: the machine model, the optimal schedule for `M` MFMAs against `N` units
  of math, and a matching-lower-bound proof that it is optimal.
- [`../../plugins/llir_scheduler/llir_scheduler.html`](../../plugins/llir_scheduler/llir_scheduler.html)
  — how the scheduler classifies a region, and how it packs or triages the windows.
- [`../../docs/warp_pipelining.md`](../../docs/warp_pipelining.md) and
  [`../../docs/mfma_efficiency.md`](../../docs/mfma_efficiency.md) — the theory behind
  `warp_pipeline_stage` (§4) and behind the metric §8 reports.
- **Provenance.** Ported from
  [`AMD-Triton/gluon-kernels`](https://github.com/AMD-Triton/gluon-kernels)
  (`kernels/cdna4/fa/`). `f16_fa_gfx950_common.py` is verbatim; `fav3.py` is the upstream
  rotated-4-cluster kernel reduced to the single best config for this shape — the
  per-`(D, BLOCK_N, warps)` layout dispatch, causal/masked-tail scheduling, non-pipelined
  fallbacks, head-dim padding and the multi-config autotune space were removed and the pipelined
  loop inlined into one flat `gluon_attn_fwd`. The full version is upstream and in git history.
  Both vendored files are excluded from this repo's black/ruff (see `pyproject.toml`);
  `bench.py` is tutorial-native and linted.
