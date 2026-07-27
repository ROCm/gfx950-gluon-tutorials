# Flash Attention on gfx950 (CDNA4) in Gluon

Forward flash-attention kernels for gfx950 (MI350 / MI355X), written in Triton Experimental
**Gluon**. Two kernels, `fav3` and `fav4`, share one pipeline architecture and differ in how
they handle the softmax rescale.

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
the compiler has to do for them (§6).

This assumes you know the flash-attention algorithm — the streaming softmax that carries a
running max `m`, a running sum `l` and an unnormalized accumulator `acc`, and rebases them with
`alpha = exp2(m − m_new)` as the max moves. The term to keep in mind is **`acc·alpha`**: `acc`
is the largest live value in the kernel, so rescaling it every tile is 64 vector instructions
that are pure overhead whenever the row max did not actually move. §5 is the story of removing
them.

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

That is reachable — `num_warps=4` puts one wave of a workgroup on each SIMD, so three resident
workgroups per CU give three waves per SIMD. But each workgroup then carries its own `Q` and
its own accumulator (§4.3: 96 VGPRs of live-in before any working set), which triples the part
of the register budget that cannot be shared. And `s_barrier` synchronizes *within* a
workgroup, so there is no primitive to keep waves from three different workgroups in the phase
relationship the pipeline depends on.

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
> here as behaviour rather than mechanism.

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
of §6. Regions mixing all three are future work — an intra-wave FA kernel would need them, and
the two models would have to be reconciled rather than selected between.

## 4. Designing the loop

One workgroup owns a `BLOCK_M`=256-row slab of the output for one head and walks all of K/V;
the grid is `(HQ, ceil(S/BLOCK_M), B)`. Per tile there are eight things to do.

![the eight per-tile operations and their dependencies](images/tile_deps.svg)

Run them in dependency order and the matrix core idles through every copy and every LDS read.
The fix is the standard one — software-pipeline the loop so that the memory for a *future* tile
overlaps the matrix work of the current one.

![the four stages of each iteration, marching diagonally over time](images/pipeline.svg)

Read the diagonal: at any moment the wave is working on four different tiles at once. The
copy of tile *j+3*, the LDS reads of *j+2*, the first MFMA chain of *j+1*, and the second MFMA
chain of *j* all happen in the same time step. That vertical slice **is** the loop body, and
the tile index attached to each operation says how far ahead of the current tile it runs.

§1 then fixes how the slice is subdivided: every group must pair matrix work with memory so
that a wave in one kind of group always faces a wave in the other. Four clusters do it.

![the loop body split into four alternating clusters](images/clusters.svg)

`warp_pipeline_stage("…")` marks each cluster and the compiler lowers each boundary to a
barrier. With `waves_per_eu=2` the two waves on a SIMD run **one cluster apart**:

![two waves running the same four clusters, offset by one](images/pingpong.svg)

### 4.2 How the softmax is split

The softmax is cut into two groups that land in *different* dot clusters, so that neither
cluster has to hide all of it and both MFMA chains have independent vector work beside them:

| | work | lands in |
|---|---|---|
| **VEC1** | the new row max, then `fma` (subtract the max) and `exp2` on the score tile | `dot2`, beside the PV MFMA |
| **VEC2** | the row sum, the downcast of `p` to fp16 for the next PV MFMA, and the accumulator rescale | `dot1`, beside the QK MFMA |

The consequence to hold on to while reading the code: `VEC1` runs in `dot2` on tile *j+1* while
the PV MFMA there is still working on tile *j*, so the `p` it produces is consumed a full turn
of the pipeline later. Every buffer in the kernel is sized around that skew.

### 4.3 Register budget, and why the loop is unrolled 2×

With `warps_per_cta=[8,1]` each of the 8 waves owns 32 of the 256 rows, so the row-wise softmax
reductions are per-wave — one cross-lane shuffle each, no cross-workgroup reduction. That also
sets what each wave has to hold.

![the two MFMA chains and the tile shapes one wave works on](images/tile_shapes.svg)

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

`fav3` applies the rescale unconditionally — every iteration multiplies `acc` by `alpha`, 64
`v_mul` per tile straight into `dot1`'s budget. `fav4` keeps the identical pipeline and removes
them. That is the whole difference between the two kernels, and it is worth **+6.7%**
(1167 → 1245 TFLOPS) plus a qualitative change in what the compiler has to do (§6).

**Lazy rescaling.** Softmax is shift-invariant, so the running max does not have to be *tight*;
it only has to keep `exp2`'s argument in range. `fav4` lets it **lag**: the max is bumped, and
`acc` rescaled, only when a tile's max exceeds the running max by more than a log2 threshold of
8 — a 256× safety margin, trivially inside fp32's range. When the max is stable, which is the
common case after the first few tiles, `p` is allowed to rise as high as 256 and the correction
is skipped entirely. `acc` and `l` are carried in the same lagging frame, so the result is
unchanged.

Skipping needs a branch, and the useful granularity is the wave: each wave owns 32 rows and can
decide independently. `gl.warp_predicate` expresses exactly that — it lowers to
`s_and_saveexec` + `s_cbranch_execz`, with no cross-wave reduction and no barrier. Rows that did
not advance carry `alpha == 1`, so even an unskipped multiply is a no-op for them.

### Four design rules the final kernel embodies

Getting from a working lazy-rescale kernel to the committed one was worth a further **+7.5%**
(1158 → 1245), spread over half a dozen steps that are all the same idea seen from different
angles. Rather than a version-by-version journey, they are more useful as rules — each one a
consequence of §2:

| rule | why the hardware demands it |
|---|---|
| **Nothing that branches may lead a dot cluster.** The rescale's `warp_predicate` was moved out of `VEC2` into the `mem2` cluster, and the `p`→fp16 cast hoisted to the top of `VEC2`. | The matrix core stalls on control flow. A branch's latency should overlap memory, not stand in front of an MFMA chain. `dot1` now begins `[sum + cast] + QK MFMA` with nothing branching ahead of it. |
| **Balance the two dot clusters' budgets.** The score tile is sliced along N and the `sub`+`exp2` for 3/8 of the slices is computed in the *other* cluster. | One cluster overflowing while the other has slack wastes shadow. After the rescale moved out, `VEC1` had roughly twice `VEC2`'s work. |
| **Rebalancing must not cost data movement.** `reshape` → `permute` → `split` on an MFMA-layout tensor is a pure re-interpretation: every slice keeps the same layout, so the split partitions each lane's own registers and emits **no instructions at all**. `assert_trivial=True` makes the compiler prove that at build time and fail the compile otherwise. | A distributed-tensor slice normally costs a shuffle. Free slicing is what makes the previous rule affordable — and it is a Gluon-level technique worth knowing well beyond attention. |
| **Move per-element work out of the loop when the math allows.** `qk_scale` is folded into `Q` once before the loop, turning a per-score `fma(qk, scale, −m)` into a plain `sub`. | 64 elements per tile per wave, every tile. Kept as the `SCALE_ON_Q` constexpr: `--scale-on-q 0` restores the fma, costing ~0.9% but slightly *more* accurate, since pre-scaling rounds `q·scale` back to fp16 before the loop. |

A fifth rule belongs to the scheduler rather than the kernel: **pace the mem clusters.** A
couple of `s_nop`s at the head of each mem cluster shift that wave's LDS burst slightly later so
it does not collide with the other wave's. It is on by default at the measured optimum.

## 6. Making the compiler co-operate

Everything above is about *having* independent vector work beside each MFMA. Actually issuing it
there is the compiler's job, and by §3 the dot clusters are intra-wave regions, so it needs
help. The out-of-tree **llirSched** plugin computes which vector ops belong behind which MFMA
and *declares* that schedule with `sched_group_barrier`, which AMDGPU's IGroupLP then builds in
the machine scheduler.

Measured from the compiled kernels, here is what the two are asking of the 384-cycle budget:

![co-execution budget of each dot cluster in fav3 and fav4](images/coexec_budget.svg)

**`fav4` fits.** 348 and 340 cycles of 384. Every op can be given a window slot, so the
scheduler's job is simply to spread the work evenly — and because packed math cannot co-execute
at all (§2), every packed op must first be split into per-element scalars. That is what
`AMDGCN_SCALARIZE_PACKED_FOPS=1` does, and why `fav4` requires it.

**`fav3` does not.** Its eager rescale adds 64 `v_mul` to `dot1`, and its PV cluster fills the
window exactly with `exp2` alone. There the scheduler answers a different question — not *how do
I spread this*, but *what gets a slot and what shape should the rest take*:

- ops that **cannot** be packed (`v_max3`, `v_exp`) are covered first, since packing is not an
  option for them anyway;
- whatever window is left goes to packable ops, **split into scalars** so they can use it;
- everything still uncovered stays **packed**, because one `v_pk_mul` retires two elements in a
  single issue slot and it is going to be exposed either way.

Which is why **`fav3` must not set `AMDGCN_SCALARIZE_PACKED_FOPS`** — a blanket split would
double the issue cost of exactly the work that failed to get a slot. The asymmetry is the
sharpest statement of what lazy rescaling bought: `fav4` removed enough vector work from the
matrix core's shadow that scheduling became a packing problem instead of a triage problem.

## 7. Results

`B=1, HQ=HK=64, D=128, S=16320, fp16`, non-causal, MI355X, `rocm-smi` GPU[0]. TFLOPS from
`rocprofv3` kernel time with a prepared launch; MFMA efficiency from an ATT instruction trace,
per SIMD. Every row does the same 2048 MFMA cycles of work per loop body, so the columns are
comparable across implementations.

| kernel | TFLOPS | cyc/iter | MFMA eff / SIMD | ceiling from §6 |
|---|---:|---:|---:|---:|
| **`fav4`** (`SCALE_ON_Q=1`) | **1245** | **4306** | **95.1%** | **100%** |
| `fav4`, `--scale-on-q 0` | 1231 | 4419 | 92.7% | 100% |
| `fav3` | 1167 | 4802 | 85.3% | **91.4%** |
| `fav3`, no scheduling plugin | 1127 | — | — | 91.4% |
| *ROCm/FlyDSL, lazy rescale* | *1242* | *4747* | *86.3%* | |
| *ROCm/FlyDSL, eager rescale* | *1044* | *6695* | *61.2%* | |

**The ceilings come straight from §6's budget**, and they are what make the efficiency column
readable. `fav4`'s demand fits inside the window, so nothing is exposed and its ceiling is
100%; it reaches 95.1%. `fav3` has 4 × 48 = 192 exposed cycles per loop body against 2048 of
MFMA, so its ceiling is 2048/2240 = **91.4%**; it reaches 85.3%.

That reframes the comparison. The two kernels are 9.8 points apart in efficiency, but their
*ceilings* are 8.6 points apart — so almost the whole gap is the exposed packed work that
`fav3`'s budget cannot absorb, not the scheduler doing a worse job on it. Each kernel is within
about 5–6 points of its own ceiling, and that remainder is barrier and pacing overhead.

Three things worth reading off the table.

**MFMA efficiency is derived from the cycle count**, not measured independently: it is
`mfma_count × 32 / loop_cycles` per wave, doubled for the per-SIMD figure because two waves
share the matrix unit. It is the right metric for judging a *scheduling* change,
because that is exactly what a scheduling change moves — but it is the reciprocal of `cyc/iter`
and never adds information beyond it.

**Cycles and TFLOPS disagree, and both are honest about different things.** `fav4` needs 9%
fewer cycles per iteration than the FlyDSL kernel yet only ties it on wall time: a denser MFMA
stream draws more power, and on a board already at 96% of its power cap that buys a lower
clock. Judge a scheduling change by cycles; judge a kernel by both.

**The eager/lazy gap is far larger in the hand-scheduled kernel** — −16% for FlyDSL against
−6.7% here — because `fav3`'s over-capacity handling in §6 recovers most of it. A scheduler can
compensate for vector work it cannot hide, but not completely.

## 8. Files, and how to run

| file | what it is |
|---|---|
| `fav3.py` | eager rescale: Gluon kernel + its autotune config + host launcher `run_gluon_attention` |
| `fav4.py` | `fav3` plus lazy rescaling and the rules of §5 |
| `f16_fa_gfx950_common.py` | shared helpers (`input_helper`, `sdpa_reference`, `compute_flops`, layout/stride plumbing) |
| `bench.py` | correctness against torch SDPA + `do_bench` TFLOPS; `--rocprof` / `--prepared` dispatch loops for external timing |
| `note.md` | the optimization notebook: what was tried, what it measured, and why |
| `att_attn*.json` | `rocprofv3` ATT (instruction-trace) configurations |

```bash
# correctness + do_bench TFLOPS
FA_MODULE=fav4 python bench.py --seqlen 16320

# the reported metric: kernel time from rocprofv3, prepared launch
FA_MODULE=fav4 python ../../scripts/fa_kernel_time.py --seqlen 16320
```

Pick the kernel with `FA_MODULE=fav3` (default) or `FA_MODULE=fav4`. Defaults are
`B=1, HQ=HK=64 (MHA), D=128, fp16, bhsd`, non-causal.

Both need the scheduling plugin loaded and two other passes configured. The exact variables,
which component owns each, and which are cache-invalidating are tabulated under **Environment
variables** in [`note.md`](note.md). One asymmetry bears repeating because it is easy to get
backwards: **`AMDGCN_SCALARIZE_PACKED_FOPS=1` is for `fav4` only — `fav3` must not set it**
(§6 explains why).

**Scope.** Both kernels are reduced to the single most-performant path: non-causal, head dim
128, fp16/bf16, `bhsd`/`bshd`, MHA/GQA/MQA, and a K length that is a multiple of `BLOCK_N`=64
with an odd number of tiles (a `static_assert` checks it — 16320 is fine, 16384 is not). Causal
masking, ragged tails, other head dims and the wide autotune space were removed to keep the
code readable; see the provenance note below.

## 9. Where to go deeper

- [`../gemm/README.md`](../gemm/README.md) — read this **first** if you have not. §3 above
  assumes its intra-wave / inter-wave taxonomy, and `gemm/inter_wave/` is the two-wave
  ping-pong that attention builds on.
- [`note.md`](note.md) — the optimization notebook: every step with its measurement, the
  per-stage instruction inventories, the environment-variable reference, the measurement
  protocol, and the comparison methodology.
- [`../../plugins/llir_scheduler/llir_scheduler.html`](../../plugins/llir_scheduler/llir_scheduler.html)
  — how the scheduler classifies a region, and how it packs or triages the windows.
- **Provenance.** Ported from
  [`AMD-Triton/gluon-kernels`](https://github.com/AMD-Triton/gluon-kernels)
  (`kernels/cdna4/fa/`). `f16_fa_gfx950_common.py` is verbatim; `fav3.py` is the upstream
  rotated-4-cluster kernel reduced to the single best config for this shape — the
  per-`(D, BLOCK_N, warps)` layout dispatch, causal/masked-tail scheduling, non-pipelined
  fallbacks, head-dim padding and the multi-config autotune space were removed and the pipelined
  loop inlined into one flat `gluon_attn_fwd`. The full version is upstream and in git history.
  Both vendored files are excluded from this repo's black/ruff (see `pyproject.toml`);
  `bench.py` is tutorial-native and linted.
