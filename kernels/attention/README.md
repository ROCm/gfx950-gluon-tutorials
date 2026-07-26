# Flash Attention on gfx950 (CDNA4) in Gluon — `fav3` and `fav4`

Forward flash-attention kernels for gfx950 (MI350 / MI355X), written in Triton
Experimental **Gluon**. Two kernels share one pipeline architecture:

| kernel | softmax rescaling | TFLOPS | MFMA efficiency / SIMD |
|---|---|---:|---:|
| `fav3` | eager — correct the accumulator every tile | 1167 | 85.3% |
| **`fav4`** | **lazy — correct it only when a row's max actually moves** | **1245** | **95.1%** |

`B=1, HQ=HK=64, D=128, S=16320, fp16`, non-causal, on MI355X. For scale: the same shape
on a build with no scheduling plugin measures ~1127, and ROCm/FlyDSL — a DSL kernel whose
schedule is written out by hand — measures 1242.

This document explains how the kernel works and how it got there. The optimization
notebook — per-step measurements, instruction inventories, compiler findings — lives in
[`note.md`](note.md); the scheduler these kernels co-operate with is documented in
[`../../plugins/llir_scheduler/llir_scheduler.html`](../../plugins/llir_scheduler/llir_scheduler.html).

---

## 1. Files

| file | what it is |
|---|---|
| `fav3.py` | the baseline kernel: Gluon kernel + its single autotune config + host launcher `run_gluon_attention` |
| `fav4.py` | `fav3` plus lazy rescaling and the optimizations in §6 |
| `f16_fa_gfx950_common.py` | shared helpers (`input_helper`, `sdpa_reference`, `compute_flops`, layout/stride plumbing) |
| `bench.py` | correctness against torch SDPA + `do_bench` TFLOPS; `--rocprof` / `--prepared` dispatch loops for external timing |
| `note.md` | the optimization notebook: what was tried, what it measured, and why |
| `att_attn*.json` | `rocprofv3` ATT (instruction-trace) configurations |

Pick the kernel with `FA_MODULE=fav3` (default) or `FA_MODULE=fav4`.

**Scope.** Both are reduced to the single most-performant path: non-causal, head dim 128,
fp16/bf16, `bhsd`/`bshd`, MHA/GQA/MQA, and a K length that is a multiple of `BLOCK_N`=64
with an odd number of tiles (a `static_assert` checks it — 16320 is fine, 16384 is not).
Causal masking, ragged tails, other head dims and the wide autotune space were removed to
keep the code readable; see the provenance note in §9.

## 2. How to run

```bash
# correctness + do_bench TFLOPS
FA_MODULE=fav4 python bench.py --seqlen 16320

# the reported metric: kernel time from rocprofv3, prepared launch
FA_MODULE=fav4 python ../../scripts/fa_kernel_time.py --seqlen 16320
```

Both need the scheduling plugin loaded and two other passes configured. The exact
variables, which component owns each one, and which are cache-invalidating are tabulated
under **Environment variables** in [`note.md`](note.md). One asymmetry is worth repeating
here because it is easy to get backwards: **`AMDGCN_SCALARIZE_PACKED_FOPS=1` is for
`fav4` only — `fav3` must not set it** (§7 explains why).

Defaults: `B=1, HQ=HK=64 (MHA), D=128, fp16, bhsd`, non-causal.

## 3. The algorithm: attention without an S×S matrix

Attention is `O = softmax(Q·Kᵀ · scale) · V`. Done literally, the `S×S` score matrix never
fits in registers or LDS, so flash attention streams it: walk the K/V sequence in tiles
and keep the softmax *running* rather than finishing it.

Softmax needs a max (to keep the exponent in range) and a sum (to normalize), both over a
whole row — neither is known until the last tile. The insight is that both can be carried
and later corrected. Per query row the kernel holds

```
m    running max of the scores seen so far
l    running sum of exp2(score − m)
acc  running Σ exp2(score − m)·V, the unnormalized output
```

and for tile `j`:

```
s      = Q·Kⱼᵀ · scale                    one MFMA chain          ("QK")
m_new  = max(m, rowmax(s))
p      = exp2(s − m_new)                  the transcendental burst
alpha  = exp2(m − m_new)                  how far the frame moved
l      = l·alpha + rowsum(p)
acc    = acc·alpha + p·Vⱼ                 the second MFMA chain   ("PV")
```

The one division, `acc / l`, happens after the loop. Everything inside it is exact:
`alpha` retroactively rebases the earlier partial sums into the new frame, which is what
makes the streaming form equal the literal one.

**`acc·alpha` is the term this document is mostly about.** `acc` is the largest live value
in the kernel — 64 VGPRs per lane at this tile size — so rescaling it every iteration
costs 64 vector instructions, and they are pure overhead whenever the row max did not
actually move. §6 is largely the story of removing them.

## 4. The kernel: one output tile per workgroup

The grid is `(HQ, ceil(S/BLOCK_M), B)`. Each workgroup owns a `BLOCK_M`=256-row slab of
the output for one head and walks all of K/V.

**Data placement.** `Q`'s slab is loaded once, staged through LDS to put it in MFMA
operand layout, and then *stays in registers* for the whole loop — it is read every
iteration and never changes. `K` and `V` tiles stream through a double-buffered pair of
LDS slots (`BUF_DEPTH=2`), filled by `buffer_load_to_shared` async copies straight from
global memory and read back with `load_shared_relaxed`. `acc`, `m` and `l` live in
registers across iterations.

**Tile shape.** `BLOCK_M=256`, `BLOCK_N=64`, `D=128`, 8 warps, `mfma_f32_32x32x16`. With
`warps_per_cta=[8,1]` each wave owns 32 of the 256 rows, so the row-wise softmax
reductions are per-wave — one cross-lane shuffle each, no cross-workgroup reduction. Two
MFMA chains per tile, 16 MFMAs each: QK (`256×64×128`) and PV (`256×128×64`).

**Two hardware details that shape the code.** Accumulators are pinned to VGPRs rather
than AGPRs (`llvm_fn_attrs amdgpu-agpr-alloc="0,0"`): with the 2×-unrolled loop the
default AGPR placement makes the backend shuffle accumulators through `v_accvgpr` moves
on the critical path, worth ~50 TFLOPS here. And `waves_per_eu=2` places two waves on each
SIMD, so one wave's stalls are covered by the other's compute — which is what the pipeline
below is built around.

## 5. The software pipeline: four clusters, rotated

A naive body would run QK → softmax → PV → load-next-tile in sequence, leaving the matrix
core idle during the loads and the loads idle during the math. Instead the body is split
into **four clusters**, rotated so each MFMA cluster is paired with memory traffic for a
*future* tile:

```
dot1 :  QK MFMA (tile j)      +  VEC2   softmax denominator work
mem1 :  LDS read of V(j)      +  async global→LDS copy of K(j+3)
dot2 :  PV MFMA (tile j)      +  VEC1   softmax numerator work
mem2 :  LDS read of K(j+1)    +  async global→LDS copy of V(j+2)
```

`warp_pipeline_stage("…")` marks each cluster and the compiler lowers each boundary to a
barrier. With two waves per SIMD the waves ping-pong — one is in a `dot` cluster while the
other is in a `mem` cluster — so LDS and global latency hide under the other wave's matrix
work. The loop is **2×-unrolled** (even tile, then odd tile) so the double-buffer slot
indices are compile-time constants instead of runtime modulo arithmetic, and the async
copies stay two commit-groups deep.

**Why the softmax is split in two.** The per-tile softmax work is deliberately cut into
two groups that land in *different* MFMA clusters:

- **VEC1** (in `dot2`, beside the PV MFMA): the new row max and the `exp2` burst that
  produces `p`. These are transcendentals — 8 cycles each, and they cannot dual-issue with
  an MFMA — so they get a cluster's worth of MFMA shadow to themselves.
- **VEC2** (in `dot1`, beside the QK MFMA): the row sum, the running-denominator update,
  and the cast of `p` to fp16 for the next PV MFMA.

Neither cluster then has to hide the whole softmax, and both MFMA chains have independent
vector work available to co-execute with. One consequence to keep in mind while reading
the code: `VEC1` of tile `j` produces the `p` that `VEC2` and the PV MFMA consume in tile
`j+1`, so the software pipeline is skewed by one iteration.

## 6. From `fav3` to `fav4`: the optimization journey

`fav3` is the honest implementation of §3 — every iteration multiplies `acc` by `alpha`.
`fav4` keeps the same pipeline and attacks the softmax overhead in four movements.

### Act I — stop rescaling when nothing moved

Softmax is shift-invariant, so the running max does not have to be *tight*; it only has to
keep `exp2`'s argument in range. `fav4` lets it **lag**: the max is bumped, and `acc`
rescaled, only when a tile's max exceeds the running max by more than a log2 threshold
(8 — a 256× safety margin, trivially inside fp32's range). When the max is stable, which
is the common case after the first few tiles, `p` is allowed to rise as high as 256 and
the correction is skipped entirely. `acc` and `l` are carried in the same lagging frame,
so the final result is unchanged.

Skipping needs a branch, and the useful granularity is the wave: each wave owns 32 rows
and can decide independently. `gl.warp_predicate` expresses exactly that — it lowers to
`s_and_saveexec` + `s_cbranch_execz`, with no cross-wave reduction and no barrier. Rows
that did not advance carry `alpha == 1`, so even an unskipped multiply is a no-op for them.

### Act II — keep the branch off the MFMA's critical path

A branch immediately ahead of a cluster's first MFMA is expensive: the matrix core waits
on control flow. Two moves fix that.

- **`opt1`** hoists the `p`→fp16 cast to the top of `VEC2` so it starts earlier, while the
  QK MFMA still leads the cluster. *+1.4%*
- **`opt3`** pulls the rescale out of `VEC2` into its own step in the **`mem2`** cluster.
  Its branch latency now overlaps LDS and global latency instead of standing in front of a
  matrix chain, and `dot1` becomes `[sum + cast] + QK MFMA` with nothing branching ahead of
  it. *+2%*

### Act III — balance the two dot clusters

After Act II the clusters are lopsided: `dot2` carries the entire `exp2` burst (33
transcendentals at 8 cycles each) while `dot1` has room to spare. The fix is to move part
of the burst across — but a score tile is a distributed tensor, and slicing one usually
costs data movement.

It does not have to. `reshape` → `permute` → `split` on an MFMA-layout tensor is a pure
*re-interpretation*: every slice keeps the same layout, so the split is a partition of each
lane's own registers and emits **no instructions at all**. `assert_trivial=True` makes the
compiler prove that at build time and fail the compile otherwise.

So the score tile is sliced along N and the `sub`+`exp2` work for some slices is computed
in the *other* cluster: `opt4` splits 1:1, `opt7` four ways, and **`opt8`** eight ways,
moving 3/8 of the work into `dot1`. `VEC1` carries the raw slices forward and `VEC2`
computes their subtract where it consumes them; concatenating back into a full `p` is
equally free. Register-neutral, exact (`exp2` is elementwise), and it flattens the two
clusters to within a few cycles of each other.

### Act IV — cheaper per-element math, and pacing

- **`opt5` — scale `Q`, not the scores.** `qk_scale` used to be applied per score element
  as `fma(qk, scale, −m)`. Folding it into `Q` once before the loop turns that into a plain
  `sub`. The choice survives as the `SCALE_ON_Q` constexpr (default true): `--scale-on-q 0`
  restores the fma, which costs ~0.9% but is slightly *more* accurate, because pre-scaling
  rounds `q·scale` back to fp16 before the loop.
- **`opt6` — pace the mem clusters.** A couple of `s_nop`s at the head of each mem cluster
  shift that wave's LDS burst slightly later so it does not collide with the other wave's.
  This one lives in the scheduler rather than the kernel (§7) and is on by default at the
  measured optimum.

## 7. Making the compiler co-operate: MFMA/VALU co-execution

Everything above is about *having* independent vector work beside each MFMA. Actually
issuing it there is the compiler's job, and it needs help.

An MFMA occupies the matrix core for many cycles while the vector ALU sits idle next to
it. The two issue independently, so VALU placed behind an MFMA **co-executes** with it and
is effectively free — a 32×32×16 MFMA exposes a **24-cycle window**. Exceed the window and
the surplus is exposed; leave it empty and the matrix core runs alone while that vector
work waits its turn later.

The out-of-tree **llirSched** plugin fills those windows. For a region containing MFMA and
VALU but no memory traffic — precisely the `dot` clusters of §5 — it computes which vector
ops belong behind which MFMA and *declares* that schedule with `sched_group_barrier`, which
AMDGPU's IGroupLP then builds in the machine scheduler. Two properties matter to a reader
of these kernels:

- **`fav4`'s dot clusters fit their windows** (360 and 348 cycles of an available 384), so
  the scheduler simply spreads the work evenly. This is why `fav4` wants
  `AMDGCN_SCALARIZE_PACKED_FOPS=1`: packed `v_pk_*` math cannot co-issue with an MFMA at
  all, so splitting it into per-element scalars is what makes it schedulable.
- **`fav3`'s do not** (476 and 496 cycles of 384) — its eager rescale puts 64 extra
  `v_mul` into `dot1`, more than the windows can hide. There the scheduler answers a
  different question: which ops get a window (the ones that cannot be packed go first),
  and what shape the rest take (uncovered work stays *packed*, since one `v_pk_mul`
  retires two elements in a single issue). That is why **`fav3` must not** set
  `AMDGCN_SCALARIZE_PACKED_FOPS` — a blanket split would defeat it.

That asymmetry is the sharpest statement of what Act I bought. `fav4` removed enough vector
work from the matrix core's shadow that scheduling became a packing problem instead of a
triage problem.

## 8. Results

`B=1, HQ=HK=64, D=128, S=16320, fp16`, non-causal, MI355X. TFLOPS from `rocprofv3` kernel
time with a prepared launch; MFMA efficiency from an ATT instruction trace, per SIMD.
`cyc/iter` is one 2×-unrolled loop body, and every row does the same 2048 MFMA cycles of
work per body, so the columns are comparable across implementations.

| kernel | TFLOPS | cyc/iter | MFMA eff / SIMD |
|---|---:|---:|---:|
| **`fav4`** (`SCALE_ON_Q=1`) | **1245** | **4306** | **95.1%** |
| `fav4`, `--scale-on-q 0` | 1231 | 4419 | 92.7% |
| `fav3` | 1167 | 4802 | 85.3% |
| `fav3`, no scheduling plugin | 1127 | — | — |
| *ROCm/FlyDSL, lazy rescale* | *1242* | *4747* | *86.3%* |
| *ROCm/FlyDSL, eager rescale* | *1044* | *6695* | *61.2%* |

Two things worth reading off this table.

**Cycles and TFLOPS disagree, and cycles are the honest metric for scheduling work.**
`fav4` needs 9% fewer cycles per iteration than the FlyDSL kernel yet only ties it on wall
time: a denser MFMA stream draws more power, and on a board already at 96% of its power cap
that buys a lower clock. Judge a scheduling change by MFMA efficiency; judge a kernel by
both.

**The eager/lazy gap is far larger in the hand-scheduled kernel** (−16% for FlyDSL against
−6% here), because `fav3`'s over-capacity handling in §7 recovers most of it. That is the
same phenomenon seen from the other side: a scheduler can compensate for vector work it
cannot hide, but not completely.

## 9. Where to go deeper

- [`note.md`](note.md) — the optimization notebook: every step with its measurement, the
  per-stage instruction inventories, the environment-variable reference, the measurement
  protocol and the comparison methodology.
- [`../../plugins/llir_scheduler/llir_scheduler.html`](../../plugins/llir_scheduler/llir_scheduler.html)
  — how the scheduler decides which model a region gets, and how it packs the windows.
- **Provenance.** Ported from
  [`AMD-Triton/gluon-kernels`](https://github.com/AMD-Triton/gluon-kernels)
  (`kernels/cdna4/fa/`). `f16_fa_gfx950_common.py` is verbatim; `fav3.py` is the upstream
  rotated-4-cluster kernel reduced to the single best config for this shape — the
  per-`(D, BLOCK_N, warps)` layout dispatch, causal/masked-tail scheduling, non-pipelined
  fallbacks, head-dim padding and the multi-config autotune space were removed and the
  pipelined loop inlined into one flat `gluon_attn_fwd`. The full version is upstream and in
  git history. Both vendored files are excluded from this repo's black/ruff (see
  `pyproject.toml`); `bench.py` is tutorial-native and linted.
