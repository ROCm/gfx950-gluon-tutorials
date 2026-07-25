# FAv3 gfx950 — MFMA/VALU co-issue scheduling exploration

Companion to `mfma_coissue_scheduling.md` (the formal cycle-cost proof). What we
tried to better overlap VALU with MFMA in the rotated-4-cluster gluon kernel.

**Setup.** `gluon_attn_fwd`, 1×16320 bshd fp16 non-causal, 32×32×16 MFMA,
BLOCK_M=256 BLOCK_N=64 nw=8. Mechanism: emit `rocdl.sched.group.barrier` groups
at the END of each DOT cluster in Triton's `ConvertWarpPipeline.cpp`, built on the
pinned upstream LLVM (no LLVM rebuild); env-gated `TRITON_HIP_DOT_COISSUE=1`.
Measured via do_bench TFLOPS + ATT `process_json.py` (MFMA eff = mfma_cyc/iter;
2 waves/SIMD, so ~100%/SIMD ≈ 50%/wave).

## Results (1×16320)

| variant | iter cyc | MFMA eff/wave | TFLOPS |
|---|---:|---:|---:|
| baseline | 6282 | 32.60% | 1018 |
| K=3 co-issue (packed math present) | 5768 | 35.51% | 1046 |
| scalarize + uniform K=8 | 5451 | 37.57% | 1066 |
| **scalarize + VALU/TRANS split (best)** | **5390** | **38.00%** | **1071** |
| mfma-only floor (ablation, numerically wrong) | 4105 | 49.89% | 1322 |

Net baseline → best: **+5.2% TFLOPS**, MFMA eff 32.6 → 38.0% (~76%/SIMD).

## SOL sweep — 5-way build comparison (per-SIMD MFMA eff)

End-to-end sweep across compiler branch + LLVM variant, one config each. MFMA
efficiency is **per-SIMD** (= 2× the per-wave number `process_json.py` prints;
`waves_per_eu=2`). Configs 1–4 are the gluon kernel with the `gl.fma` source
fusion at `b1 h64 d128 sq16320 bshd fp16`; config 5 is the reference kernel at
`sq16384`. All correctness `match`. ATT traces in `/data/sol_sweep_*`.

| # | implementation | build | TFLOPS | eff/SIMD | eff/wave | iter cyc |
|---|---|---|---:|---:|---:|---:|
| 1 | baseline | gfx950-tutorial + pinned LLVM, no llirSched | 1002 | 62.4% | 31.2% | 6561 |
| 2 | fix_boundary | llir_FAv3 + pinned LLVM, no llirSched | 1008 | 65.1% | 32.6% | 6288 |
| 4 | sched.group.barrier | schedGroupBarrier + peephole LLVM (AUTO split) | 1038 | 70.1% | 35.1% | 5839 |
| 3 | llirSched_FAv3 | llir_FAv3 + peephole LLVM + llirSched ON | 1046 | 71.9% | 35.9% | 5701 |
| 5 | triton_reference | fav3_padded + Austin's LLVM | 1077 | 76.4% | 38.2% | 2681 |

**Caveat on #4:** this run used the *auto-derived* co-issue split on *packed* math
(no scalarize) — the weaker Path A variant. The hand-tuned **scalarize + VALU/TRANS
split** (Results table above: 1071 / 38.0%/wave ≈ 76%/SIMD) is Path A's real
ceiling and matches the reference; reproduce with `AMDGCN_SCALARIZE_PACKED_FOPS=1`
+ the reference split. Iter-cycles are not comparable across kernels (reference
2681 vs gluon 5701 is loop structure — 2×-unroll — not speed); **MFMA eff/SIMD is
the comparable microarch metric**.

## Findings

1. **Placement.** `SCHED_GROUP_BARRIER` groups are formed scanning *upward*
   (`AMDGPUIGroupLP::initSchedGroupBarrierPipelineStage`), so the whole sequence
   must sit *after* every real op in the region. Top-of-region = empty groups = no-op.
2. **Masks.** VALU=0x2, MFMA=0x8, TRANS(transcendental)=0x400. `canAddMI`'s VALU
   branch *excludes* TRANS — `v_exp` matches only the TRANS mask. This mirrors
   `isNeverCoissue` (gfx950): TRANS, packed (`v_pk_*`), and DOT/MFMA cannot
   co-issue with an MFMA.
3. **Packed ambiguity.** With packed math present, `[VALU K]` conflates packed
   (2 units) and unpacked (1). K=3 "worked" only because `SIPreEmitPeephole`
   unpacks packed ops that land inside an MFMA window (3 packed → 6 unpacked).
   Fragile; some windows stayed under-filled (dependency-limited, not schedulable).
4. **Start unpacked.** `AMDGCN_SCALARIZE_PACKED_FOPS=1` makes all math unpacked,
   so `[VALU K]` is unambiguous (K ops = K units). Cleaner windows; beats K=3.
5. **Even-spread, not window-fill.** The best K is the *smallest* that drives a
   stage's end-of-stage VALU leftover ("tail") to zero — i.e. *over-demand*
   (K > available). Leftover VALU runs with the MFMA unit idle → cool-down →
   the next stage's first MFMA pays a jump-start stall (barrier-gated in the warp
   pipeline). K=8 is the smallest tail-zero for DOT1; K=10/12 stall the cadence and
   regress. Ablation confirms: deleting *only* DOT2's 19-op softmax tail = +3.3%
   (≈73% of K=3's entire gain).
6. **VALU/TRANS split (best).** Don't load every MFMA with both types. Dedicate
   some MFMAs to VALU and others to exp — DOT2 = `[MFMA][VALU 8]×7` then
   `[MFMA][TRANS 4]×9`. `K2 = K1/2` keeps each MFMA's co-execute load equal *in
   cycles* (8·4cyc = 4·8cyc). Homogeneous per-MFMA load + exp block placed *last*
   respects the softmax dependency chain. Interleaving valu+exp on every MFMA
   *loses* (over-constrains). Order matters: valu-first 1071, exp-first 1047.
   - Split math: pretend 1 exp = 2 valu; `K1=⌈(V+2E)/M⌉`, `K2=⌈K1/2⌉`,
     `g0=round(V/K1)`, `g1=M−g0`. (M16 V54 E33 → 8/4/7/9.)

**Per-stage optimal ceiling** (proof, 24-cyc window): DOT1 86.5%, DOT2 83.7%.
DOT2 is the tighter stage — its 33 `v_exp` (8 cyc, never-co-issue) cap it.

## Dead ends

- **Count-matched K = V/M**: loses to uniform over-demand (matching leaves a tail).
- **Interleaved `[VALU][TRANS]` on every MFMA**: loses (mixing types over-constrains).
- **Deriving the split params at TTGIR**: M is exact from `instr_shape` +
  `warpsPerCTA` + tile (no kWidth needed); E is exact from `getTotalElemsPerThread`;
  g0/g1 reproduce the reference. But V is only *approximate* at TTGIR — element
  counts there ≠ post-codegen scalar-instruction counts (canonicalization/CSE), and
  `tt.reduce` is cross-lane shuffles, not arith. Auto landed at 1054 vs 1071.
  **Conclusion: this instruction-level accounting doesn't belong at TTGIR** — it
  should live in the LLIR / LLVM backend, where real MFMA/VALU/TRANS instructions
  are countable, keeping TTGIR free of instruction-level detail.

## Recommendation

Productionize as an LLVM pass (a new `IGLPStrategy`, or a `SIPreEmitPeephole`
extension) that, per scheduling region, counts MFMA / co-issuable VALU / TRANS,
computes the `K1/K2/g0/g1` split, and emits the valu-first
`[MFMA][VALU]×g0 + [MFMA][TRANS]×g1` sequence — with math scalarized first. The
throwaway Triton prototype (hardcoded reference split, env-gated) is in
`ConvertWarpPipeline.cpp`; ATT traces are archived at `/data/att_gluon_1x16320_bshd_*`.

## Stage-boundary barriers & di/dt — the biggest single win (+21 TFLOPS)

Independent of the interleave: at each `mem → DOT` stage boundary
`ConvertWarpPipeline` emits either a **LOCAL** barrier (`ds_wait + s_barrier` →
`s_waitcnt lgkmcnt(0); s_barrier`) or a **bare** `s_barrier`. In the shipped asm
the QK boundary was LOCAL, the PV boundary bare:

```
QK:  s_waitcnt lgkmcnt(0)      PV:  s_barrier
     s_barrier                      s_waitcnt lgkmcnt(14)   ┐ progressive drain
                                     s_waitcnt lgkmcnt(11)  │ hoisted INTO the
                                     ... lgkmcnt(3,2,1,0)   ┘ MFMA stage
```

A bare barrier lets the backend hoist a **progressive `lgkmcnt(N..0)` into the
following MFMA stage**. Each `s_waitcnt` costs a **4-cyc MFMA co-exec slot and
stalls the matrix unit** — the di/dt trigger we'd been fighting with the
interleave. The LOCAL barrier instead drains once with a single `lgkmcnt(0)` at
the *mem-stage* end, hidden by inter-wave scheduling (`waves_per_eu=2`).

**Why QK≠PV — it's arbitrary, not intrinsic** (`TRITON_WP_DEBUG` dump of the
cluster LDS effects): the **DOT clusters have zero LDS effects** (register-only);
only the mem clusters touch LDS. The one hazard is `mem1↔mem1` across iterations
(K/V buffer reuse). `analyzePipelineDependencies` places the required LOCAL
barrier at slot `dst-1` — a "somewhat arbitrary" (its own comment) choice that
happens to land on the QK (`dot1`) slot. PV just isn't where the heuristic put it.

**Fix:** emit LOCAL unconditionally (LOCAL strictly dominates bare, always
correctness-safe; the extra barriers only order the mem clusters, which the
minimal placement already had to do). Triton `llir_FAv3` commit *"[AMD]
warp-pipeline: always emit LOCAL cluster barriers"*.

**Result:** loop `lgkmcnt` 14→4, bare barriers 2→0, **~1050 → ~1071 TFLOPS**,
MFMA eff **71.4 → 76.2% per-SIMD** — matching the `triton_reference` (1077 /
76.4%). This barrier/lgkmcnt handling was the single largest gap to the
reference, larger than all the interleave tuning combined.

**GEMM regression check.** The unconditional-LOCAL emission fires for *every*
warp-pipeline kernel, so we A/B'd force-local vs the old minimal `bars[i]`
placement on all three inter-wave GEMMs (`a16w16` fp16/bf16, `a8w8` f8, `a4w4`)
across M=N=4096, K=512…65536. Correctness matches everywhere; TFLOPS is within
run-to-run noise (this f8 kernel alone swings ~10% at large K from thermal),
and force-local *slightly helps* small/mid-K a4w4/a8w8 (up to +1.0%). Expected:
DOT clusters have no LDS effects, so the extra LOCAL barriers only re-order the
mem clusters (already ordered) — no added work. **No regression.**

**Isolation** (hand-edited asm via `TRITON_KERNEL_OVERRIDE`, `s_barrier`+lgkmcnt
in the PV boundary): the win needs a single `lgkmcnt(0)` **and** it before the
barrier — neither change alone suffices.

| variant | lgkmcnt inside MFMA stage | drain position | TFLOPS |
|---|---:|---|---:|
| baseline (multiple, after) | 6 | after | 1049 |
| single lgkmcnt(0), after | 1 | after | 1054 |
| reorder only (keep multiple) | 6 | before | 1050 |
| **single lgkmcnt(0), before** | **0** | before | **1070** |

The benefit is monotonic in **#lgkmcnt inside the MFMA stage** (6→1049, 1→1054,
0→1070); the win is getting to **zero** in-stage waits, which needs both the
single-drain *and* the reorder before the barrier.

**Revisit of the di/dt / interleave tradeoff.** The earlier "spread VALU so the
stage tail has few VALU, else di/dt" reasoning was working around the *bare-
barrier* stall. With the LOCAL-barrier fix removing the in-stage `lgkmcnt`
stalls, the interleave is freer — even-spread is no longer forced by di/dt, so a
front-loaded VALU prologue + tightly-packed `[MFMA][6-7 VALU]` groups becomes
worth trying (WIP).

---

# FAv4 — opt1 / opt2 / opt3 (DOT1-stage restructuring)

Follow-on to the co-issue/barrier work above, on `fav4.py` (lazy-softmax-rescale
FA). Same 2×-unrolled rotated 4-stage ping-pong (`dot1`/`mem1`/`dot2`/`mem2`), but
the online-softmax correction (`acc *= alpha`, `l_i *= alpha`) is applied **lazily**
and skipped per-wave via `gl.warp_predicate(alpha < 1.0, …)` (an
`s_and_saveexec`/`s_cbranch_execz` skip, no cross-warp reduction). Numbers are
at `b1 h64 d128 sq16320 fp16` on rocm-smi GPU[4]. **CORRECTION:** the eff column
below is what `process_json.py` prints, which is **per WAVE**
(`mfma_cycles / iteration`), not per SIMD -- with `waves_per_eu=2` the matrix
unit's occupancy is 2x these numbers. See the metric note in the opt4-6 section.

| version | commit | change | eff/wave | TFLOPS |
|---|---|---|:---:|:---:|
| baseline | `8f1410b` | per-wave lazy rescale via `gl.warp_predicate` | 36.6% | — |
| **opt1** | `4f29b69` | hoist p→fp16 cvt to top of `sc_vec2` (mfma still leads dot1) | 36.8% | — |
| **opt2** | `f33b86e` | `sc_vec2` **before** `compute_dot1_qk` (branch leads dot1) | 31.9% → 38.5% | — |
| **opt3** | `26eca45` | rescale hoisted out of dot1 into the prior `mem2` stage | **41.7%** | ~1156 |

opt2's 31.9% is the *naive* build; 38.5% is after the three compiler fixes it
surfaced (below). opt3 is the current best (41.7%, 41.96% with the wrap-around drain).

## opt1 — hoist the p→fp16 convert (+1.4%)

`sc_vec2` did rescale → `sum(p)` → `p.to(fp16)` cvt (DOT2 operand). Moving the cvt
to the **top** of `sc_vec2` overlaps it earlier; `compute_dot1_qk` still leads the
dot1 stage. 36.6 → 36.8%.

## opt2 — interleave DOT1 mfma with sum+cvt (negative, but surfaced 3 real bugs)

Reversed dot1 so `sc_vec2` runs **before** `compute_dot1_qk`, to let the llir
scheduler interleave the QK mfma with the sum/cvt VALU. **−8%** — the reversal puts
the `warp_predicate` *branch* ahead of the QK mfma, exposing three backend issues:

1. **block-placement out-of-lining.** `MachineBlockPlacement` laid the cold rescale
   block ~50/50. Fix: `gl.warp_predicate(…, unlikely=True)` → branch weights
   `[1,2000]` → cold block out of line. (Committed to triton `llir_FAv3` as a
   frontend param; opt3 no longer needs it.)
2. **MachineSink leaked exp/fma.** `MachineSink` sinks dot2's `exp2`/`fma` (VEC1)
   toward its consumer (dot1's `sum`/`cvt`) **across** the mem2 `s_barrier` —
   `s_barrier` is `IntrNoMem`, so it's not a code-motion fence for pure ALU ops.
   Fix: `DISABLE_LLVM_OPT=disable-machine-sink`.
3. **the llir scheduler was silently rolling back its ENTIRE interleave.** The QK
   region is the first span of the loop's merge block, so its `Begin` is a PHI; the
   plugin moved an mfma above a phi → invalid IR → the whole warp-pipeline
   interleave reverted (the `interleaved N/M` log prints *before* the verify, so it
   looked applied). Every "+llir" number was actually un-interleaved. Fix in
   `LlirSchedPlugin.cpp`: **advance the region start past leading phis** → interleave
   applies → **+8%** (32.6 → 38.5%). A second tweak (adaptive group weight for
   VALU-light stages) spreads the QK mfma instead of clustering; perf-neutral.

Lesson: "branch-leads-dot1" is the wrong shape; it just happened to surface bugs.

## opt3 — hoist the rescale into mem2 (+2%, current best)

Split the rescale out of `sc_vec2` into `rescale_lazy()` and run it in the **mem2**
stage (with LRK/ACV), leaving dot1 as `[sum + cvt] + QK mfma` with **no branch
ahead of the mfma** — the branch's latency overlaps the mem stage. The rescale for
tile *i+1* uses `alpha` from tile *i*'s DOT2 `sc_vec1` (live in tile *i*'s mem2);
prologue rescales tile 0, drain rescales n-2/n-1 (n-3 done by the loop's last mem2).
Semantically identical. **38.5 → 41.7%/SIMD, ~1156 TFLOPS.**

## Wrap-around barrier — the slot force-LOCAL can't reach

The "always emit LOCAL cluster barriers" win above covers 3 of the 4 dot boundaries.
The 4th — the **wrap-around barrier** (cluster 0, loop-top QK) — behaved as if bare:
the loop-top QK stage carried a staggered `lgkmcnt(14→0)` instead of one drain.

**Root cause — not a missing barrier.** The wrap-around barrier's LOCAL *release*
fence **does** ask for the drain, but `SIMemoryLegalizer` materializes it as
**`S_WAITCNT_soft`** — an *advisory* wait that `SIInsertWaitcnts` is free to relax.
At the seven other cluster boundaries the soft wait survives as
`s_waitcnt lgkmcnt(0)`; at the loop *header* `SIInsertWaitcnts` **deletes** it and
re-places minimal **per-consumer** waits instead (`lgkmcnt(14), 13, 12, …` before
each mfma), because the first mfma only needs 2 of the 16 backedge-carried
`ds_read`s. The staggering is the backend deliberately maximizing overlap. Confirm by
diffing the *"SI Memory Legalizer"* vs *"SI insert wait instructions"* MIR dumps
(`DISABLE_LLVM_OPT=print-after-all`):

```
after SI Memory Legalizer          after SI insert wait instructions
  SCHED_BARRIER 0                    SCHED_BARRIER 0
  S_WAITCNT_soft .Lgkmcnt_0   <--    S_BARRIER            (soft wait deleted)
  S_WAITCNT_lds_direct               V_PK_ADD
  S_BARRIER                          S_WAITCNT .Lgkmcnt_14   <-- re-placed
                                     V_MFMA ...  (_13, _12, ...)
```

**So no fence/barrier change can fix it** — everything the memory model emits is
soft; the drain must be a **hard** wait.

**Fix, in the lowering (default on).** `ConvertWarpPipeline` emits
`amdgpu.memory_counter_wait(ds = 0)` immediately before the cluster-0 barrier —
arch-portable (`s_waitcnt lgkmcnt(0)` on gfx9, `s_wait_dscnt 0` on gfx12+, encoding
derived from the ISA version) and hard, so `SIInsertWaitcnts` must honor it. With no
env flags:

```
.LBB0_19 (latch):   s_or exec; s_add x5; s_cmp; s_cbranch   <- all s_xxx here
.LBB0_20 (header):  s_waitcnt lgkmcnt(0); s_barrier;        <- drain + boundary
                    v_pk_add; v_mfma ...                    <- MFMA stage clean
```

Loop: 15 staggered waits → **4 single `lgkmcnt(0)` drains**. **Perf-neutral** on fav4
(~1157 TFLOPS either way, 41.96%) — unlike the *bare*-barrier PV case above, these
staggered waits were already hidden by inter-wave scheduling, so this is a
cleanliness/robustness fix rather than a perf one. Inter-wave a16w16 warp-pipeline
GEMM: correct, within run-to-run noise. `TRITON_WP_NO_WRAP_DRAIN` opts out.

**Dead end:** `TRITON_WP_WRAP_BOTTOM` (move the wrap-around barrier to the loop
bottom) also drains once, but it reverts the top-barrier prototype (`fc55d65df`) and
costs **−1%** — and it was never needed; the hard drain alone yields the layout above.

## Run recipe (opt3)

```bash
HIP_VISIBLE_DEVICES=5 \                    # rocm-smi GPU[4] (shared box)
DISABLE_LLVM_OPT=disable-machine-sink \    # opt2 finding #2 (exp/fma leak)
LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \   # or the plugin regresses hard
FA_MODULE=fav4 python bench.py --seqlen 16320
```

- Needs triton from `llir_FAv3` (warp_predicate + `unlikely` + force-LOCAL barriers +
  the wrap-around hard drain) and `libLlirSched.so` rebuilt with the phi-rollback fix.
  The wrap-around drain is on by default — no env var needed.
- **Cache trap:** `DISABLE_LLVM_OPT` / `LLVM_PASS_PLUGIN_*` / `TRITON_WP_*` are NOT
  in Triton's cache key → `rm -rf ~/.triton/cache` when switching configs, or you
  silently benchmark a stale kernel.
- **Seqlen:** `ceil(N_CTX/64)` must be odd (static_assert). 16320 OK, 16384 not.
- Traces: `/data/fav4_opt{1,2,3}_*_seqlen16320_se0_all4simd_att`.

---

# FAv4 — opt4 / opt5 / opt6, and the compiler defects they exposed

Continuation of the opt1–opt3 log above. Same shape (`b1 h64 d128 sq16320 fp16`,
rocm-smi GPU[4] = `HIP_VISIBLE_DEVICES=5`).

**Metric note (important).** `process_json.py`'s `"mfma efficiency"` is
`total_mfma_cycles_in_loop / average_iteration_duration` — that is **per WAVE**.
With `waves_per_eu=2` the matrix unit's occupancy is **2x** that number. Both are
quoted below (`eff/wave` and `eff/SIMD`). The opt1–opt3 table above lists per-wave
numbers despite what its header used to say.

## Where it ended up

| step | TFLOPS | eff/wave | eff/SIMD | iter cyc |
|---|---:|---:|---:|---:|
| opt3 (previous best) | 1158 | 41.95% | 83.9% | — |
| opt4 alone (no scheduler fixes) | 1151 | 41.24% | 82.5% | — |
| + max-counting/fold + head fix | 1173 | 43.83% | 87.7% | 4672 |
| + `sched_group_barrier` mode + scalarize-fma | 1181 | 45.47% | 90.9% | 4504 |
| + opt5 (`qk_scale` on Q) | 1189 | 46.31% | 92.6% | 4423 |
| **+ opt6 (`s_nop` at mem-stage head)** | **1196** | **46.78%** | **93.6%** | **4377** |
| reference: no llir scheduler at all | 1132 | 37.97% | 75.9% | 5394 |

For scale: 4377 cyc/iteration against 4096 cyc of mfma work (64 mfma x 32 cyc x 2
waves) leaves only 281 cycles of non-mfma time. The kernel is **spill-free at 256
VGPRs** (`vgpr_spill_count=0`, `private_segment_fixed_size=0`, no `v_accvgpr` in the
loop), so there is no register headroom for further work — watch
`vgpr_spill_count` on any change.

## opt4 — split the exp2 burst across both DOT stages (`2f5b9ee`)

Once opt3 moved the rescale out of DOT1, VEC1 (DOT2) had ~2x the VALU of VEC2
(DOT1). Fix: `sc_vec1` computes `t = fma(qk, scale, -m_new)` and exp2's only the
LEFT half, handing the right half's *argument* `t_r` to `sc_vec2`, which exp2's it
in the DOT1 stage and rejoins before the sum. Exact (exp2 is elementwise) and
register-neutral. exp per stage 0/33 -> 16/17; llir-weighted VALU 51/105 -> 83/91.

`_split_halves`/`_concat_halves` are reshape/permute/split and join/permute/reshape
with `convert_layout(..., assert_trivial=True)`, so the compiler *proves* the split
is free — confirmed in the ISA: whole-kernel `v_mov` count unchanged (114 both
ways). Same pattern as `split_subtile()` in the upstream `mxfp_fa_gfx1250` example.

**opt4 alone is a small regression** (1151 vs opt3's 1158). The win came from three
defects it exposed:

### (a) The max3 reduction was invisible to the interleave

`valuWeight()` returned **0** for `maximum`/`minimum` unless opted in, so the
softmax row-max reduction was never grouped and its 16 `v_maximum3` piled up
*before* the stage's first mfma, covered by nothing: **76 cycles stranded at the PV
head while that stage's own windows sat 92 cycles under-filled.** Counting them
(fold-aware, so the inner max/mins that isel merges into `v_maximum3` count 0) is
now the default: head 76 -> 0 cyc, fill 292 -> 368/384, **+1.1%**.
`LLIRSCHED_WP_NOCOUNTMAX` / `NOMAX3FOLD` opt out.

Also fixed: the reverse group-walk only closed a group at G weight, so the
front-most partial run got no mfma ahead of it (2 x exp2 stranded at the QK head).
Now a leftover mfma is spent on it. **+0.2%.**

Tried and rejected: sizing groups by the window (`G=6`) instead of `X/Y`. Bigger
groups made codegen pile a heavier tail into the last sub-region (asm overflow
16/20 -> 32/32 cyc) and measured **worse** (1162 vs 1174). `LLIRSCHED_WP_WINDOWG`
opts in.

### (b) LLVM `SIPreEmitPeephole` bails on TRANS, hiding packed ops from unpacking

`collectUnpackingCandidates()` returns at the first instruction that is
`isNeverCoissue() && !isUnpackable`, and `SIInstrInfo::isNeverCoissue()` has
`if (isTRANS(MI)) return true;`. So in `mfma + 2 x v_exp + v_pk_add` the scan
**terminates at the first `v_exp`** and never reaches the `v_pk_add`, which stays
packed (8 cyc, cannot co-issue with an MFMA at all) even though the window had room.
LLVM treats a TRANS op as ending the co-exec window; on gfx950 it does not.

Fix: make a never-coissue non-unpackable op a *latency consumer* rather than a scan
terminator (stop only at a following MFMA/DOT, which opens its own window). Proved
with `llc` A/B on identical IR: `v_pk_add` 31 -> 27, `v_add_f32` 105 -> 113,
`v_pk_mul` unchanged. End-to-end +0.2%. **Later made redundant** by (c) — the patch
lives in a `/data/llvm-pin` worktree at triton's pin and is NOT needed for the
current recipe.

### (c) `ScalarizePackedFOps` missed `fmuladd`/`fma` (`fbe309e1d`)

The pass matched only `m_BinOp` (FMul/FAdd/FSub); `llvm.fmuladd`/`llvm.fma` are
intrinsic **calls**, so vector ones survived as `v_pk_fma_f32`. Extended with
`maybeReplaceVectorFMAWithScalarFMAs()` (per-element scalar intrinsics, fast-math
flags copied). PV fill 344 -> 368/384 with zero packed ops left; **+0.3%** on the
physical interleave, **+0.6%** with `sched_group_barrier`.

Consequence worth noting: with all packed fp ops gone before codegen, the
peephole bug in (b) is unreachable, so **no custom LLVM build is needed** — the
whole optimization now runs on triton's stock pinned LLVM.

## What FlyDSL does (and does not) do differently

- **LLVM:** `/root/llvm-project` at upstream `7f77ca0db` (Mar 2026, 23.0.0git),
  **no local patches**. Not a fork, no custom pass. So its edge is not the backend.
- **Hints:** it emits `llvm.amdgcn.sched.group.barrier(mask, size, syncID)` — 321 of
  them vs 60 plain `sched.barrier` — with masks MFMA `0x8` / VALU `0x2` / TRANS
  `0x400` and **one syncID per cluster**, e.g. `16 x [MFMA 1] + 10 x [VALU 5] +
  6 x [TRANS 3]`. Plain `sched_barrier` is used only at cluster boundaries.
- **It never reorders instructions.** Its clusters are written mfma-led with the
  VALU after, so IGroupLP only has to *confirm* a schedule, not construct one.
- `_s_nop(7)` (raw side-effecting `llvm.inline_asm`, since ROCDL has no `s.nop` op)
  as the first statement of every mem cluster — see opt6.

## `sched_group_barrier` mode in llirSched (`LLIRSCHED_WP_SGB=1`)

Motivation: `sched_barrier(0)` is only advisory to the machine scheduler, and we
measured codegen consolidating a stage's last two sub-regions anyway. IGroupLP
instead *builds* the requested pipeline. `declareRegionGroups()` counts the region
(M mfma, per-op cycle lists for VALU/TRANS, max3-fold-aware) and emits the
declaration **after every real op of the region** (IGroupLP forms groups scanning
upward; a declaration at the top yields empty groups and silently does nothing).

Three things had to be right, each found the hard way:

1. **Declaration order must follow the dependency order.** In `sc_vec2` the
   adds/cvt *consume* the exps (`exp2(t_r) -> concat -> sum/cvt`), so a VALU-first
   declaration is unsatisfiable: IGroupLP formed ~5 of 16 groups, left 11 mfma bare
   and overflowed 216 cyc -> **1114 TFLOPS**. Declaring TRANS-first for QK and
   VALU-first for PV (chosen automatically from the first co-issuable op in the
   region) fixed it -> **1174**. FlyDSL orders its `exp_pairs`/`valu_pairs`
   per cluster by hand for the same reason.
2. **Scalarize is load-bearing.** IGroupLP counts *instructions*; a packed op
   becomes two after `ScalarizePackedFOps`. Without scalarize, QK keeps 24 cyc of
   `v_pk` and fill drops 336 -> 288 (-1.2%).
3. **Group sizes must be packed by CYCLES, counted in INSTRUCTIONS.** Deriving
   `g1 = M - g0` over-claimed groups for a class that could not fill them (3 starved
   windows). Now each class's group count comes from its own cycle cost, and the
   per-group instruction count from packing the region's ops in program order.

Not a silver bullet: SGB **without** a feasible starting order fails badly (QK
60/384 fill, 156 cyc stranded), and interleave+SGB was worse than either
(1120). Only pure-declaration + dependency order + scalarize wins.

## opt5 — fold `qk_scale` into Q before the loop (`75f40a0`)

Removes both uses of the scale from the loop: `fma(qk, qk_scale, -m_new)` becomes a
plain subtract, and the row max drops its scale multiply (which also shortens the
row-max dependency chain — `max(qk)` feeds the compare directly). Costs one extra
fp16 rounding of Q: max error 1.22e-04 vs 6.10e-05, both far inside the 1e-3
tolerance. Selectable via the `SCALE_ON_Q` constexpr kernel arg (default = on);
`False` reproduces pre-opt5 numerics bit-for-bit at ~-1%.

## opt6 — `s_nop` at the head of each mem stage (FlyDSL trick)

`LLIRSCHED_WP_MEMNOP=k` emits `k x s_nop 7` (8 idle cycles each) at the head of
every mem stage, via the real `llvm.amdgcn.s.nop(i16)` intrinsic. Not wasted time —
it is **phase tuning**: in the two-wave ping-pong a delay at the mem-stage head
shifts this wave's `ds_read` burst so it does not collide with the other wave's LDS
traffic and issue slots.

| idle cyc / mem stage | 0 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| TFLOPS | 1190.7 | 1193.1 | 1194.7 | **1196.5** | 1176.8 |

Sharp optimum at 24 (`k=3`), falling off hard after. Paying 96 idle cycles/iteration
*reduced* iteration time by 46 cycles.

Bug worth remembering: the mem2 stage **spans 2-4 basic blocks** (the lazy-rescale
`warp_predicate` branch splits it), so a per-basic-block span walk never sees its
closing barrier and skips it — only mem1 got nops. Stage classification must be done
in function layout order, not per block.

## The `fsub` sink — ISel's pre-RA scheduler (proven)

Symptom: 16 of the 32 `v_sub` computing the softmax exponent leave the PV stage and
come to rest in the following mem stage, where no mfma can hide them (PV fill
296/384 instead of ~360, ~64 cyc/iteration).

- **Not opt6**: the sub distribution is byte-identical with 0, mem1-only and
  all-stage `s_nop`s.
- **It is opt5**: pre-opt5 all 32 `v_fma` stayed in PV (0 in the mem stages). The
  2-operand `fsub` is cheaper for the scheduler to move than the 3-operand fma.
- **Which pass:** ISel's input (the `.llir`) has `fsub = 34` in the PV region and
  `0` in the mem region; **ISel's output (first MIR dump) already has 18 / 16**, and
  every later pass preserves it. `llc -pre-RA-sched=linearize` on the *same* IR
  keeps all 34 in the DOT stage -> the pre-RA list scheduler is the mover.
  The four `disable-sched-*` heuristic flags change nothing, because the freedom
  comes from the DAG having **no chain edge** on a pure `fsub` — `s_barrier` and
  `sched.barrier` are chain nodes and impose no ordering on chain-free arithmetic.
- Only half move because opt4 gave `t_r`'s subs their only consumer in the *next*
  stage; `t_l`'s stay with PV's own exps. The second unrolled half keeps all 34 —
  a heuristic decision, not a rule.

`TRITON_PRE_RA_SCHED=<source|linearize|list-ilp|...>` was wired into `llvm.cc`
(`setLLVMOptionFromString` -> `addOccurrence`, since `-pre-RA-sched` is a
`RegisterPassParser` option, not `cl::opt<bool>/<std::string>`; registered in
`CACHE_INVALIDATING_ENV_VARS` so it is part of the cache key).

**`linearize` is a proof instrument, not a fix: 877 TFLOPS (-26%).** It doubles
iteration time (9119 vs 4377 cyc) and drops to 44.9%/SIMD, with no spills in either
build. Causes, all originating at ISel: `s_waitcnt lgkmcnt` in the loop goes **4 ->
41** in the staggered per-consumer form (every `ds_read` is consumed where the
source wrote it, so LDS latency is exposed per read), loop barriers go **18 -> 24**,
and **4 of them end up adjacent** (`s_barrier / s_waitcnt vmcnt(2) / s_barrier`) —
an adjacent pair means that stage's body migrated into its neighbour, which is the
compute/`ds_read` interleaving that destroys the ping-pong. `ConvertWarpPipeline`
is *not* at fault: the MLIR/`.llir` is identical in both builds. The DAG scheduler's
real job here is hoisting loads away from their uses, and that is worth ~2x.

The remaining fix is kernel-side: carry `qk_r` + `m_new` and compute
`t_r = qk_r - m_new` inside `sc_vec2`, so those subs are in the DOT1 region by
construction rather than at ISel's discretion. Register-neutral, but the kernel is
at exactly 256 VGPRs — check `vgpr_spill_count` afterwards.

## Component ablation — what is actually required

| component | required? | measured if removed |
|---|---|---|
| `AMDGCN_SCALARIZE_PACKED_FOPS=1` (incl. the fma extension) | **yes** | `v_pk_fma` survives, PV 344/384 -> ~1174; no scalarize at all -> ~1160 |
| llir scheduler + `LLIRSCHED_WP_SGB=1` | **yes** | ~1132, 75.9%/SIMD, QK/PV overflow 88/92 cyc |
| `DISABLE_LLVM_OPT=disable-machine-sink` | **yes** | PV's `fma 32`/`exp 16` migrate into mem2, PV windows 368 -> 112/384, **~1140 (-3.5%)** |
| the `SIPreEmitPeephole` TRANS patch | **no** | identical inventory and perf — superseded by scalarize-fma |

SGB does **not** protect against MachineSink: MachineSink runs *before* the machine
scheduler, so it moves ops out of the region entirely and IGroupLP never sees them.

## Run recipe (current best, stock pinned LLVM)

```bash
HIP_VISIBLE_DEVICES=5 \                    # rocm-smi GPU[4] (shared box)
AMDGCN_SCALARIZE_PACKED_FOPS=1 \           # required (needs the fma extension)
LLIRSCHED_WP_SGB=1 \                       # sched_group_barrier hints
LLIRSCHED_WP_MEMNOP=3 \                    # 24 idle cyc at each mem-stage head
DISABLE_LLVM_OPT=disable-machine-sink \    # required
LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
FA_MODULE=fav4 python bench.py --seqlen 16320
```

- `LLVM_PASS_PLUGIN_*` and `LLIRSCHED_*` are read by the plugin, not by Triton, so
  they are **not** in the cache key — `rm -rf ~/.triton/cache` when changing them.
  (`DISABLE_LLVM_OPT` and `TRITON_PRE_RA_SCHED` *are* registered as
  cache-invalidating.)
- `ceil(N_CTX/64)` must be odd (static_assert): 16320 OK, 16384 not.
- Traces: `/data/fav4_opt{4,5,6}_*_seqlen16320_se0_all4simd_att`.

## Measurement protocol: rocprofv3 kernel time, prepared launch

Reported FA numbers now come from `scripts/fa_kernel_time.py`, not from `do_bench`, so
attention is measured the same way as the GEMM kernels
(`scripts/run_perf_table.py:run_rocprof_trace`):

```bash
FA_MODULE=fav4 <plugin env from the recipe above> \
python scripts/fa_kernel_time.py --seqlen 16320        # --launch prepared is the default
```

- `rocprofv3 --kernel-trace -f csv --kernel-include-regex gluon_attn_fwd`, with
  `AMD_SERIALIZE_KERNEL=3`, averaging the **last N of 1000** dispatch durations; rows are
  sorted by numeric `Dispatch_Id` first (per PR #50 -- a lexicographic sort puts 999 before
  1000). TFLOPS = `compute_flops(..., causal=False) / avg_kernel_ns`.
- **Prepared launch** (`bench.py --prepared`, using `scripts/prepared_kernel.py` from PR
  #50): binds the specialization and the whole argument list once, then re-enters the
  compiled launch stub, so no Python argument binding happens between dispatches. Validated
  against the ordinary launcher every run -- bit-identical output (max diff 0.00e+00).
- **Rotating buffers**: `bench.py` derives the set count from `--rotating-buffer-size`
  (512 MB default, matching the GEMM bench). Necessary because a small FA shape stays
  MALL-resident: at S=1088 one set is 71 MB and reports 667 TFLOPS, eight sets report 626.
  Large shapes (S=16320 -> 1.07 GB) already exceed the cache with one set.

**What the switch does and does not change.** At S=16320 all methods agree within 0.5%:
`do_bench` 1243, rocprof/jit 1238, rocprof/prepared 1245 -- the kernel runs 7 ms, so the
async queue hides the ~40 us of host binding entirely (wall/launch at S=1088: 58.01 us jit
vs 57.98 us prepared, i.e. no gap to remove). Prepared launch is adopted for **protocol
consistency with the GEMM path**, not for a speedup: order-balanced A/B/A/B gives prepared
1218.5/1243.2 vs jit 1229.0/1238.5, a -0.24% mean inside the +-1.3% session noise. The
mechanism should pay off as 1/kernel-duration, i.e. in the us-scale GEMM regime.

Bigger error sources than method choice, both measured on this box:
- **die**: GPU[0] is repeatably ~4% faster than GPU[4] at equal settings (A/B/A/B 1241/1244
  vs 1198/1197). Same average XCD clock (1553 vs 1545 MHz) but a lower *floor* XCD
  (1416 vs 1449) while drawing 13 W more of the 1400 W cap -- FA's wall time is set by the
  slowest XCD.
- **thermal/duty-cycle state**: a cool part reads high (cold 1st iter 976 -> 1256 plateau
  as DPM ramps), so single-shot `bench.py`-style numbers sit 4-7% above steady state.
  Alternate A/B/A/B with idle between when comparing anything under ~1.5%.
