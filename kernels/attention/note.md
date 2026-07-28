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
now the default: head 76 -> 0 cyc, fill 292 -> 368/384, **+1.1%**. (The
`LLIRSCHED_WP_NOCOUNTMAX` / `NOMAX3FOLD` opt-outs have since been removed -- both
behaviours are hardwired to the measured winner.)

Also fixed: the reverse group-walk only closed a group at G weight, so the
front-most partial run got no mfma ahead of it (2 x exp2 stranded at the QK head).
Now a leftover mfma is spent on it. **+0.2%.**

Tried and rejected: sizing groups by the window (`G=6`) instead of `X/Y`. Bigger
groups made codegen pile a heavier tail into the last sub-region (asm overflow
16/20 -> 32/32 cyc) and measured **worse** (1162 vs 1174). The
`LLIRSCHED_WP_WINDOWG` opt-in has since been removed.

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

## `sched_group_barrier` declaration in llirSched (now the default)

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
| `AMDGCN_SCALARIZE_PACKED_FOPS=1` (incl. the fma extension) | *was yes, now **no*** | `v_pk_fma` survives, PV 344/384 -> ~1174; no scalarize at all -> ~1160. **Superseded** -- see the group-size fix at the end of this file; neither kernel needs the pass now |
| llir scheduler + `LLIRSCHED_WP_SGB=1` | **yes** | ~1132, 75.9%/SIMD, QK/PV overflow 88/92 cyc |
| `DISABLE_LLVM_OPT=disable-machine-sink` | **yes** | PV's `fma 32`/`exp 16` migrate into mem2, PV windows 368 -> 112/384, **~1140 (-3.5%)** |
| the `SIPreEmitPeephole` TRANS patch | **no** | identical inventory and perf — superseded by scalarize-fma |

SGB does **not** protect against MachineSink: MachineSink runs *before* the machine
scheduler, so it moves ops out of the region entirely and IGroupLP never sees them.

## Run recipe (current best, stock pinned LLVM)

```bash
HIP_VISIBLE_DEVICES=1 \                    # rocm-smi GPU[0], the fast die (ATT: use 5)
DISABLE_LLVM_OPT=disable-machine-sink \    # required
LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
FA_MODULE=fav4 python bench.py --seqlen 16320
```

### Environment variables

Three independent things get configured here, and conflating them is the usual source of
an unreproducible number. **Nothing in group A is part of the llir scheduler** -- they
configure other passes or the plugin loader -- and **nothing in group B is needed to
reproduce any number in this file.**

**A. Required, and not scheduler knobs**

| var | owned by | why it is required |
|---|---|---|
| `LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so` | LLVM | how an out-of-tree pass plugin gets loaded at all |
| `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1` | Triton (`gfx950-tutorial-v1.1` pin) | keeps the TargetMachine for plugins; without it `optimize_module` runs all of O3 with no target machine and codegen regresses |
| `DISABLE_LLVM_OPT=disable-machine-sink` | LLVM pass manager | disables **MachineSink**, which moves exp/fma out of the scheduled region (opt2 finding #2). It runs on MIR, long after any IR pass, so this is not something the scheduler can handle itself |

**B. llir scheduler knobs -- all optional**

| var | effect |
|---|---|
| `LLIRSCHED_WP_NOSGB` | fall back to physically reordering + `sched.barrier(0)` pinning instead of declaring with `sched_group_barrier`. Slower; kept for A/B |
| `LLIRSCHED_WP_MEMNOP=k` | override mem-stage head pacing. Default **2**; `0` disables |
| `LLIRSCHED_WP_NOOVERCAP` | keep the fitting packer even when a region's VALU exceeds the mfma windows, to bisect a regression to the over-capacity path |
| `LLIRSCHED_WP_DEBUG` | per-region model, counts, group sizes, chosen window |

The two that used to be load-bearing are now defaults. Verified: with **no `LLIRSCHED_*`
set at all**, fav3 and fav4 SOQ=1/0 build byte-identical assembly to the old
`LLIRSCHED_WP_SGB=1 LLIRSCHED_WP_MEMNOP=2` recipe. `MEMNOP`'s default of 2 is not the 3
opt6 originally chose -- re-swept after opt8 / the dependency-ordered SGB / mixed-class
windows / the `fneg` fix; see the retune section below. The six experiment flags that
used to sit alongside these are gone: each had a losing alternative, so the loser was
deleted rather than left as a flag.

**C. Caching.** `LLVM_PASS_PLUGIN_*` and `LLIRSCHED_*` are read by the plugin, not by
Triton, so they are **not in the Triton cache key** -- change one and you must
`rm -rf ~/.triton/cache` or use a fresh `TRITON_CACHE_DIR`, or you will silently
re-measure the previous build. `DISABLE_LLVM_OPT`, `AMDGCN_SCALARIZE_PACKED_FOPS` and
`TRITON_PRE_RA_SCHED` *are* registered as cache-invalidating.

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

## fav3 — the over-capacity `sched_group_barrier` algorithm

`declareRegionGroups` assumes the stage's VALU work FITS the mfma co-exec capacity
(`24 cyc x M`) and spreads it so no group overflows. fav3 breaks that assumption. Its
dot stages carry more VALU than the mfma shadows can hide:

```
fav3 QK: 476 cyc of VALU vs 384 cyc of capacity   -> minGroups 20 for 16 mfma
fav3 PV: 496 cyc                                  -> minGroups 21 for 16 mfma
```

The balanced packer's response was to merge cheap pairs into 48-cycle groups, which the
assembly faithfully reproduced (three windows of `12 x v_mul`, four of `6 x v_exp`).
IGroupLP was not failing -- the *request* was infeasible.

When the work does not fit, the question changes from "how do I spread it" to "which ops
get a window, and what shape do the rest take". `declareRegionGroupsOverCap` (triggered
automatically when `total > 24*M`; opt out with `LLIRSCHED_WP_NOOVERCAP`):

1. **Non-splittable ops claim windows first** (exp2, the max3 reduction, `v_cvt_pk`,
   permlane). Packing cannot help them, so they get first call on the capacity.
2. **Remaining capacity buys packable coverage from the END of the stage backwards** --
   mfmas inserted in reverse program order -- which keeps the uncovered remainder
   contiguous and leaves it where it already sits: at the head in QK (the rescale muls),
   in the middle in PV (the qk_scale fmas, between the max3 reduction and the exp2s).
3. **Covered packed ops are scalarized by the plugin itself** (`scalarizePackedFP`,
   narrow: `fmul/fadd/fsub/fma` on `<N x float>`; `fptrunc` excluded because a packed
   convert IS one `v_cvt_pk`, `<N x half>` excluded because `v_pk_*_f16` is the natural
   form). **Uncovered ops stay packed** -- nothing hides them either way, and one
   `v_pk_mul_f32` retires two elements in one issue where two `v_mul_f32` need two.

**So fav3 must NOT set `AMDGCN_SCALARIZE_PACKED_FOPS`** -- that pass splits every packed
op in any block containing an mfma, including the ones this path deliberately keeps
packed. fav4 still needs it.

Slot model as specified: 1 mfma = 6 slots, unpacked op = 1, packed op = 2, exp2 = 2,
permlane = 5.

### Mixed-class windows

A window is one mfma's 24-cycle shadow and may hold **several groups of different
IGroupLP classes**: `[MFMA 1][VALU 1][TRANS 1]` asks for one mfma and then a sub and an
exp2 behind it. fav3's PV stage ends in exactly that pair and used to spend two windows
on 12 cycles of work. The merge step is class-agnostic too, so a VALU fragment can share
a window with a TRANS fragment.

```
before:  V6 V10 V5 *V14 V4 T3 x9 T5 V1 T1               worst group 40 cyc
after:   [V6]24 [V6]24 [V4]16 [V5]36 *[V14]112 [V4+T1]24 [T3]24 x10 [T1+V1+T1]20
                                        ^^^^^^^        ^^^^^^^^^^^^
                                    mixed class     one mfma covers exp+sub+exp
```
(`*` = declared with no mfma in front of it, i.e. deliberately uncovered.)

PV overflow 64 -> 32 cyc, empty windows 1 -> 0. A freed window is fed back into the
coverage budget by a fixed-point loop, though on fav3's PV there is nothing to buy: its
unpackable work is 368 cyc of a 384 cyc capacity, so only 16 cyc (2 `v_pk_fma`) can ever
be covered while rule 1 holds. 14 stay packed by arithmetic, not by a packing failure.

**Measured** (GPU[0], kernel-time protocol): 1139.5 (balanced packer + scalarize) ->
1158.3 (over-capacity path) -> **1165.0** (+ mixed-class windows) = **+2.2%**. fav4 is
provably unaffected: its stages are under capacity, it never enters this path, and its
compiled asm hashes identically before/after.

## The `fneg` source-modifier bug (`valuWeight` must return 0)

`fneg` and `llvm.fabs` are **not instructions** on AMDGPU: they fold into the consumer as
source modifiers (`v_fma_f32 v0, v0, s44, -v129`). `valuWeight` counted `fneg` because it
is an FP `UnaryOperator`, which inflates a declared group by an instruction that never
gets emitted. IGroupLP cannot fill that group and **its pipeline solver then discards the
schedule for the entire region**, not just that group.

Found via fav4 `SCALE_ON_Q=0`, whose VEC1 computes `fma(qk, qk_scale, -m_new)`:

```
seg1 (a QK stage): MFMA 6xfma MFMA 6xfma MFMA 6xadd MFMA 6xadd MFMA 3xadd+cvt+add
                   MFMA 5xcvt MFMA 4xcvt          <- 7 of 16 groups placed
                   12xexp2 15xadd mov s_nop permlane 2xadd    <- no mfma at all
                   8xMFMA                                     <- the rest dumped
```
9/16 windows filled, longest bare-mfma run 8. The tell: the two QK regions declared
**39 vs 38** ops, and diffing them in the `.llir` showed the only difference was one
`fneg` (its twin had none, and scheduled 16/16). `SCALE_ON_Q=1` has zero `fneg` in the
whole kernel, which is why this stayed hidden. `dependsOnAny` was not at fault -- it is
properly transitive and stops at PHIs.

Neither pre-reordering before the declaration (still 9/16; this was
`LLIRSCHED_WP_SGB_REORDER`, since removed) nor interleave mode (fixes the clustering at
15/16 but is slower overall) worked around it. The weight model was the fix.

After: all four stages 16/16, declarations symmetric, 4725.7 -> 4634.2 cyc/iter (-1.9%),
MFMA eff 86.7 -> 88.4%/SIMD, 1215.5 -> 1220.2 TFLOPS. fav3 1164.5 -> 1166.5.
**Diagnostic worth remembering:** two stages that should be identical declaring different
op counts in the `[sgb]` debug lines.

## `LLIRSCHED_WP_MEMNOP` retuned: 3 -> 2

opt6 picked k=3 before opt8, the dependency-ordered SGB declaration, mixed-class windows
and the `fneg` fix. Re-swept at b1 h64 d128 sq16320 fp16, kernel-time protocol, 45 s
cool-down, >=2 samples each:

| `MEMNOP` | fav4 SOQ=0 | fav4 SOQ=1 |
|---|---|---|
| 0 | 1227.4 | -- |
| 1 | 1221.6 (noisy, 0.8% spread) | -- |
| **2** | **1234.3** (+-0.13%) | **1244.9** |
| 3 (old default) | 1218.8 | 1239.7 |

k=2 wins for **both** settings: +1.3% on SOQ=0, +0.4% on the default. Removing the nops
entirely also beats k=3 on SOQ=0 (1227.4), but k=2 is better still. **All numbers below
use `LLIRSCHED_WP_MEMNOP=2`.**

## `SCALE_ON_Q=0` evaluated (`--scale-on-q 0`)

Applying `qk_scale` per element inside VEC1 instead of pre-scaling Q costs **0.9%** at
matched settings (1234.3 vs 1244.9) and is **more accurate**: max_err 6.10e-05 vs
1.22e-04, since the pre-scaled path rounds `q * qk_scale` back to fp16 before the loop.
Op-wise it is nearly a wash -- QK is a 1:1 `sub` -> `fma` swap (75 valu both), PV adds 2
ops (one max3, one mul, to materialise the negated row max). Both builds are 256 VGPRs,
0 AGPRs, 0 scratch, occupancy 2. Keep `SCALE_ON_Q=True` as the default; flip it only if
the tighter numerics are worth 0.9%.

## Status: gluon vs FlyDSL (2026-07-26)

Same shape throughout: **B=1 HQ=64 HK=64 S=16320 D=128 fp16 non-causal**.

- **TFLOPS**: GPU[0] (`HIP_VISIBLE_DEVICES=1`), `scripts/fa_kernel_time.py` -- rocprofv3
  kernel time, prepared launch, `AMD_SERIALIZE_KERNEL=3`, 45 s idle before each run,
  >=2 samples in both orders. Repeatability: fav3 and FlyDSL-eager <=0.1%, the two
  lazy-rescale kernels ~0.5-1% (denser MFMA stream -> power-limited -> clock-sensitive).
- **In-loop MFMA efficiency**: also **GPU[0]**, `att_attn_se0.json` (SE mask 0x1, all 4
  SIMDs, dispatch iteration 8) with `att_target_cu` chosen by `scripts/att_pick_cu.py`
  (CU1 on this die), + `scripts/process_json.py`. **Per-SIMD = 2x the per-wave number**
  `process_json.py` prints. All rows: 126 iterations x 2048 mfma cycles per iteration, so
  the cycle counts are directly comparable across both implementations.
  Re-measured single-die 2026-07-26; the earlier GPU[4] traces agree to **<=0.3%** on
  cyc/iter (4299.9 vs 4306.1, 4420.7 vs 4419.2, 4799.6 vs 4801.7, 4734.2 vs 4747.4,
  6689.9 vs 6694.5), which retroactively validates the cross-die tables this replaces.
- FlyDSL helpers referenced below are archived at `/data/fa_compare/`
  (`fly_iters.py`, `fly_ktime.py`, `att_fly_se0.json`); FlyDSL checkout `/root/FlyDSL`.

### 1. Eager rescale: gluon fav3 vs FlyDSL `lazy_rescale=False`

| kernel | TFLOPS | cyc/iter | eff/wave | **eff/SIMD** | loop |
|---|---:|---:|---:|---:|---:|
| **gluon fav3** (over-cap SGB + mixed-class windows + fneg fix) | **1167.1** | **4801.7** | 42.65% | **85.3%** | 95.4% |
| FlyDSL `dualwave_swp`, `lazy_rescale=False` | 1043.8 | 6694.5 | 30.59% | 61.2% | 97.5% |
| | **+11.8%** | **-28.3%** | | **+24.1 pts** | |

gluon is **+11.9%** on throughput and needs **28% fewer cycles per iteration**. FlyDSL's
eager path is a correctness fallback with no scheduling attention -- it recovers part of
the cycle deficit through clock headroom (it ran at 1820 MHz / 1218 W, being VALU-bound
rather than power-bound, vs ~1450 MHz for ours) and still loses by 11.9%. Turning lazy
rescale off costs FlyDSL **-16.0%** but costs gluon only **-6.2%**; our eager path
degrades ~2.6x more gracefully, which is what the over-capacity algorithm buys.

**Reproduce -- gluon fav3 TFLOPS** (note: **no** `AMDGCN_SCALARIZE_PACKED_FOPS`):

```bash
cd <repo>
HIP_VISIBLE_DEVICES=1 FA_MODULE=fav3 \
LLIRSCHED_WP_SGB=1 LLIRSCHED_WP_MEMNOP=2 \
DISABLE_LLVM_OPT=disable-machine-sink \
LLVM_PASS_PLUGIN_PATH=$PWD/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
python scripts/fa_kernel_time.py --seqlen 16320 --iters 400 --last-n 200
```

**Reproduce -- gluon fav3 MFMA efficiency:**

```bash
cd <repo>/kernels/attention
ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ \
HIP_VISIBLE_DEVICES=5 FA_MODULE=fav3 \
LLIRSCHED_WP_SGB=1 LLIRSCHED_WP_MEMNOP=2 \
DISABLE_LLVM_OPT=disable-machine-sink \
LLVM_PASS_PLUGIN_PATH=$PWD/../../plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
rocprofv3 --att -i att_attn_se0.json -d /tmp/att_fav3 -- \
  python bench.py --rocprof --seqlen 16320 --n-iters 20
python ../../scripts/process_json.py /tmp/att_fav3/ui_output_*
```

**Reproduce -- FlyDSL eager TFLOPS and MFMA efficiency:**

```bash
cd /root/FlyDSL
HIP_VISIBLE_DEVICES=1 python /data/fa_compare/fly_ktime.py 0 400      # TFLOPS

ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ HIP_VISIBLE_DEVICES=5 \
rocprofv3 --att -i /data/fa_compare/att_fly_se0.json -d /tmp/att_fly0 -- \
  python /data/fa_compare/fly_iters.py --seqlen 16320 --hq 64 --batch 1 \
    --dtype fp16 --n-iters 20 --lazy-rescale 0
python <repo>/scripts/process_json.py /tmp/att_fly0/ui_output_*
```

The eager path is one flag on FlyDSL's builder -- `flash_attn_gfx950.py:60` takes
`dualwave_swp_lazy_rescale` (default `True`), gating `rescale_o` vs `lazy_rescale_o` at
line 366/440. The public wrapper exposes it as `lazy_rescale=`
(`flash_attn_interface.py:146`).

### 2. Lazy rescale: gluon fav4 (SOQ=1 / SOQ=0) vs FlyDSL default

| kernel | TFLOPS | cyc/iter | eff/wave | **eff/SIMD** | loop |
|---|---:|---:|---:|---:|---:|
| **gluon fav4, `SCALE_ON_Q=1`** (default) | **1245.5** | **4306.1** | 47.56% | **95.1%** | 95.1% |
| gluon fav4, `SCALE_ON_Q=0` | 1231.2 | 4419.2 | 46.34% | 92.7% | 95.2% |
| FlyDSL `dualwave_swp`, `lazy_rescale=True` (default) | 1242.4 | 4747.5 | 43.14% | 86.3% | 96.8% |

`SCALE_ON_Q=0` is the one row that needs care: it is thermally sensitive and read 1214.7
straight after the hottest kernel with only 45 s idle, then recovered 1224.5 -> 1230.6 ->
1231.2 over three runs at 90 s idle (it had measured 1233-1235 earlier in a cooler
session). Treat it as ~1231 +-4 and do not read a 1% delta off a single sample.

**gluon fav4 and FlyDSL tie on throughput (+0.2%) but not on cycles**: fav4 needs
**9.2% fewer cycles per iteration** (4300 vs 4734) and is **8.8 points ahead on MFMA
efficiency**, yet ends up level in wall time. Clock accounts for part of it, not all:

- total kernel cycles, `126 x cyc_iter / loop_ratio`: fav4 ~571 k, FlyDSL ~618 k, so at
  equal clock fav4 should be **~8% faster**; measured is +0.2%, leaving ~7.8 points to
  explain.
- sampled XCD clocks during each run: fav4 1420-1440 MHz at 1318-1345 W, FlyDSL-lazy
  1492 MHz -- a **4.3%** gap, i.e. roughly half of what is needed.

So **fav4's denser MFMA stream buys cycles and pays for them in clock** on a die already
at 96% of its power cap, but the clock samples (1-2 per run, `amd-smi` at 2 s intervals)
only cover about half the gap; the remainder is not yet accounted for. Candidates worth
checking before quoting this as settled: sampling density, and whether the traced dispatch
(iteration 8 of 20) sits at the same power state as the steady-state timing loop.
The safe reading: **MFMA efficiency is the metric that tracks our scheduling work; TFLOPS
on this board partly tracks the power budget.**

**Reproduce -- gluon fav4 TFLOPS** (`--scale-on-q 1` or `0`):

```bash
cd <repo>
HIP_VISIBLE_DEVICES=1 FA_MODULE=fav4 \
AMDGCN_SCALARIZE_PACKED_FOPS=1 LLIRSCHED_WP_SGB=1 LLIRSCHED_WP_MEMNOP=2 \
DISABLE_LLVM_OPT=disable-machine-sink \
LLVM_PASS_PLUGIN_PATH=$PWD/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
python scripts/fa_kernel_time.py --seqlen 16320 --iters 400 --last-n 200 --scale-on-q 1
```

**Reproduce -- gluon fav4 MFMA efficiency:**

```bash
cd <repo>/kernels/attention
ROCPROF_ATT_LIBRARY_PATH=/opt/rocm/lib/ \
HIP_VISIBLE_DEVICES=5 FA_MODULE=fav4 \
AMDGCN_SCALARIZE_PACKED_FOPS=1 LLIRSCHED_WP_SGB=1 LLIRSCHED_WP_MEMNOP=2 \
DISABLE_LLVM_OPT=disable-machine-sink \
LLVM_PASS_PLUGIN_PATH=$PWD/../../plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
rocprofv3 --att -i att_attn_se0.json -d /tmp/att_fav4 -- \
  python bench.py --rocprof --seqlen 16320 --n-iters 20 --scale-on-q 1
python ../../scripts/process_json.py /tmp/att_fav4/ui_output_*
```

**Reproduce -- FlyDSL lazy (default):** same two commands as FlyDSL eager above with
`--lazy-rescale 1` / `fly_ktime.py 1 400`.

### Gotchas that invalidate these measurements

- **`AMDGCN_SCALARIZE_PACKED_FOPS` no longer applies to either kernel** (group-size fix at the
  end of this file). It used to be required for fav4 and forbidden for fav3. Every recipe and
  number recorded *above* this line was measured with it set for fav4, and those recipes still
  reproduce those numbers -- the pass still exists, it is just no longer needed.
- **`LLVM_PASS_PLUGIN_*` and `LLIRSCHED_*` are not in the Triton cache key.** Use a
  separate `TRITON_CACHE_DIR` per configuration or `rm -rf ~/.triton/cache`, or you will
  re-measure the previous build. (`DISABLE_LLVM_OPT` *is* cache-invalidating.)
- **ATT can silently capture nothing if `att_target_cu` names a harvested CU** -- exit 0,
  a `ui_output_*` directory, but a ~8-35 KB `.att` instead of 65 MB, no
  `se*_sm*_sl*_wv*.json` wave files, and `process_json.py` dies with `'NoneType' object is
  not iterable`. **Check the `.att` size before trusting a trace.** Root cause and fix in
  the section below; the configs here now use `att_target_cu: 4`.
- **The two columns therefore come from different dies** (TFLOPS GPU[0], traces GPU[4]),
  which is sound because cycle counts are die-portable: the same kernel and settings traced
  on GPU[4] and GPU[5] give **4299.95 vs 4295.79 cyc/iter and 47.63 vs 47.67 %/wave --
  0.1%**, despite those dies differing ~0.6% in throughput. Cross-check on the mix: the
  clock implied by (GPU[0] wall time, GPU[4] cycles, 16 workgroups per CU) is 1303 MHz for
  fav4 SOQ=1, 1359 for fav3, 1405 for FlyDSL-lazy and 1654 for FlyDSL-eager -- a plausible
  band, ordered exactly as MFMA density predicts and matching the ~1820 MHz sampled
  directly for the VALU-bound FlyDSL-eager.
- **Thermal state dominates sub-1.5% deltas.** A fav3 run taken 30 s after a heavy series
  read 1130 instead of 1165 (-3%) on a kernel that otherwise repeats to 0.05%. Use
  45-60 s idle and alternate A/B.
- `fly_ktime.py` averages the final 250 of 400 dispatches where the gluon driver is told
  `--last-n 200`; both are steady-state tails and the difference is immaterial (<0.1% on
  the kernels stable enough to measure it).

### Archived traces (`/data/`)

| directory | what |
|---|---|
| `fav4_opt8_SGB_mergepack_memnop3_...` | fav4 SOQ=1, `MEMNOP=3` |
| `fav4_opt8_SCALEONQ_false_SGB_memnop3_...` | fav4 SOQ=0 before the `fneg` fix (9/16 windows) |
| `fav4_opt8_SCALEONQ_false_fnegfix_SGB_memnop3_...` | fav4 SOQ=0 after the fix (16/16) |
| `fav3_overcapSGB_nopackedscalarize_memnop3_...` | fav3, chunk layout |
| `fav3_overcapSGB_mixedclasswindows_memnop3_...` | fav3, mixed-class windows |


## Why ATT captured nothing on GPU[0] -- a harvested CU, not a broken die

Symptom: `rocprofv3 --att` on GPU[0] exited 0 and wrote a `ui_output_*` directory holding
only `code/filenames/occupancy/realtime.json`, with a ~35 KB `.att` instead of ~65 MB and
no wave files, so `process_json.py` failed with `'NoneType' object is not iterable`.
**It is not a broken die and not specific to GPU[0]** -- the shipped `att_target_cu: 0`
simply named a CU that does not exist on that die's shader array.

Triage, using a fast probe (`--seqlen 1088`, ~40 s per capture, pass/fail = wave files
present):

1. **Only GPU[0] failed**, all 7 other dies captured -- so not a tool or driver problem.
2. On GPU[0], **CU0 failed on SE0 and SE3 but captured fine on SE1 and SE2**, and
   `att_target_cu` 1, 2, 4 all captured on SE0. So the failure is per-(SE, CU) slot, not
   per-die. rocprofv3's logs were clean -- timestamps only, no warning.
3. KFD topology explains the mechanism: every die reports `array_count=32` with
   **`cu_per_simd_array=9`** but only 256 CUs enabled, i.e. **one CU per array is harvested**
   and *which* one is a per-die yield artifact.
4. Direct proof: a HIP kernel reading `HW_REG_HW_ID` (CU[11:8], SE[15:13]) and
   `HW_REG_XCC_ID` per workgroup, histogrammed into a per-(XCC, SE) census of CUs that
   actually execute waves. Every array shows exactly 8 of 9 slots, and the missing slot
   lines up with the ATT failures:

   ```
   GPU[0] XCC0:  SE0 = 1..8 (CU0 GONE)   SE1 = 0-5,7,8   SE2 = 0-7   SE3 = 1..8 (CU0 GONE)
   GPU[4] XCC0:  SE0 = 0-7               SE1 = 1..8      SE2 = 0-7   SE3 = 1..8
   ```

   SQTT armed on a CU that cannot run waves records only the SE-level occupancy/realtime
   streams -- hence a small `.att` with no instruction data. rocprofv3 does not validate
   the requested CU against the harvest mask, so it reports success.

**Fix, static:** `att_target_cu: 4`, which the `att_attn*.json` configs here now use. A
census across all 8 dies gives the union of harvested indices as **{0, 1, 2, 3, 6, 7, 8}**,
so only **CU 4 and 5 are never harvested anywhere** on this box. (CU2 works on GPU[0] and
GPU[4] but GPU[3] harvests it -- 4 or 5, not "some low number".) Verified capturing on
GPU[0], GPU[3] and GPU[4].

**Fix, portable:** there is *no* "use whatever CU is enabled" option in rocprofv3 --
`att_target_cu` is a single index, `-1` aborts (exit 134), an out-of-range 15 is accepted
and silently yields an empty trace, and the CLI default is 1 (also harvested somewhere on
this box). So discover it: `scripts/att_pick_cu.py` runs `scripts/att_cu_census.cpp` (every
workgroup reports its (XCC, SE, CU) via `HW_REG_HW_ID` + `HW_REG_XCC_ID`; slots that never
appear are harvested), intersects the enabled sets over **every array the SE mask selects**,
and rewrites a template config with a valid index. Cached per die by PCI bus id, so it costs
one sub-second launch per machine, and it honours `HIP_VISIBLE_DEVICES`.

```bash
python scripts/att_pick_cu.py --template kernels/attention/att_attn_se0.json --out /tmp/att.json
rocprofv3 --att -i /tmp/att.json -d /tmp/trace -- python bench.py --rocprof --seqlen 16320 --n-iters 20

python scripts/att_pick_cu.py --se-mask 0x1 --print-cu   # just the index
python scripts/att_pick_cu.py --report                   # per-array census
```

It picks per die and per mask rather than assuming: GPU[0] -> CU1 for `se_mask 0x1` but CU2
for `0xF`, GPU[3] -> CU1, GPU[4] -> CU0. All verified capturing. Note the intersection must
be over every *selected* array, not one: on GPU[0] the SE0 arrays harvest CU0 in XCC0,
CU7 in XCC1 and CU8 in XCC2-7, so only the intersection is safe.

**Bonus: this also re-validated the cross-die metric mixing on the fast die itself.** With
`att_target_cu: 4`... actually with CU2, the full-shape trace of fav4 SOQ=1 on **GPU[0]**
gives **4303.8 cyc/iter, 47.59%/wave** against **4299.95 / 47.63%** on GPU[4] -- **0.09%**.
So the tables' TFLOPS-from-GPU[0] + cycles-from-GPU[4] mix is sound, and traces can now be
taken on whichever die is free.

## MFMA operand reuse — the SP XDL srcA/srcB read-suppression feature (verified, not adopted)

gfx950 SP tracks 8 VGPR addresses per operand side tied to its XDL buffer; a matching,
non-dirty entry suppresses the VGPR read. With fp16 4-VGPR operands that is **2 entries per
side**. The feature is a **CAC (dynamic power)** optimization, not a throughput one, and it is
controlled entirely by the order in which MFMAs are emitted.

Sharing an operand and sharing an accumulator are **mutually exclusive**: for
`D[m,n] += Σ_k A[m,k]·B[k,n]`, holding an operand fixed forces the output tile to change, and
holding the output tile fixed (k-inner) forces both operands to change. So a shape can exploit
one or the other, never both. For **16x16** the accumulator side wins — its 4-VGPR Matrix C/D
can be kept in the XDL accumulator flops and have both the read *and* the write killed, which
is worth ~2x the operand side, and k-inner already produces it for free. For **32x32** that
path does not exist (C/D is 16 VGPRs, too wide), so the operand side is all that is available.

MFMA emission order lives in Triton, not in Gluon: the `b/m/n/k` rep nest in
`third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp`. It is k-innermost. Walking
`n` innermost instead holds `operandA[{b,m,k}]` fixed across the `n` run. Measured with a
temporary `AMDGCN_MFMA_N_INNER` gate there (default branch byte-identical), GPU[0], seqlen
16320 fp16, each configuration at its own `LLIRSCHED_WP_MEMNOP` optimum:

| | suppressed | cyc/iter | mfma eff /SIMD | sclk | power | kernel time |
|---|---|---|---|---|---|---|
| **fav4** k-inner, MEMNOP=2 | 0/128 (0%) | 4298.4 | 95.3% | 1435.7 MHz | 1399.7 W | 1243.0 TF |
| **fav4** n-inner, MEMNOP=3 | 40/128 (31%) | 4285.4 (−0.30%) | 95.6% | 1446.7 MHz (+0.77%) | 1399.7 W | 1246.1 TF (+0.25%) |
| **fav3** k-inner, MEMNOP=2 | 0/64 (0%) | 4806.0 | 85.2% | 1485.6 MHz | 1396.3 W | 1168.5 TF |
| **fav3** n-inner, MEMNOP=2 | 20/64 (31%) | 4797.1 (−0.19%) | 85.4% | 1496.3 MHz (+0.72%) | 1394.8 W | ~1171.8 TF |

**Conclusion: the effect is real and reproducible but limited, so the tree keeps k-inner.**
Power is pinned at the cap in all four runs, so the saving appears as **clock, not watts** —
+0.77% and +0.72% on two kernels with very different schedules, spill behaviour and starting
suppression. Throughput follows at ~+0.25%, i.e. sub-1%, which does not justify a lowering
change that also costs GEMM `a16w16` v9 1.6% (1416.9 -> 1394.9 TF, VGPRs 498 -> 508, a VGPR
pressure effect that no retune recovers). 31% is the *optimum* for these kernels, not a
partial result: srcA is all-distinct in every stage, so only srcB is reusable (62.5% of srcB
reads, weighted over QK's 2-tile and PV's 4-tile runs) without the 2x2 register blocking that
GFXIPARCH-1379's own example uses. Both kernels land on exactly the same 0% -> 31%.

Details worth keeping:

- **`mfma efficiency` from `process_json.py` is derived from the cycle count** (mfma count x
  32 / loop cycles). It is the reciprocal of cycles and is *not* independent evidence about
  interleave quality — do not read it as one.
- **Extra stall cycles raise the clock**, because idle waves burn no power. An early version of
  this comparison ran n-inner on k-inner's MEMNOP and read the resulting +1.9% clock as a power
  win; it was ~100 cycles/iter of barrier wait. Never compare clocks across builds with
  different cycle counts.
- **The reorder does not perturb the schedule.** fav4 asm inventory is identical (573 instrs,
  same classes per stage, same `s_nop`/`s_waitcnt` counts), the QK class interleave is
  byte-identical, and VGPRs go *down* 256 -> 248 with no spills. The llir scheduler counts
  classes and is indifferent to register identity, exactly as expected.
- **Mem-stage pacing is order-sensitive on fav4 but not fav3.** fav4's MEMNOP optimum flips
  2 -> 3 (n-inner: 4447.5 / 4313.4 / 4400.9 / **4285.4** / 4295.5 for MEMNOP 0-4; k-inner is
  4298.4 at 2 and 4394.4 at 3). fav3 stays at 2 for both. Per-stage attribution from raw ATT
  records localized this — and it must **include the `s_barrier` records**, which hold
  ~440-540 cyc/iter that instruction-level attribution drops entirely.
- **Count suppression from the ATT dynamic stream, not from the asm.** Deriving the loop body
  from the `This Inner Loop Header` comment plus a `.LBB0_<n-1>` latch guess over-collects on
  fav3 (144 mfma instead of the true 64 that `process_json`'s back-edge detection finds), which
  first produced a bogus "fav3 starts at 12%". Walking a wave's executed instructions has no
  loop-structure heuristic to get wrong.
- **FlyDSL already emits the n-inner order**, and gets 62.5% of srcB / 31% overall for free;
  gluon k-inner gets 0%. Its accumulator adjacency costs it nothing, since the 16x16-only C/D
  kill was never available at 32x32. Both implementations leave srcA entirely unexploited.

## Unifying the packed/scalar flow: declare group sizes in instructions

`sched_group_barrier`'s size operand is a count of **instructions**. The fitting path was pushing
one 4-cycle entry per **element**, so a packed op contributed two entries and a 24-cycle window
declared six VALU where only three instructions existed. IGroupLP cannot fill such a group, and an
unfillable group makes its pipeline solver abandon the region -- the same failure the miscounted
`fneg` produced. Pushing one entry per instruction priced by its weight is what the TRANS branch
beside it always did (one entry of 8), and what `declareRegionGroupsOverCap` always did for its
Chunks. So this is the fitting path being brought into line with the path next to it, not a new
mechanism.

The point is that **neither kernel needs `AMDGCN_SCALARIZE_PACKED_FOPS` now.** Both start from
packed math, the declaration is satisfiable either way, and `SIPreEmitPeephole` splits whatever
lands in an MFMA shadow. fav3 never set the variable -- its over-capacity path already reasoned in
instructions -- and fav4 no longer has to, so there is no longer an environment variable that has
to agree with which kernel is being compiled.

Measured on GPU[0], seqlen 16320 fp16, both correct against torch SDPA:

| | cyc/iter | mfma eff /SIMD | ceiling | TFLOPS | spills |
|---|---:|---:|---:|---:|---:|
| fav3 (byte-identical asm to before) | 4803.5 | 85.3% | 91.4% | 1169 | 9 |
| fav4 | 4373.2 | 93.7% | 100% | 1241 | 2 |
| fav4 SCALE_ON_Q=0 | 4438.0 | 92.3% | 100% | 1220 | -- |

fav4 gives up **1.6% of cycles and 0.3% of throughput** against the old force-everything-scalar
build (4304.6 / 95.2% / 1245). Its loop still reaches the 100% ceiling with no packed op left in
it, so the loss is not coverage but **fill accuracy**: a count is a faithful cycle budget only when
the class is uniform-cost, and a VALU group holding both 4-cycle scalars and 8-cycle packed ops can
be satisfied by a cheap triple or an expensive one. TRANS never has this problem -- every
transcendental costs 8.

Two alternatives were measured and rejected:

- **Leave the declaration in elements and rely on the peephole.** 4719.6 cyc, 86.8%, 16 packed ops
  stranded outside the shadow, ceiling 97%. Costs 9.8% because most of the region's pipeline is
  abandoned, not because of the stranded ops.
- **Scalarize inside the plugin**, after the over-capacity hand-off, so the covered ops are split
  before the declaration is emitted. 4412.0 cyc, 92.8% -- worse than the one-line change and
  thirty lines instead of one. Note the hand-off ordering is essential either way: the dispatch to
  `declareRegionGroupsOverCap` happens *inside* `declareRegionGroups` after run collection, so
  anything done at the top of that function also hits fav3 and destroys its packed remainder
  (measured: ceiling 91.4% -> 85.6%, 0 packed left).

fav4 also picks up 2 VGPR spills (`private_segment_fixed_size` 0 -> 12), all outside the loop:
without the global Triton pass the prologue and drain keep their packed ops, which shifts pressure
there. No scratch traffic in the loop body.

**Measurement note.** fav3's asm is byte-identical, which makes it a free control for thermal
drift, and it earned its keep here: a mid-session batch read fav3 at 1136.7 against its usual
1168.5, i.e. the whole batch was ~2.7% low. After a 200 s cool-down fav3 read 1169.1 and 1169.6
either side of fav4's 1241.5. Bracket any TFLOPS measurement with an unchanged binary.

## mem-stage pacing x scale-on-Q matrix (2026-07-27, GPU[0], unified plugin)

Three settings, both kernels, seqlen 16320 fp16. `LLIRSCHED_WP_MEMNOP` controls the `s_nop`s at
each mem-cluster head. fav3 had no `SCALE_ON_Q` when this was first run; it has one now (see
below), so all three settings exist for both kernels.

| setting | | fav3 | fav4 |
|---|---|---|---|
| 1 | SOQ=0, no `s_nop` (`MEMNOP=0`) | 1165.0 TF / 4882.1 cyc / 83.9% | 1222.8 TF / 4577.2 cyc / 89.5% |
| 2 | SOQ=0, `MEMNOP=2` | 1168.8 TF / 4802.0 cyc / 85.3% | 1233.1 TF / 4434.5 cyc / 92.4% |
| 3 | SOQ=1, `MEMNOP=2` | **1176.6** TF / 4752.2 cyc / 86.2% | **1242.6** TF / 4368.7 cyc / 93.8% |

What the pacing is worth on its own (setting 1 -> 2): fav3 **-1.6% cycles**, +1.4 pts, +0.3% TF;
fav4 **-3.1% cycles**, +2.9 pts, +0.8% TF. It helps fav4 about twice as much as fav3, which fits
the mechanism -- fav3's dot clusters are over capacity, so its mem clusters are less often the
thing the other wave is waiting on.

Scale-on-Q on top (2 -> 3): fav4 **-1.5% cycles**, +1.4 pts, +0.8% TF; fav3 **-1.0% cycles**,
+0.9 pts, +0.8% TF. Together the two are worth **-4.6% cycles / +1.6% TF** on fav4 and
**-2.7% cycles / +1.0% TF** on fav3. Both kernels gain about the same 0.8% from scale-on-Q, which
is what you would expect -- it removes the same per-element multiply from the same cluster in both.

**`MEMNOP=2` is the optimum in all three configurations**, which is the current default. Full
sweep, cyc/iter:

| `MEMNOP` | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| fav3 SOQ=0 | 4882.1 | 4828.1 | **4802.0** | 4818.5 | 4814.6 |
| fav3 SOQ=1 | 4823.8 | 4783.5 | **4752.2** | 4761.6 | 4765.7 |
| fav4 SOQ=0 | 4577.2 | 4538.2 | **4434.5** | 4523.1 | 4444.0 |
| fav4 SOQ=1 | 4422.9 | 4467.1 | **4368.7** | 4524.9 | 4483.8 |

Note the sweep is not monotonic and not smooth -- 4 beats 3 in two of the three rows. Pacing is a
phase relationship between two waves' LDS bursts, not a quantity, so bisecting it does not work;
sweep the small range.

TFLOPS were bracketed by an unchanged fav3 `MEMNOP=2` build before and after, reading 1168.8 and
1168.9, so the box was stable across the batch (see the drift incident recorded above).

### `SCALE_ON_Q` added to fav3

Mirrors fav4: `qk_scale` is folded into `Q` once before the loop, so VEC1's row max needs no scale
multiply and its `fma(qk, qk_scale, -m_new)` collapses to a plain subtract. `qk_scale`'s definition
had to be hoisted above the `Q` load, which is where it now sits. Default `True`, matching fav4 --
**so fav3's default build changes**, from 4802.0 cyc / 85.3% to 4752.2 / 86.2%.

- `SCALE_ON_Q=False` compiles to a **byte-identical opcode stream** to fav3 before the change, so
  the branch costs nothing when it is off.
- Accuracy follows fav4's tradeoff exactly: max error 3.05e-05 -> 1.22e-04 from the extra fp16
  rounding of Q, both far inside the 1e-3 tolerance.
- Spills improve slightly, 9 -> 8.
- The **ceiling does not move**: 192 exposed cycles either way, so 91.4%. SOQ=1 shifts 8 cycles of
  QK demand into a slightly different shape (PV 384 -> 392 demand, 12 packed -> 10) without
  changing what is left outside the shadow. The gain is fill quality, not coverage.
- TFLOPS measured interleaved with an unchanged SOQ=0 control: control 1166.5 / 1168.1,
  SOQ=1 1177.4 / 1175.7. An earlier batch had to be discarded because its two controls disagreed
  by 1.3%.

## Final comparison at S=16384 (2026-07-28, GPU[0], interleaved)

Five configurations, run one after another three times round so drift hits every row equally.
TFLOPS is the mean of the three; efficiency is the in-loop per-SIMD ATT figure from a final run.

| | run 1 | run 2 | run 3 | mean TF | mfma eff /SIMD |
|---|---:|---:|---:|---:|---:|
| gluon `fav4` tuned | 1228.1 | 1243.0 | 1236.8 | **1236.0** | **93.8%** |
| ROCm/FlyDSL `63eb891` | 1178.3 | 1168.5 | 1202.2 | **1183.0** | 86.4% |
| gluon `fav3` tuned | 1179.4 | 1171.1 | 1146.3 | **1165.6** | 86.1% |
| gluon `fav4` stock LLVM | 1114.3 | 1121.3 | 1107.3 | **1114.3** | 68.0% |
| gluon `fav3` stock LLVM | 1090.3 | 1094.1 | 1089.2 | **1091.2** | 67.9% |

Tuned = llirSched plugin + `DISABLE_LLVM_OPT=disable-machine-sink` + `SCALE_ON_Q=1` +
`MEMNOP=2`. Stock = no plugin, no env vars at all. The plugin stack is worth **+10.9%** on fav4
and **+6.8%** on fav3; in efficiency, **+25.8** and **+18.2 points**.

**Stock LLVM cannot tell fav3 and fav4 apart** -- 68.0% vs 67.9%, and only 2.1% of throughput.
The 7.7-point gap between the tuned rows is the scheduler exploiting the budget headroom lazy
rescaling creates, not the algorithm on its own.

Caveats, both against these numbers rather than for them: S=16384 gives an even `n_blocks`, so
both Gluon kernels run the `ODD_TAIL` tile (~0.5 pt of epilogue) that 16320/16448 avoid; and the
variance of the tuned rows is larger than the stock ones (spreads 15-34 vs 5-14), since a denser
MFMA stream sits nearer the power cap. fav4's 53 TF lead over FlyDSL exceeds any spread here;
fav3-vs-FlyDSL (17 TF) does not and should be read as a tie.

### FlyDSL is measurement-sensitive; use a deep window

Its own harness averages a shallower window, which leaves the kernel in the thermal transient.
Six consecutive runs of one config, 20 s apart:

    1236.9   1242.8   1165.9   1167.7   1159.3   1157.5

A monotonic 7% decay, and *not* the box: an unchanged gluon fav3 build interleaved with these read
1176.7 twice. Re-running FlyDSL with the same 1000-dispatch / last-100 window the Gluon rows use
gives 1159.0, 1167.2, 1191.4, 1201.6. **Earlier revisions of the README quoted 1242 for FlyDSL,
which was a cool-die reading.** `scripts/fly_kernel_time.py` now applies our window to it.

The protocol asymmetry runs in FlyDSL's favour, not ours: its Python launcher spaces dispatches
further apart than our prepared launch, so its die runs cooler, and our numbers are taken in the
more saturated regime.

### FlyDSL config sweep (non-causal, S=16320)

Its shipped defaults are its optimum. Relative to canonical:

| knob | effect |
|---|---|
| `dualwave_swp_enable_stagger=False` | 1001.8 TF, **-15%** |
| `dualwave_swp_setprio=False` | 1145.4 TF, -3% |
| `waves_per_eu=1` | indistinguishable (interleaved pairs: 1242.4 vs 1242.7, then 1180.7 vs 1168.1) |
| `dualwave_swp_lazy_rescale=False` | ~1044 TF, the fav3-equivalent path |

Causal vs non-causal: **non-causal is its better number** -- 1178.0 TF against 1037 causal with
the FLOPs halved for the skipped tiles, and causal's max error is 1.26e-03 against 5.4e-05. The
builder defaults to `causal=True`; every number here forces `causal=False` to match our kernels.

## Why B=32 S=8192 beats B=1 S=16384, and what that says about the comparison

Both shapes do **exactly the same work** -- 8.796 TFLOP per dispatch -- and both divide evenly
into the machine (4096 and 8192 workgroups against 256 resident), so the 11% throughput
difference is not a FLOP-count or tail-effect artifact.

**The loop body is identical.** In-loop MFMA efficiency and cyc/iter barely move with the shape,
which they cannot: the loop does not depend on B or H.

| bf16 | eff /SIMD | loop | pro | epi | cyc/iter |
|---|---:|---:|---:|---:|---:|
| fav4 B=1 S=16384 | 93.9% | 94.07% | 2.58% | 3.35% | 4361.5 |
| fav4 B=32 S=8192 | 94.5% | 88.31% | 5.47% | 6.22% | 4334.6 |
| fav3 B=1 S=16384 | 86.0% | 95.14% | 1.68% | 3.18% | 4761.7 |
| fav3 B=32 S=8192 | 86.1% | 90.52% | 3.42% | 6.06% | 4757.1 |

**Per cycle the B=32 shape is strictly worse.** Each workgroup walks 128 K/V blocks instead of
256, so the fixed prologue and drain amortize over half as much work and their share **doubles**
(5.9% -> 11.7%). Counting whole dispatches it needs about **4% more cycles**.

**It wins on clock, because it is not power-limited.**

| bf16 | TFLOPS | sclk | power |
|---|---:|---:|---:|
| fav4 B=1 S=16384 | 1296.5 | 1527.7 MHz | **1396.5 W -- at the cap** |
| fav4 B=32 S=8192 | 1325.6 | **1569.8 MHz** | 1317.6 W |
| FlyDSL B=1 S=16384 | 1186.3 | 1541.1 MHz | 1366.1 W |
| FlyDSL B=32 S=8192 | 1327.1 | **1650.9 MHz** | 1339.4 W |

The chain: shorter loops -> double the prologue/drain -> a **less MFMA-dense** instruction stream
-> lower power -> off the ~1400 W cap -> the governor grants more clock, and the clock is worth
more than the cycles it cost. The "better" config is better partly *because* it wastes more time
on pipeline fill and drain.

**This is why the ranking flips between shapes.** The two kernels have opposite strengths:

- fav4 is **cycle**-efficient -- 94.5% in-loop against FlyDSL's 84.7%.
- FlyDSL is **power**-efficient per cycle -- at B=32 it clocks 5.2% higher at comparable power,
  and it also spends more of its time in the loop (94.13% vs 88.33%), so its pipeline fill/drain
  is cheaper than our four-stage one.

At B=1 S=16384 the board is pinned at the cap, cycle efficiency decides, and fav4 wins by 9.3%.
At B=32 S=8192 there is power headroom and the two cancel almost exactly:

    fav4 cycle advantage   (0.945 x 0.8833) / (0.847 x 0.9413) = +4.8%
    FlyDSL clock advantage  1650.9 / 1569.8                    = +5.2%

which is a dead heat, and the measurement agrees (1318.0 vs 1320.1). **Neither shape gives "the"
answer** -- B=1/S=16384 flatters us, B=32/S=8192 flatters them. The one quantity that is stable
across both shapes and both dtypes is the in-loop MFMA efficiency, which is the thing the
scheduler actually controls.

### The B=32 matrix (bf16, GPU[0], interleaved, 3 rounds)

FlyDSL's published config: B=32, S=8192, H=8, D=128, bf16, non-causal. Their post claims 1320
TFLOPS; **reproduced at 1319.4 / 1319.6 / 1320.2** with `scripts/fly_kernel_time.py`.

| | run 1 | run 2 | run 3 | mean TF | eff /SIMD |
|---|---:|---:|---:|---:|---:|
| ROCm/FlyDSL `63eb891` | 1318.7 | 1320.8 | 1320.8 | **1320.1** | 84.7% |
| gluon `fav4` tuned | 1317.1 | 1318.8 | 1318.2 | **1318.0** | **94.5%** |
| gluon `fav3` tuned | 1242.1 | 1242.6 | 1243.1 | **1242.6** | 86.2% |
| gluon `fav4` stock LLVM | 1197.6 | 1196.4 | 1198.7 | **1197.6** | 68.5% |
| gluon `fav3` stock LLVM | 1141.0 | 1140.7 | 1141.1 | **1140.9** | 67.8% |

This shape is a far better benchmark than B=1/S=16320: every row is stable to ~2 TFLOPS, where
B=1 gave FlyDSL a 34-point spread and a 7% warm-up decay. B=32 puts 8192 workgroups and a 2.1 GB
working set on the die, so it reaches a uniform steady state and holds it. **Prefer it for
headline numbers.**

The plugin stack is worth **+10.0%** on fav4 (1197.6 -> 1318.0) and **+8.9%** on fav3
(1140.9 -> 1242.6) here, against +10.9% / +6.8% at B=1 S=16384 fp16.
