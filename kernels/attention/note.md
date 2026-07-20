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
