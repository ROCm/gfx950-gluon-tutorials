# Optimal scheduling of MFMA + math under gfx950 co-issue rules

This note formalizes and solves the instruction-scheduling problem for a region
that contains matrix (MFMA) instructions and elementwise FP32 math (as in the
DOT1 stage of the FAv3 attention kernel), under the gfx950 (CDNA4) co-execution
rules, and proves the optimal schedule.

## Machine model

A wave issues onto two execution resources — the **MFMA pipe** and the **VALU
pipe** — that can run in parallel (co-execution) subject to constraints.

Constants (cycles):

| symbol | value | meaning |
|---|---:|---|
| `c_m` | 32 | MFMA occupancy (`v_mfma_f32_32x32x16`) |
| `r`   | 8  | MFMA input-read phase — **no** co-issue during `[0, r)` |
| `w = c_m − r` | 24 | co-issue window `[r, c_m)` of one MFMA |
| `c_u` | 4 | unpacked VALU op (`v_add/v_mul_f32`); **co-issuable** with MFMA |
| `c_p` | 4 | packed VALU op (`v_pk_*_f32`), does **2** units; **not** co-issuable |
| `h`   | 12 | hazard: a packed op cannot issue for `h` cycles after an MFMA finishes |

Rules:
1. MFMAs cannot co-execute with each other → the MFMA pipe is serial; `M` MFMAs
   take `c_m·M = 32M` cycles, and back-to-back issue is achievable.
2. During an MFMA's `[0, r)` read phase nothing co-issues; during its `[r, c_m)`
   window the VALU may co-issue **unpacked** ops (free — hidden under the MFMA).
3. Packed ops never co-issue with an MFMA, and additionally cannot issue for `h`
   cycles after the last MFMA of a run finishes (the `44 = 32 + 12` hazard).

**Co-issue capacity** of one MFMA `= w / c_u = 24 / 4 = 6` unpacked ops.

## Problem

**Input:** `M` (number of MFMAs), `N` (number of unpacked math *units* of work;
two units may be merged into one packed instruction).
**Output:** an ordering of MFMAs / unpacked / packed ops minimizing the makespan
`T`, obeying the rules above.

## Solution

Let `k = max(0, N − 6M)` be the number of units that cannot be hidden. Define the
exposed-tail cost
```
g(0) = 0
g(k) = 4k                         for 1 ≤ k ≤ 3
g(k) = 4k − 4·floor((k−3)/2)       for k ≥ 3     (= 2k+6 if k odd, 2k+8 if k even)
```
Then the optimum is
```
T*(M, N) = 32·M + g( max(0, N − 6M) )
```
i.e.
- **`N ≤ 6M`  →  `T* = 32M`**  (MFMA-bound; all math hides for free)
- **`N > 6M`  →  `T* = 32M + g(N−6M) ≈ 20M + 2N`**  (+ a small parity constant)

**Optimal schedule.** Interleave `MFMA ‖ 6 unpacked` for each of the `M` MFMAs
(fill every co-issue window), then a tail of `3 unpacked` (to cover the hazard)
followed by packing all remaining units:
```
(MFMA, 6×unpacked) × M ,  then  3×unpacked ,  then  packed for the rest
```

## Proof (matching lower bound ⇒ optimality)

**(1) MFMA pipe.** MFMAs are mutually exclusive on the matrix pipe, so it is busy
for `32M` cycles and the last MFMA finishes no earlier than `32M`. Hence
`T ≥ 32M`.

**(2) Hiding capacity.** A hidden op must be unpacked and must sit in some MFMA's
`[r, c_m)` window; that window holds `w/c_u = 6` unpacked ops, and packed ops hide
none. So at most `6M` units hide, and at least `k = N − 6M` units are *exposed*
(run with no MFMA overlap).

**(3) Exposed ops run after `32M`.** During the MFMA pipe the only VALU-usable
time is the co-issue windows — saturated by the `6M` hidden ops when `N > 6M` —
and the read windows `[0, r)`, which admit neither co-issued unpacked (no
co-issue) nor packed (an MFMA is executing). So every exposed op starts at
`≥ 32M`.

**(4) Exposed cost is `g(k)`.** On the VALU, `k` exposed units cost:
- without any packed: `4k` (unpacked at `c_u = 4`/unit);
- with packed: the hazard forbids packed in the first `h = 12` cycles, which hold
  at most `3` unpacked units; the remaining `k−3` units run at best at `c_p/2 = 2`
  cyc/unit, for `12 + 2(k−3)`.

Minimizing over the two gives `g(k) = min(4k, 2k+6)` up to the integer-packing
parity term `4k − 4·floor((k−3)/2)`. Filling the hazard window with `3` unpacked
(rather than idling or paying packed's stall) is what makes the tail `2k+6`
instead of `2k+12` — a fixed 6-cycle win.

**(5) Combine.** `T ≥ 32M + g(k)`. The schedule above attains this bound, so it is
optimal. ∎

## Consequences for DOT1

For the FAv3 DOT1 stage, `M = 16` (QK MFMAs) and `N ≈ 97` math units, so
`6M = 96 ≥ N` almost exactly: **the optimum is essentially MFMA-bound, `T* ≈ 32·16`,
with the entire softmax rescale hidden.** Any packed op left un-hidden in the real
schedule is near-pure loss. Crucially, that loss is an **ordering** problem (the
`6M` co-issue slots are not being filled because the scheduler clustered the
MFMAs), *not* a packed-vs-unpacked rewrite problem — the packed→unpacked rewrite
(LLVM `SIPreEmitPeephole`) runs after scheduling and cannot create the interleave
the optimum requires.

## Path A experiment — driving the schedule with `SCHED_GROUP_BARRIER`

We validated the theory by making the compiler emit the optimal interleave, on
the *pinned* upstream LLVM (no LLVM rebuild — only a Triton pass change).

**Mechanism.** `ConvertWarpPipeline` (`emitDotCoIssueGroups`, env-gated by
`TRITON_HIP_DOT_COISSUE=1`) appends an over-provisioned sequence of
`rocdl.sched.group.barrier` ops to the **end** of each DOT stage cluster:

```
[MFMA 1][VALU K] × N            (mask MFMA=0x8, VALU=0x2; N≈24 over-provisioned)
```

Placement matters: `AMDGPUIGroupLP::initSchedGroupBarrierPipelineStage` builds
each group by scanning **upward** from the barrier, so the whole sequence must
sit *after* every real instruction in the region (a top-of-region placement
yields empty groups and is a no-op — we hit exactly that first). The AMDGPU
IGroupLP machine-scheduler mutation then permutes the region to match the group
order. Because this runs *before* `tt.dot`→MFMA lowering, the group count is
over-provisioned (empty tail groups are dropped); because packing
(`SIPreEmitPeephole`) runs *after* scheduling, the VALU groups fill with
**unpacked** ops — exactly the ops we want hidden.

**Best `K` is 3, not 6.** Sweeping `K` (VALU ops demanded per MFMA group):

| K | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| TFLOPS (1×16320) | 1025 | **1049** | 1028 | 1037 | 1039 | 1034 |

`K=3` wins even though the co-issue *capacity* is 6. Demanding the full 6 per
group over-constrains the solver: several VALU ops (the `v_cvt_pk_f16` of the
MFMA result, and dependent rescales) *depend on that MFMA* and cannot sit in its
own window, so a hard `[VALU 6]` group forces a stall. `K=3` anchors only the
genuinely-independent prev-iteration softmax work and lets the greedy solver
hide ~6 while cleanly deferring all packed ops to the tail.

**The realized schedule is the proof's optimum.** DOT1 instruction stream
(`M`=mfma, `s`=unpacked math, `p`=packed, `c`=cvt):

```
baseline : (M ssssss pppp)×5 … M ccccc M M M M M M .   ← packed exposed between
                                                          MFMAs; 6-MFMA tail cluster
K=3      : (M ssssss)×16  then  ppp s cccc…(16)         ← every co-issue window
                                                          full; ALL packed deferred
```

i.e. `(MFMA ‖ 6 unpacked) × 16, then packed rest` — precisely
`(MFMA,6×unpacked)×M , 3×unpacked , packed` from §Solution. Exposed-packed
cycles per DOT1 drop from ≈80 (baseline) to ≈12 (K=3).

**Measured (1×16320 non-causal bshd fp16, gfx950, ATT + `process_json.py`):**

| metric | baseline | K=3 | Δ |
|---|---|---|---|
| MFMA efficiency /wave | 32.60% | **35.51%** | +2.9 pp (+8.9% rel) |
| loop iteration duration | 6282 cyc | **5768 cyc** | −8.2% |
| end-to-end TFLOPS | ~1019 | **~1049** | +2.9% |
| correctness | ✅ | ✅ | — |

The loop *compute* speeds up 8.2%; end-to-end dilutes to +2.9% because the loop
is partly bound on the async K/V HBM loads (the mem clusters). Consistent with
the proof: DOT1 is MFMA-bound, so the win is not a shorter MFMA pipe but the
removal of the **exposed-packed** cycles that had been padding the region past
`32M`. Variance also collapses (baseline 1002–1022 → K=3 1047–1050).

**Next (Path B).** Repackage this as a proper `IGLPStrategyID` in
`AMDGPUIGroupLP` (an FAv3-shaped strategy that emits the `[MFMA 1][VALU 3]`
groups itself, gated on the DOT region shape) so it needs no per-kernel Triton
env knob and no over-provisioning — the current Path A is the validating cut.
