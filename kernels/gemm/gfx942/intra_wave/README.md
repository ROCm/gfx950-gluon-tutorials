# intra_wave — 4-wave a16w16 GEMM for gfx942 / MI300X

Port of the gfx950 tutorial's [`intra_wave/a16w16/v9_beyond_hotloop`](../../intra_wave/a16w16/v9_beyond_hotloop):
4 warps/CTA = **1 wave/SIMD**, a 256×256 output tile sliced into a 2×2 grid of
128×128 quadrants, and XCD-aware PID remapping with `GROUP_SIZE_M=4`.

> **Read [`../README.md`](../README.md) first.** It has all the performance
> numbers (§3) and the CDNA3 constraints this design is a response to (§4) — no
> async copy, 64 KB of LDS, the 16-cycle MFMA, and the shared
> `SwizzledSharedLayout(8, 1, 8)`. This file assumes them and covers only this
> kernel's schedule.

**Start with this kernel.** At 1 wave/SIMD there is nothing to hide behind: the
memory ops and the MFMAs share one instruction stream, so every cost §4 describes
lands directly in the schedule. [`../inter_wave`](../inter_wave/) is the answer to
the problems that are visible here.

```bash
cd ..                                              # kernels/gemm/gfx942
python bench.py -k intra_wave                      # correctness + TFLOPS
python tools/bench_prepared.py --kernel intra_wave --gpus all   # sustained timing
python tools/lds_conflict.py  --kernel intra_wave              # LDS bank conflicts
python tools/run_att.py intra_wave --K 8256 --out att_intra     # ATT trace
```

Two things are **required** for this kernel, unlike
[`../inter_wave`](../inter_wave/):

* `TRITON_FORCE_MFMA_AGPR=1` (set by `bench.py`), which supplies the
  `amdgpu-agpr-alloc=256` hint. Without it the 256 f32 accumulators compete with
  the 192 VGPRs of operands and staging, and the kernel spills.
* the LLIR scheduler plugin, which is what interleaves the memory ops through the
  MFMA stream (see [`note.md`](note.md) §1–2):

```bash
LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 python bench.py -k intra_wave
```

Codegen: **256 arch VGPR + 256 AGPR, 0 spills, 1 wave/SIMD** (LDS-bound),
65536 B LDS, and `SQ_LDS_BANK_CONFLICT` measures exactly **0**.

## Design

### Half-tile LDS recycling

64 KB of LDS is *exactly one* 256×256×64 stage, so double buffering is
impossible. Rather than shrink `BLOCK_K` to 32 — which would halve the contiguous
run of every global load from 128 B to 64 B and roughly double TCP cache-line
pressure — this kernel keeps `BLOCK_K = 64` and recycles LDS at **half-tile
granularity**. Each of the four half-tiles owns a 16 KB slot, refilled one region
after its last read:

| region | DOT | LR (next region's operand) | LW | GR |
|---|---|---|---|---|
| 0 | `C_tl` | `A_bot(k)` | `B_left(k+1)` | `B_left(k+2)` |
| 1 | `C_bl` | `B_right(k)` | `A_top(k+1)` | `A_top(k+2)` |
| 2 | `C_tr` | `B_left(k+1)` | `A_bot(k+1)` | `A_bot(k+2)` |
| 3 | `C_br` | `A_top(k+1)` | `B_right(k+1)` | `B_right(k+2)` |

Every slot's write lands strictly between its previous read and its next read,
one region apart on both sides, so one barrier per region boundary is sufficient
and correct. One region is 64 mfma ≈ 1024 cycles, ample to absorb the barrier
plus the LDS traffic.

The pipeline depth this buys equals gfx950's double buffer: `GR(k+2) → LW(k+1)`
is a full K-step (~4096 cycles) of global-latency hiding, and `LR(k+1) → DOT(k+1)`
is one region (~1024 cycles) of LDS-latency hiding.

**Why not `BLOCK_K=32` + double buffering?** It fits (8 × 8 KB = 64 KB) and looks
like the obvious answer. It is measurably worse — 400 vs 476 TFLOPS — and the
reason is global, not LDS: at `BLOCK_K=64` a tile row is 128 B and one
`buffer_load_dwordx4` touches 8 back-to-back cache lines; at `BLOCK_K=32` a row is
64 B, the same instruction touches 16 lines, and TCP pressure roughly doubles.

### MFMA shape and register budget

CDNA3's widest f16/bf16 16×16 intrinsic is `v_mfma_f32_16x16x16_f16` (K=16, 16
cycles) rather than CDNA4's `..._16x16x32`. `k_width=8` still yields
`ds_read_b128`: the dot-operand K tile is 8 × (64/16) = 32, which Triton splits
into two K=16 mfmas.

Per lane, with 1 wave/SIMD so 256 VGPR **and** 256 AGPR are available:

```
C accumulators   4 x [128x128] f32   = 256   -> AGPR (amdgpu-agpr-alloc=256)
dot operands     4 x half-tile f16   = 128   -> VGPR
global staging   4 x half-tile f16   =  64   -> VGPR
                                       ----
                                       448
```

The accumulators only fit because AGPRs are separate capacity at 1 wave/SIMD —
the opposite of [`../inter_wave`](../inter_wave/), where two resident waves split
one unified 512-register file and AGPRs buy nothing.

### LDS layout

`SwizzledSharedLayout(8, 1, 8)`, derived in [`../README.md`](../README.md) §4.4
together with the global-load lane remapping it requires. Making it conflict-free
is worth only **−1.0% loop cycles** here, against −9.9% in
[`../inter_wave`](../inter_wave/): at 1 wave/SIMD the LDS duty cycle is low enough
that a 40% saving in LDS *cycles* barely surfaces in loop cycles. That asymmetry
is itself the diagnosis — this kernel is not LDS-throughput-bound, it is bound by
the `ds_write` **issue** cost, below.

## Differences from the gfx950 kernel

The 2×2 quadrant slicing, the `B_left → A_top → A_bot → B_right` region order and
the XCD-aware PID remap are all inherited unchanged from
[`v9_beyond_hotloop`](../../intra_wave/a16w16/v9_beyond_hotloop). The *platform*
differences — no async copy, 64 KB of LDS, the 16-cycle MFMA, 128 B/clk LDS, the
swizzle — are in [`../README.md`](../README.md) §4. What changes at the **kernel**
level is smaller than that list suggests:

| | gfx950 v9 | **this kernel** |
|---|---|---|
| LDS pipelining | `nBuffers = 2`, double-buffered | **half-tile slot recycling**, 1 buffer |
| mfma / K-step / wave | 128 | **256** |
| loop | 2× unrolled | not unrolled |
| accumulators | VGPR | **AGPR** (`amdgpu-agpr-alloc=256`) |
| scheduling | in-tree pipeliner is enough | **LLIR plugin required** |

Two of those are worth explaining.

**Accumulators move to AGPRs** because at 1 wave/SIMD the AGPR file is separate
capacity rather than a slice of a shared one. The register round-trip costs 64
VGPRs of staging gfx950 never spends, and adding 256 f32 of accumulator on top
overflows the VGPR file; `amdgpu-agpr-alloc=256` is what makes the budget close.
[`../inter_wave`](../inter_wave/) is the exact opposite — two resident waves split
one unified 512-register file, so AGPRs buy nothing there.

**The LLIR plugin becomes required** because this is a scheduling problem rather
than a bandwidth one, and CDNA3's in-tree pipeliner does not solve it. It leaves
the memory ops clumped — `R4 … W4 … G5` separated by runs of 20–70 MFMAs — so an
8-deep `ds_read` burst has to drain with only ~96 cycles of compute to cover it.
The plugin spreads them, but it had to be taught CDNA3 first: its MFMA table held
only CDNA4 shape names, so it returned 0 cycles for `mfma.f32.16x16x16f16` and was
silently skipping every region.

The remaining difference is the one that cannot be scheduled away, and it is the
subject of the next section.

## Where the time goes, and the ceiling

Ablating the hot loop op-by-op (4096²×8192 fp16, the shape used while developing)
isolates the cost:

| in-loop ops | ms | vs MFMA-only |
|---|---|---|
| mfma only | 0.379 | 1.00× |
| + `local_load` | 0.372 | free |
| + `buffer_load` | 0.373 | free |
| + `local_store` | 0.417 | +10% |
| `local_load` + `local_store` | 0.536 | **+41%** |
| all three | 0.582 | +54% |

**Neither memory op costs anything on its own.** The +41% appears only when both
are present — it is the `s_barrier`s that the load/store pair forces membar to
insert (§4.1), plus the `s_waitcnt lgkmcnt(0)` each drags along. Two things that
did *not* fix it: regrouping the stores to cut barriers from 3 per K-step to 2
(worse, 428 TFLOPS — the resulting clumping costs more than the barrier saved) and
splitting into 8 shorter regions (worse, 438).

A sharper probe: delete the 16 `ds_write_b128` from the compiled loop and
reassemble via `TRITON_KERNEL_OVERRIDE`, holding register allocation and
everything else constant. The result is numerically wrong — it is a timing probe.

| | cyc/iter | MFMA eff |
|---|---|---|
| as shipped | 4621.5 | 88.63% |
| `ds_write` removed | 4255.0 | **96.26%** |

`ds_write` is **~two thirds of all remaining inefficiency**, and it is not
bandwidth: a `ds_write_b128` issues in ~20 cycles (4 for the address, 16 for the
data) while a `v_mfma_f32_16x16x16_f16` covers only 16, so **one MFMA cannot hide
one LDS write**. 16 × 20 = 320 unhideable cycles per iteration, against 366.5
measured. Full analysis in [`note.md`](note.md) §6.

**This is the limit of the intra_wave design on MI300.** Closing it means writing
*less* to LDS — a different pipeline, not a tuning knob — or not paying the cost in
this instruction stream at all. The latter is exactly what
[`../inter_wave`](../inter_wave/) does, by handing the writes to the other wave
group while this one is on the matrix unit, and it is why that kernel reaches ~98%
per-SIMD instead of 89%.

**Next:** [`../inter_wave/README.md`](../inter_wave/README.md).

## Optimization history

Detailed in [`note.md`](note.md); summarised:

| | cyc/iter | MFMA eff | what changed |
|---|---|---|---|
| baseline | 5877 | 69.7% | as first ported |
| opt 1 | 5343 | 76.7% | LLIR scheduler taught CDNA3 MFMA shapes + LDS bandwidth |
| opt 2 | 4874 | 84.0% | surplus MFMAs parked between the last LR and the first LW |
| opt 3 | 4601 | 89.0% | prologue LDS reads drained before the loop: 3→2 barriers, 3→1 `lgkmcnt(0)` |
| opt 4 | — | — | **rejected** — register-form change, 1.1% slower |
| opt 5 | 4622 | 88.4% | prologue load order pinned; `vmcnt(3)/(1)/(0)` → `(15)/(13)/(12)` |
| **+ swizzle** | **4588** | **89.3%** | conflict-free LDS layout |

Opt 4 and opt 5 are worth reading together: chasing a conservative `vmcnt` led
first to a wrong conclusion (an LLVM register-form flag) and then to the real
cause — the prologue issuing the loop-carried global loads in a rotated order, so
that `SIInsertWaitcnts` could not preserve their age ordering across the loop
join. There is no LLVM bug; the fix is one `gl.barrier()` in the prologue.
