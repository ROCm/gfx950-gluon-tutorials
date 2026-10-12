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
python tools/run_att.py intra_wave --K 8256 --out att_intra     # ATT trace
```

Two things are **required** for this kernel, unlike
[`../inter_wave`](../inter_wave/), and both are in the kernel itself:

* `cd_regclass="a"` on every MFMA, which pins the 256 f32 accumulators to AGPRs.
  Without it they compete with the 192 VGPRs of operands and staging.
* Triton's MFMA scheduler, which is what interleaves the memory ops through the
  MFMA stream: `matmul()` passes `schedule_hint="mfma-schedule"`, and the tools
  take `--schedule-hint ""` to turn it off. It runs on gfx942 since the `v3.1` pin
  ([triton-lang/triton#12229](https://github.com/triton-lang/triton/pull/12229),
  [`../README.md`](../README.md) §5).

Codegen: **256 arch VGPR + 256 AGPR, 1 wave/SIMD** (LDS-bound), 2 VGPRs (12 B) of
spill outside the loop, 65536 B LDS, and `SQ_LDS_BANK_CONFLICT` measures exactly
**0**.

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
like the obvious answer. It is measurably worse — 400 vs 476 TFLOPS, measured
during the port on `v1.1` — and the
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
C accumulators   4 x [128x128] f32   = 256   -> AGPR (cd_regclass="a")
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
is worth **−16.2% loop cycles** here (−10.8% in [`../inter_wave`](../inter_wave/)).
On `v1.1` it looked almost free here, −1.0%, because the plugin parked surplus
MFMAs between the last LDS read and the first write and that slack absorbed the
replays; Triton's MFMA scheduler leaves no such slack, so a replayed access costs
loop cycles directly ([`../README.md`](../README.md) §4.4). With the layout right,
what bounds this kernel is the `ds_write` **issue** cost, below.

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
| MFMA scheduler | required, upstream since `v3.0` | required, upstream since `v3.1`, **with CDNA3's LDS bandwidth** |
| `amdgcnas` peephole | used (worth the last 97% → 98%) | **not tried** |

Several things are deliberately *the same* and worth noting because they are easy
to assume differ:

* **Accumulators are in AGPRs on both.** v9 pins them the same way, with
  `cd_regclass="a"` on every MFMA. What differs is only how much slack there is:
  the register round-trip costs 64 VGPRs of staging gfx950 never spends, so the
  same pins are doing more work here.
  [`../inter_wave`](../inter_wave/) is the genuine contrast — two resident waves
  split one unified 512-register file, so AGPRs buy nothing there and it runs with
  `amdgpu-agpr-alloc=0,0`.
* **Both need the MFMA scheduler.** The stock machine scheduler leaves memory
  clumped on gfx950 too; that is why the scheduler exists (v5 introduces it, and it
  takes v9's MFMA efficiency from 58% to 80%). The difference is the LDS cost: CDNA3
  serves 128 B/clk against CDNA4's 256, so a `ds_read_b128` needs twice the MFMA
  cover here. Until `v3.1` the scheduler ran on gfx950 only and was a silent no-op on
  gfx942. See [`../README.md`](../README.md) §5.

The one tooling gap is `amdgcnas`, the post-assembly peephole that packs the
scattered SALU at iteration boundaries and is worth gfx950's last 97% → 98%. It
has not been tried here, so some of the remaining gap may be the same SALU cost
rather than anything CDNA3-specific.

The difference that cannot be scheduled or tooled away is the subject of the next
section.

## Where the time goes, and the ceiling

Ablating the hot loop op-by-op (4096²×8192 fp16, the shape used while developing,
on `v1.1` with the plugin schedule) isolates the cost:

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
| as shipped | 4507.2 | 90.88% |
| `ds_write` removed | 4173.5 | **98.14%** |

`ds_write` is **~80% of all remaining inefficiency**, and it is not
bandwidth: a `ds_write_b128` issues in ~20 cycles (4 for the address, 16 for the
data) while a `v_mfma_f32_16x16x16_f16` covers only 16, so **one MFMA cannot hide
one LDS write**. 16 × 20 = 320 unhideable cycles per iteration, against 333.7
measured (fp16 K=4160, median of 3 traces each).

**This is the limit of the intra_wave design on MI300.** Closing it means writing
*less* to LDS — a different pipeline, not a tuning knob — or not paying the cost in
this instruction stream at all. The latter is exactly what
[`../inter_wave`](../inter_wave/) does, by handing the writes to the other wave
group while this one is on the matrix unit, and it is why that kernel reaches ~99.7%
per-SIMD instead of ~91%.

**Next:** [`../inter_wave/README.md`](../inter_wave/README.md).

## Optimization history

At fp16 K=8256, decoded with `scripts/process_json.py` (every row up to "+ swizzle"
on `v1.1` with the LLIR scheduler plugin; that row is the final `v1.1` build at
K=4160, whose K=8256 traces scatter too much to use):

| | cyc/iter | MFMA eff | what changed |
|---|---|---|---|
| baseline | 5877 | 69.7% | as first ported |
| opt 1 | 5343 | 76.7% | LLIR scheduler taught CDNA3 MFMA shapes + LDS bandwidth |
| opt 2 | 4874 | 84.0% | surplus MFMAs parked between the last LR and the first LW |
| opt 3 | 4601 | 89.0% | prologue LDS reads drained before the loop: 3→2 barriers, 3→1 `lgkmcnt(0)` |
| opt 4 | — | — | **rejected** — register-form change, 1.1% slower |
| opt 5 | 4622 | 88.4% | prologue load order pinned; `vmcnt(3)/(1)/(0)` → `(15)/(13)/(12)` |
| + swizzle | 4551 | 90.0% | conflict-free LDS layout |
| **`v3.1`** | **4509.1** | **90.8%** | Triton's MFMA scheduler, upstream on gfx942 with CDNA3's LDS cost ([#12229](https://github.com/triton-lang/triton/pull/12229)); opt 2 dropped ([`../README.md`](../README.md) §5) |

Opt 4 and opt 5 are worth reading together: chasing a conservative `vmcnt` led
first to a wrong conclusion (an LLVM register-form flag) and then to the real
cause — the prologue issuing the loop-carried global loads in a rotated order, so
that `SIInsertWaitcnts` could not preserve their age ordering across the loop
join. There is no LLVM bug; the fix is one `gl.barrier()` in the prologue.
