# intra_wave — 4-wave a16w16 GEMM for gfx942 / MI300X

Port of the gfx950 tutorial's [`intra_wave/a16w16/v9_beyond_hotloop`](../../intra_wave/a16w16/v9_beyond_hotloop):
4 warps/CTA = **1 wave/SIMD**, a 256×256 output tile sliced into a 2×2 grid of
128×128 quadrants, and XCD-aware PID remapping with `GROUP_SIZE_M=4`.

Performance numbers for both gfx942 kernels are in [`../README.md`](../README.md)
§3. This file is about the design.

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

### LDS layout — `SwizzledSharedLayout(8, 1, 8)`

`per_phase = 1`, **not 2**. The global-load layout has consecutive lanes walking
consecutive rows, so lanes 0–15 of a `b128` access cover two adjacent rows; with
`per_phase = 2` those rows share a swizzle phase, land on the same 32 banks, and
every access replays. Measured with `tools/lds_conflict.py --sweep`:

| | conflict ratio | cyc / LDS instr | `SQ_LDS_IDX_ACTIVE` |
|---|---|---|---|
| `(8, 2, 8)` | 0.667 | 13.33 | 2.510e9 |
| **`(8, 1, 8)`** | **0.000** | **8.00** | **1.506e9** |

Getting the `local_store` side conflict-free also required changing the
global-load layout away from the tutorial's. The gfx950 layout has lanes 0–7 cover
row `M0` and lanes 8–15 cover row `M0+16`; with a 128 B row stride those two rows
land on the same banks and **no swizzle can separate them**, because the swizzle
only permutes chunks *within* a row. Remapping so consecutive lanes walk
consecutive rows makes lanes 0–15 cover two adjacent 128 B rows = 256 B = all 64
banks exactly once, and improves the global side too (one instruction now reads
1024 contiguous bytes).

The 40% LDS-cycle saving is worth only −1.0% loop cycles here, versus −9.9% in
`inter_wave`, because at 1 wave/SIMD the LDS duty cycle is far lower. See
[`../inter_wave/README.md`](../inter_wave/README.md) for the swizzle parameter
semantics.

> [!WARNING]
> `tools/layout_check.py`'s analytical model reports `(8, 2, 8)` as
> conflict-free. It is wrong. Trust `SQ_LDS_BANK_CONFLICT` over the model.

## Differences from the gfx950 kernel

Same 2×2 quadrant slicing, same `B_left → A_top → A_bot → B_right` region order,
same XCD-aware PID remap and `GROUP_SIZE_M` swizzle as
[`v9_beyond_hotloop`](../../intra_wave/a16w16/v9_beyond_hotloop). Everything below
is forced by CDNA3.

| | gfx950 (CDNA4) | **gfx942 (CDNA3)** |
|---|---|---|
| HBM → LDS | `buffer_load_to_shared` (async, direct) | `buffer_load` → **`local_store`** |
| synchronisation | `wait_group(n)` on async copies | **`s_barrier`** + `s_waitcnt lgkmcnt` |
| LDS capacity | 160 KB/CU | 64 KB/CU |
| LDS buffers | `nBuffers = 2`, double-buffered | **1**, half-tile slot recycling |
| LDS layout | `PaddedSharedLayout([[512, 16]])` | **`SwizzledSharedLayout(8, 1, 8)`** |
| global-load lanes | lanes 0–7 row `M0`, 8–15 row `M0+16` | **consecutive lanes → consecutive rows** |
| LDS bandwidth | 256 B/clk | 128 B/clk |
| MFMA | `v_mfma_f32_16x16x32_f16`, 32 cyc | `v_mfma_f32_16x16x16_f16`, 16 cyc |
| mfma / K-step / wave | 128 | **256** |
| loop | 2× unrolled | not unrolled |
| accumulators | VGPR | **AGPR** (`amdgpu-agpr-alloc=256`) |
| scheduling | in-tree pipeliner is enough | **LLIR plugin required** |

Three consequences worth spelling out.

**1. The register round-trip is the whole story.** gfx950 streams HBM → LDS with
one async instruction. CDNA3 must go `buffer_load` (HBM → VGPR) → `local_store`
(VGPR → LDS) → `local_load` (LDS → VGPR). That costs 64 VGPRs of staging that
gfx950 does not spend at all, and it adds 16 `ds_write_b128` per K-step to the LDS
port. Ablation (see [`note.md`](note.md) §6) shows those writes are **~two thirds
of all remaining inefficiency**: removing them takes in-loop MFMA efficiency from
88.6% to 96.3%.

**2. It is a scheduling problem, not a bandwidth problem — and CDNA3 needs help
with it.** Neither `local_load` nor `buffer_load` costs anything on its own here;
the cost appears only when both are present, because that is when membar must
insert barriers. The in-tree pipeliner leaves the memory ops clumped (`R4 … W4 …
G5` separated by runs of 20–70 MFMAs), so the LLIR scheduler plugin is required to
spread them — and it had to be taught CDNA3 first, since its MFMA table held only
CDNA4 shape names and it was silently skipping every region.

**3. The `ds_write` issue cost sets a hard ceiling.** A `ds_write_b128` issues in
~20 cycles (4 for the address, 16 for the data) while a `v_mfma_f32_16x16x16_f16`
covers only 16 — so **one MFMA cannot hide one LDS write**. 16 × 20 = 320
unhideable cycles per iteration, against 366.5 measured. On gfx950 the async copy
means there is no such instruction to hide. This is the limit of the intra_wave
design on MI300: going further means writing *less* to LDS, which is a different
pipeline rather than a tuning knob, and it is why
[`../inter_wave`](../inter_wave/) — which hides its writes behind the other wave
group — reaches ~98% instead.

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
