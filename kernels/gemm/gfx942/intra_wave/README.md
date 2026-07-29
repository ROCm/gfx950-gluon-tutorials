# intra_wave — 4-wave a16w16 GEMM for gfx942 / MI300X

Port of the gfx950 tutorial's [`intra_wave/a16w16/v9_beyond_hotloop`](../../intra_wave/a16w16/v9_beyond_hotloop):
4 warps/CTA = **1 wave/SIMD**, a 256×256 output tile sliced into a 2×2 grid of
128×128 quadrants, and XCD-aware PID remapping with `GROUP_SIZE_M=4`.

```bash
cd ..                                              # kernels/gemm/gfx942
python bench.py -k intra_wave                      # correctness + TFLOPS
python tools/bench_prepared.py --kernel intra_wave --gpus all   # sustained timing
python tools/lds_conflict.py  --kernel intra_wave              # LDS bank conflicts
python tools/run_att.py intra_wave --K 8256 --out att_intra    # ATT trace
```

`bench.py` sets `TRITON_FORCE_MFMA_AGPR=1`, which supplies the
`amdgpu-agpr-alloc=256` hint. Without it the 256 f32 accumulators compete with
the 192 VGPRs of operands and staging and the kernel spills. The LLIR scheduler
plugin is also required for the numbers below:

```bash
LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 python bench.py -k intra_wave
```

## Performance

GPU 3 of an MI300X node, 4096×4864×8256, via `tools/bench_prepared.py`: prepared
launch, 3 rotating tensor sets, 1000 dispatches, **mean of the last 100**, timed
with `rocprofv3 --kernel-trace`. Efficiency from ATT.

| | TFLOPS | cyc/iter | mfma/iter/wave | **in-loop MFMA eff** |
|---|---|---|---|---|
| fp16 | **611.3** | 4587.6 | 256 | **89.28%** |
| bf16 | **655.1** | — | 256 | — |

At 1 wave/SIMD the per-wave and per-SIMD figures are the same number, unlike
[`inter_wave`](../inter_wave/).

All tested shapes (`bench.py`, `do_bench`, correctness checked against fp32):

| M | N | K | fp16 | bf16 | torch fp16 |
|---|---|---|---|---|---|
| 4096 | 4864 | 2112 | 594.7 | 618.6 | 586.2 |
| 4096 | 4864 | 4160 | 645.4 | 676.6 | 625.2 |
| 4096 | 4864 | 8256 | 577.5 | 669.7 | 572.1 |
| 4096 | 4864 | 16448 | 631.7 | 684.7 | 589.3 |

Codegen: **256 arch VGPR + 256 AGPR, 0 spills, 1 wave/SIMD** (LDS-bound),
65536 B LDS, and `SQ_LDS_BANK_CONFLICT` measures exactly **0**.

The optimization history is in [`note.md`](note.md); summarised:

| | cyc/iter | MFMA eff | what changed |
|---|---|---|---|
| baseline | 5877 | 69.7% | as first ported |
| opt 1 | 5343 | 76.7% | LLIR scheduler taught CDNA3 MFMA shapes + LDS bandwidth |
| opt 2 | 4874 | 84.0% | surplus MFMAs parked between the last LR and the first LW |
| opt 3 | 4601 | 89.0% | prologue LDS reads drained before the loop: 3→2 barriers, 3→1 `lgkmcnt(0)` |
| opt 4 | — | — | **rejected** — register-form change, 1.1% slower |
| opt 5 | 4622 | 88.4% | prologue load order pinned; `vmcnt(3)/(1)/(0)` → `(15)/(13)/(12)` |
| **+ swizzle** | **4588** | **89.3%** | conflict-free LDS layout |

## Design

Two things are genuinely different from gfx950, and they drive everything:

**1. No `buffer_load_to_shared`.** gfx950 streams HBM → LDS with an async copy.
CDNA3 must go `buffer_load` (HBM → VGPR) → `local_store` (VGPR → LDS) →
`local_load` (LDS → VGPR). The register round-trip costs 64 VGPRs of staging, and
the LDS producer/consumer hazard is closed by real `s_barrier`s rather than an
async counter.

**2. Half the LDS.** gfx950 has 160 KB/CU and double-buffers a 256×256×64 stage.
gfx942 has 64 KB — *exactly one* stage, so double buffering is impossible.

Rather than shrink `BLOCK_K` to 32 (which would halve the contiguous run of each
global load from 128 B to 64 B and roughly double TCP cache-line pressure), this
kernel keeps `BLOCK_K = 64` and recycles LDS at **half-tile granularity**. Each
of the four half-tiles owns a 16 KB slot, refilled one region after its last
read:

| region | DOT | LR (next region's operand) | LW | GR |
|---|---|---|---|---|
| 0 | `C_tl` | `A_bot(k)` | `B_left(k+1)` | `B_left(k+2)` |
| 1 | `C_bl` | `B_right(k)` | `A_top(k+1)` | `A_top(k+2)` |
| 2 | `C_tr` | `B_left(k+1)` | `A_bot(k+1)` | `A_bot(k+2)` |
| 3 | `C_br` | `A_top(k+1)` | `B_right(k+1)` | `B_right(k+2)` |

Every slot's write lands strictly between its previous read and its next read,
one region apart on both sides, so one barrier per region boundary is sufficient.
One region is 64 mfma ≈ 1024 cycles, ample to absorb the barrier plus the LDS
traffic. The pipeline depth this buys equals gfx950's double buffer:
`GR(k+2) → LW(k+1)` is a full K-step (~4096 cycles) of global-latency hiding.

**MFMA shape.** CDNA3's widest f16/bf16 16×16 intrinsic is
`v_mfma_f32_16x16x16_f16` (K=16, 16 cycles) rather than CDNA4's `..._16x16x32`.
`k_width=8` still yields `ds_read_b128`: the dot-operand K tile is 8 × (64/16) =
32, which Triton splits into two K=16 mfmas.

**Register budget** (per lane; 1 wave/SIMD, so 256 VGPR + 256 AGPR available):

```
C accumulators   4 x [128x128] f32   = 256   -> AGPR (amdgpu-agpr-alloc=256)
dot operands     4 x half-tile f16   = 128   -> VGPR
global staging   4 x half-tile f16   =  64   -> VGPR
```

### LDS layout — `SwizzledSharedLayout(8, 1, 8)`

`per_phase = 1`, **not 2**. The global-load layout has consecutive lanes walking
consecutive rows, so lanes 0–15 of a `b128` access cover two adjacent rows; with
`per_phase = 2` those rows share a swizzle phase, land on the same 32 banks, and
every access replays. Measured with `tools/lds_conflict.py --sweep`:

| | conflict ratio | cyc / LDS instr | `SQ_LDS_IDX_ACTIVE` |
|---|---|---|---|
| `(8, 2, 8)` | 0.667 | 13.33 | 2.510e9 |
| **`(8, 1, 8)`** | **0.000** | **8.00** | **1.506e9** |

Padding (gfx950 uses `PaddedSharedLayout([[512, 16]])`) is not affordable here —
the four slots fill LDS to the byte — so the swizzle has to do the work. The
40% LDS-cycle saving is worth only −1.0% loop cycles in this kernel, versus
−9.9% in `inter_wave`, because at 1 wave/SIMD the LDS duty cycle is far lower.
See [`../inter_wave/README.md`](../inter_wave/README.md) for the parameter
semantics.

> [!WARNING]
> `tools/layout_check.py`'s analytical model reports `(8, 2, 8)` as
> conflict-free. It is wrong. Trust `SQ_LDS_BANK_CONFLICT` over the model.

## The ceiling

Assembly-level ablation — delete the 16 `ds_write_b128` from the loop and
reassemble via `TRITON_KERNEL_OVERRIDE`, holding everything else constant (the
result is numerically wrong; it is a timing probe):

| | cyc/iter | MFMA eff |
|---|---|---|
| as shipped | 4621.5 | 88.63% |
| `ds_write` removed | 4255.0 | **96.26%** |

`ds_write` is **~two thirds of all remaining inefficiency**, and it is not
bandwidth: a `ds_write_b128` issues in ~20 cycles (4 address + 16 data) while a
`v_mfma_f32_16x16x16_f16` covers only 16, so one MFMA cannot hide one write.
16 × 20 = 320 unhideable cycles per iteration against 366.5 measured. **This is
the limit of the intra_wave design on MI300** — going further means writing less
to LDS, which is a different pipeline, not a tuning knob. Full analysis in
[`note.md`](note.md) §6.
