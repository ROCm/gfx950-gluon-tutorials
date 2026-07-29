# inter_wave — 8-wave warp-pipeline a16w16 GEMM for gfx942 / MI300X

8 warps per CTA = **2 waves per SIMD**, `warps_per_cta = [2, 4]`, tile
256×256×64, `v_mfma_f32_16x16x16_f16`. The hot loop is eight barrier-delimited
regions with memory and matrix work **strictly alternating**, so the two wave
groups ping-pong: while one group holds the matrix unit the other issues its
global loads, LDS reads and LDS writes.

```bash
cd ..                                              # kernels/gemm/gfx942
python bench.py -k inter_wave                      # correctness + TFLOPS
python tools/bench_prepared.py --kernel inter_wave --gpus all   # sustained timing
python tools/lds_conflict.py  --kernel inter_wave              # LDS bank conflicts
python tools/run_att.py inter_wave --K 8256 --out att_inter    # ATT trace
```

No environment variables needed — the no-AGPR setting is baked into the launch.

## Performance

GPU 3 of an MI300X node, 4096×4864×8256, via `tools/bench_prepared.py`: prepared
launch, 3 rotating tensor sets, 1000 dispatches, **mean of the last 100**, timed
with `rocprofv3 --kernel-trace`. Efficiency from ATT.

| | TFLOPS | cyc/iter | mfma/iter/wave | per-wave | **per-SIMD MFMA eff** |
|---|---|---|---|---|---|
| fp16 | **606.8** | 4192.5 | 128 | 48.85% | **97.70%** |
| bf16 | **648.1** | 4158.2 | 128 | 49.25% | **98.51%** |

Per-SIMD is the meaningful figure: the matrix unit is a per-SIMD resource, so for
a 2-wave-per-SIMD kernel it is the per-wave fraction × 2. At ~98% the loop is
essentially at the matrix-unit ceiling.

Against Triton's own `BlockPingpong` reference (`03-matrix-multiplication.py` at
256×256×64 / `num_warps=8`, with the loop-variant K mask removed so pingpong
fires):

| | cyc/iter | per-SIMD | TFLOPS | MFMA | clock |
|---|---|---|---|---|---|
| **this kernel** | **4192.5** | **97.70%** | **606.8** | `16x16x16` | 1.20–1.26 GHz |
| Triton pingpong | 4273.8 | 95.84% | 543.0 | `32x32x8` | 1.023 GHz |

Ahead on both cycles and efficiency, and further ahead on wall clock because
`16x16x16` is less power-dense than `32x32x8`: this part sits on a 750 W cap, so
the reference's denser MFMA stream clocks itself down. **Compare cycles, not
microseconds** — see [../README.md](../README.md) §3.3.

All tested shapes (`bench.py`, `do_bench`, correctness checked against fp32):

| M | N | K | fp16 | bf16 | torch fp16 |
|---|---|---|---|---|---|
| 4096 | 4864 | 2112 | 620.1 | 660.8 | 554.6 |
| 4096 | 4864 | 4160 | 677.2 | 702.5 | 626.1 |
| 4096 | 4864 | 8256 | 586.3 | 635.2 | 570.1 |
| 4096 | 4864 | 16448 | 640.6 | 684.2 | 588.2 |

Codegen: **240 VGPR, 0 AGPR, 0 spills, 2 waves/SIMD, 64 KB LDS**, and
`SQ_LDS_BANK_CONFLICT` measures exactly **0**.

How it got here:

| | cyc/iter | per-SIMD | barriers/K-step |
|---|---|---|---|
| initial port (4 quadrants × 2 K-slices, 16 regions) | 4770.6 | 85.86% | 16 |
| 8-region redesign | 4651.0 | 88.06% | 8 |
| **+ conflict-free swizzle** | **4192.5** | **97.70%** | 8 |

## Design

### Whole-tensor global and LDS writes, sliced LDS reads

Global loads and LDS stores move whole tensors; reads and dots work on
half-tiles sliced out of the same buffers:

```
GR A / LW A    A[256 x 64]    4 x buffer_load_dwordx4  /  4 x ds_write_b128
GR B / LW B    B[64 x 256]    4 x buffer_load_dwordx4  /  4 x ds_write_b128

A_t = A[0:128, :]   A_b = A[128:256, :]   B_l = B[:, 0:128]   B_r = B[:, 128:256]
```

The slice is along M (for A) or N (for B) — never along K, the swizzled
dimension — so the shared layout is undisturbed by slicing.

### One LDS buffer; the registers are the pipeline

```
A[256 x 64] x 2 B  +  B[64 x 256] x 2 B  =  32 KB + 32 KB  =  64 KB
```

exactly MI300X's LDS, so double buffering is impossible — and unnecessary.
**The pipelining buffer is the registers holding the in-flight global loads, not
LDS.** LDS holds tile *k* while tile *k+1* is in flight from HBM; the
`local_store` of *k+1* lands only after the last `local_load` of *k*.

### The eight regions

Region *k* processes tile *k* and prefetches tile *k+1*:

| region | kind | work | instructions | cycles |
|---|---|---|---|---|
| 0 | mem | `GR B[k+1]`, `LR A_t[k]` | 4 `buffer_load_dwordx4` + 8 `ds_read_b128` | 256 |
| 1 | dot | `C_tl += A_t × B_l` | 32 mfma | 512 |
| 2 | mem | `GR A[k+1]`, `LR B_r[k]` | 4 `buffer_load_dwordx4` + 4 `ds_read_b128` | 128 |
| 3 | dot | `C_tr += A_t × B_r` | 32 mfma | 512 |
| 4 | mem | `LR A_b[k]`, `LW B[k+1]` | 8 `ds_read_b128` + 4 `ds_write_b128` | 384 |
| 5 | dot | `C_bl += A_b × B_l` | 32 mfma | 512 |
| 6 | mem | `LR B_l[k+1]`, `LW A[k+1]` | 4 `ds_read_b128` + 4 `ds_write_b128` | 256 |
| 7 | dot | `C_br += A_b × B_r` | 32 mfma | 512 |

Memory-region cycles are LDS-port time with 4 of the 8 waves in the memory phase
at 128 B/clk. **Every memory region fits inside the dot region it ping-pongs
against** (256/128/384/256 against 512) — that is the property which keeps the
matrix unit fed, and the reason this schedule reaches ~98%.

### Why this order

**Dot order** `C_tl → C_tr → C_bl → C_br` together with **read order**
`B_l → A_t → B_r → A_b` halves the operand registers. A_t is live only across
regions 1–3 and A_b only across 5–7, so **the two share one register set**; B_l
(used in 1 and 5) and B_r (used in 3 and 7) both span the body and need their
own. Per lane, replicated across the warp grid:

```
A_t / A_b    [128x64] over WARPS_M=2     32 VGPR  (shared)
B_l , B_r    [64x128] over WARPS_N=4     16 VGPR each
C quadrants  4 x [128x128] f32          128 VGPR
A,B staging  whole tiles / 8 waves        32 VGPR
```

Independent registers for all four half-tiles would cost 96 instead of 64 and
spill. An earlier version avoided that by reading K=32 slices instead, at the
price of 16 regions and twice the barriers.

**GR/LW order** is B before A on both sides. Each store sits 2 dot regions
(~1024 cycles) after its load, which is the global-latency cover; B is stored
first because `LR B_l[k+1]` in region 6 needs it.

**Hazards** are all closed by region boundaries: A is read in regions 0 and 4 and
written in 6; B is read in 2 (and in 6, for *k+1*) and written in 4.

### LDS layout — `SwizzledSharedLayout(8, 1, 8)`

The three parameters place element `(r, c)` at column
`((c / vec) ^ phase) * vec + (c % vec)`, with `phase = (r / per_phase) % max_phase`:

* `vec = 8` — swizzle granularity; 8 fp16 = 16 B, which keeps `ds_read_b128` /
  `ds_write_b128` legal.
* `per_phase = 1` — one row per phase.
* `max_phase = 8` — 8 distinct phases before the pattern repeats.

`per_phase = 1`, **not 2**. The global-load layout has consecutive lanes walking
consecutive rows, so lanes 0–15 of a `b128` access cover **two adjacent rows**.
With `per_phase = 2` those rows share a phase, land on the same 32 banks, and
every access replays. Measured with `tools/lds_conflict.py --sweep`:

| | conflict ratio | cyc / LDS instr | `SQ_LDS_IDX_ACTIVE` |
|---|---|---|---|
| `(8, 2, 8)` | 0.750 | 14.00 | 3.514e9 |
| **`(8, 1, 8)`** | **0.000** | **8.00** | **2.008e9** |

8.00 cyc/instr is the floor (1024 B at 128 B/clk). Worth **−9.9% loop cycles** —
the single largest win in this kernel.

> [!WARNING]
> `tools/layout_check.py`'s analytical model reports `(8, 2, 8)` as
> conflict-free. It is wrong, and it is what led to the original choice. Trust
> `SQ_LDS_BANK_CONFLICT` over the model.

## Where the remaining time goes

Per wave per iteration, from ATT:

| | this kernel | Triton pingpong |
|---|---|---|
| mfma | 1972.0 | 1969.9 |
| `s_barrier` | 801.9 | 631.6 |
| `ds_write` | 529.9 | 293.9 |
| `ds_read` | 284.2 | 560.4 |
| `s_waitcnt` | 282.8 | 356.1 |

`ds_read` is now half the reference's. `ds_write` is still 1.8× it: the
reference's store region is *pure*, whereas regions 4 and 6 here still mix reads
with writes and pay LDS read/write turnaround. Isolating all 8 `ds_write`s into
one read-free region is the obvious next experiment.
