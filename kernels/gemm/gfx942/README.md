# gfx942 (MI300X) — a CDNA3 port of the a16w16 GEMM kernels

Two BF16/FP16 Gluon GEMM kernels for **gfx942 / MI300X (CDNA3)**, ported from
this repository's gfx950 kernels:

| | kernel | ported from |
|---|---|---|
| [`intra_wave/`](intra_wave/) | 4 warps/CTA, 1 wave/SIMD, compiler-interleaved | [`../intra_wave/a16w16/v9_beyond_hotloop`](../intra_wave/a16w16/v9_beyond_hotloop/) |
| [`inter_wave/`](inter_wave/) | 8 warps/CTA, 2 waves/SIMD, `warp_pipeline_stage` | [`../inter_wave/a16w16`](../inter_wave/a16w16/) |

> [!NOTE]
> These target **gfx942 (CDNA3)**, not the gfx950 (CDNA4) part the rest of the
> repository is written for. They are a separate port, not another version in
> the `v0 -> v9` arc: CDNA3 has no `buffer_load_to_shared` and half the LDS, and
> §4 below is entirely about what those two facts force to change.

Both keep the tutorial's core design — the 256×256 output tile sliced into a 2×2
grid of 128×128 quadrants and the XCD-aware PID remap with `GROUP_SIZE_M`
swizzling — and change only what CDNA3 forces them to change.

## How to read this

**Start here, then pick a kernel.** This file is the shared material: how to
build and run, every performance number for both kernels, and the CDNA3
constraints that both of them are shaped by. Neither kernel's design makes much
sense before §4.

| | |
|---|---|
| §1–2 | setup and how to run |
| **§3** | **all results** — both kernels, both dtypes, all shapes, one methodology |
| **§4** | **what CDNA3 forces to change** — the constraints common to both kernels |
| §5–6 | the out-of-tree plugins, and where the files are |

Then read the kernel you care about. Each of those files assumes §4 and covers
only its own design:

* [**`intra_wave/README.md`**](intra_wave/README.md) — 4 waves, 1 per SIMD. The
  simpler shape: one wave per SIMD, so the compiler has to interleave memory with
  MFMA *within* a single instruction stream. Read this one first; it is where the
  CDNA3 problems are easiest to see.
* [**`inter_wave/README.md`**](inter_wave/README.md) — 8 waves, 2 per SIMD. The
  answer to those problems: two wave groups ping-pong so one is on the matrix
  unit while the other is in memory, which hides the very costs
  [`intra_wave`](intra_wave/) has to pay in-line.

## 1. Setup

Same Triton pin as the rest of the repository (see [`SUPPORT.md`](../../../SUPPORT.md)):

```bash
git clone https://github.com/triton-lang/triton -b gfx950-tutorial-v1.1 /path/to/triton
pip uninstall -y triton
cd /path/to/triton && TRITON_EXT_ENABLED=1 pip install -e .
```

`gfx950-tutorial-v1.1` is the annotated tag the tutorial pins (see its
`SUPPORT.md`); it reports itself as Triton 3.8.0. `TRITON_EXT_ENABLED=1` is only
needed if you also want to load the tutorial's out-of-tree LLIR-scheduler plugin
— see §5.

## 2. Running

```bash
cd kernels/gemm/gfx942
python bench.py                                   # both kernels, fp16+bf16, vs torch
python bench.py -k inter_wave --K 4160 --dtype fp16 --show-clock
python bench.py --sweep-gm --K 4160               # GROUP_SIZE_M sweep
```

`bench.py` checks against an **fp32** reference (the kernels accumulate in fp32
and round once, so they are strictly more accurate than a half-precision
reference) and requires the error to be within 4 ulps of the peak output
magnitude. `--rocprof` switches to the tutorial's rotating-buffer mode for
external `rocprofv3 --kernel-trace` timing.

## 3. Results

MI300X (gfx942), 304 CUs, ROCm 7.2, Triton `gfx950-tutorial-v1.1`.

**One methodology for every number below**, so the tables are comparable with
each other: M=4096, N=4864 with K swept (see §3.1 for why that shape family), on
**GPU 3** — the fastest of the node's eight, by median over three sweeps, spread
4.0%. TFLOPS come from [`tools/bench_prepared.py`](tools/bench_prepared.py):
prepared launch (compile once, arguments pre-bound, launch stub entered
directly), 3 rotating tensor sets so no dispatch reads its predecessor's
cache-resident operands, 1000 dispatches, **mean of the last 100** kernel
durations from `rocprofv3 --kernel-trace`. `torch` is hipBLASLt via
`torch.matmul` through the *same* harness. Per-SIMD MFMA efficiency is from ATT.

**fp16**

| K | intra_wave TF | eff | inter_wave TF | eff | torch TF |
|---|---|---|---|---|---|
| 2112 | 586.4 | 87.10% | **620.0** | 96.71% | 583.4 |
| 4160 | 603.9 | 88.57% | **618.5** | 98.25% | 581.9 |
| 8256 | **614.3** | 89.28% | 598.3 | 97.81% | 579.6 |
| 16448 | 629.0 | 89.63% | **641.1** | 99.42% | 589.8 |

**bf16**

| K | intra_wave TF | eff | inter_wave TF | eff | torch TF |
|---|---|---|---|---|---|
| 2112 | 612.7 | 87.10% | **644.4** | 96.71% | 618.7 |
| 4160 | 642.5 | 88.56% | **658.3** | 98.26% | 626.6 |
| 8256 | 638.4 | 89.30% | **647.3** | 97.78% | 620.8 |
| 16448 | 678.6 | 89.63% | **689.9** | 99.42% | 627.2 |

`eff` is in-loop **per-SIMD** MFMA efficiency, the comparable figure: the matrix
unit is a per-SIMD resource, so the 8-wave kernel's per-wave fraction doubles for
its two co-resident waves. At 96.7–99.4% the 8-wave loop is essentially at the
matrix-unit ceiling; the 4-wave one sits ~10 points back, and
[`intra_wave/README.md`](intra_wave/README.md) shows via ablation that almost all
of that gap is `ds_write` the MFMA stream cannot hide.

Efficiency barely moves with dtype (fp16 and bf16 agree to three digits — the
codegen is identical) but does move with K, from 96.7% at K=2112 to 99.4% at
K=16448: a longer K means more loop iterations per prologue and better L2 reuse
on the global loads.

**inter_wave wins 7 of 8 points** and beats hipBLASLt on all 8. intra_wave beats
hipBLASLt on 7 of 8 — it loses bf16 K=2112 (612.7 vs 618.7).

For scale, Triton's own `BlockPingpong` schedule at 4096×4864×8256
(`python/tutorials/03-matrix-multiplication.py`, tile 256×256×64, `num_warps=8`,
with the loop-variant K mask removed so the pass fires) reaches 4273.8 cyc/iter
and 95.84% per-SIMD at 543.0 TFLOPS. It uses `v_mfma_f32_32x32x8_f16`, which is
more power-dense than the `16x16x16` used here and clocks down to 1.023 GHz for
it — so inter_wave is ahead on cycles, on efficiency, and by a wider margin on
wall clock. See §3.5.

Cycles per iteration per wave behind the efficiency column, for reference:

| K | intra_wave | inter_wave |
|---|---|---|
| 2112 | 4702.5 | 4235.5 |
| 4160 | 4624.8 | 4168.8 |
| 8256 | 4587.7 | 4187.7 |
| 16448 | 4569.8 | 4120.0 |

Codegen: intra_wave 256 VGPR + 256 AGPR, 0 spills, 1 wave/SIMD; inter_wave 240
VGPR, 0 AGPR, 0 spills, **2 waves/SIMD**. Both use exactly 65536 B of LDS, and
both measure `SQ_LDS_BANK_CONFLICT` = **0**.

> [!NOTE]
> `bench.py` reports `do_bench` medians on the *default* device instead, so its
> numbers will not match these: a different GPU (up to 4% across the node), a
> timer that includes the Python launch path and inter-dispatch gaps rather than
> kernel duration alone, and a shorter, cooler run. Use it for correctness
> checking and quick iteration; use the tables above for reporting.

### 3.1 Why these shapes

Both dimensions of the benchmark are chosen for gfx942 specifically, and both
matter a lot — the same kernels measured 460-540 TFLOPS on 4096x4096xPOW2.

**N = 4864, not 4096.** With a 256x256 tile, 4096x4864 is a 16 x 19 = **304
workgroup** grid, exactly the MI300X CU count: one workgroup per CU and no tail
wave. 4096x4096 gives 256 workgroups, so 48 of 304 CUs sit idle for the whole
kernel — a hard 16% ceiling that has nothing to do with the pipeline. (LDS is
full at 64 KB, so occupancy is 1 workgroup/CU and the grid must match the machine
exactly.)

**K = 4160 or 8256, not 4096 or 8192.** A power-of-two K makes A's row stride a
power of two (K=4096 -> 8192 B), so every row of a tile lands in the same L1 set
and the loads hotspot. `4160 = 65 x 64` and `8256 = 129 x 64` are still multiples
of `BLOCK_K` but make the stride an *odd* multiple of the 128 B line, spreading
the rows across sets.

Together these are worth ~20-30%: inter_wave goes from 510 TFLOPS at
4096x4096x8192 to 622 at 4096x4864x4160.

### 3.2 GROUP_SIZE_M

At this grid each XCD gets 304/8 = 38 workgroups, so the tutorial's model
(minimise `GM + ceil(38/GM)`) predicts GM = 5, 6 or 8 at f = 13, with GM = 4 just
behind at 14 and GM = 1/2/16 far worse at 39/21/19. Measured with
`bench.py --sweep-gm`, median of 4 runs at 4096x4864x4160 fp16:

| GM | f(GM) | intra_wave | inter_wave |
|---|---|---|---|
| 1 | 39 | 558 | 619 |
| 2 | 21 | 568 | 616 |
| 4 | 14 | 569 | 626 |
| 5 | 13 | 569 | 622 |
| 6 | 13 | 570 | 625 |
| 8 | 13 | 570 | 625 |
| 16 | 19 | 569 | 622 |

The model separates the bad choices correctly — GM = 1 and 2 cost 2-4% — but
GM = 4 through 8 are indistinguishable, with run-to-run spreads that overlap
completely. **`GROUP_SIZE_M = 4` is kept**; anything in 4-8 is equivalent here.

### 3.3 Reading the numbers

Sustained MFMA pushes this part into its 750 W cap and the clock falls from
~2.03 GHz to ~1.52 GHz; a `do_bench` burst runs near 2.0 GHz while a long loop
settles at ~1.55 GHz, so the same kernel measures 622 TFLOPS in one and ~470 in
the other. `--show-clock` prints the clock next to each row, and the torch column
is measured in the same loop as a same-conditions reference. K = 8256 is also
the noisiest point (±5% run to run); the tables above are medians of 3.

### 3.4 ATT traces

[`tools/run_att.py`](tools/run_att.py) wraps `rocprofv3 --att` and checks that the decoder produced
the `ui_*` directory ATTViewer needs:

```bash
bash scripts/install_att_decoder.sh   # once, from the repo root
cd tools
python run_att.py inter_wave --K 8256 --out att_inter_wave
```

### Sustained timing

[`tools/bench_prepared.py`](tools/bench_prepared.py) is the low-overhead timing path. It reuses the
tutorial's generic `scripts/prepared_kernel.py` to compile once and pre-bind every argument, so
`rocprofv3 --kernel-trace` sees back-to-back dispatches with no Python binding in the gap. It
rotates tensor sets so no dispatch reads its predecessor's cache-resident operands, and averages
the **last** 100 of 1000 dispatches, by which point the clock has settled under the 750 W cap.

It verifies the kernel's output before reporting any timing. The Triton cache key does not include
`TRITON_KERNEL_OVERRIDE`, so a hand-edited kernel from an ablation run can otherwise persist in
`~/.triton/cache` and every number will silently describe the wrong build.

```bash
python bench_prepared.py --gpus all      # sweep every GPU, report the fastest
python bench_prepared.py --gpus 3 --dtype bf16
```

Point ATTViewer at the `ui_output_agent_*_dispatch_*` directory it produces.
Two things the traces settle:

**Occupancy is visible in the file list.** The 4-wave kernel produces 16
wave-slot files (4 SE x 4 SIMD x `sl0`); the 8-wave kernel produces 32, adding
`sl1` -- the second resident wave group the warp pipeline needs.

**Per-SIMD MFMA efficiency**, decoded with the tutorial's `process_json.py`
(its `MFMA_CYCLE_MAP` already gives 16 cycles, correct for CDNA3's
`v_mfma_f32_16x16x16_f16`):

At K=8256 (all four K values are in §3):

| | mfma/iter/wave | cycles/iter/wave | per-wave | **per-SIMD** |
|---|---|---|---|---|
| intra_wave, as first ported | 256 | 5877 | 69.7% | **69.7%** (1 wave/SIMD) |
| intra_wave, current | 256 | 4587.7 | 89.3% | **89.3%** |
| inter_wave, as first ported | 128 | 4770.6 | 42.9% | **85.9%** (2 waves/SIMD) |
| inter_wave, current | 128 | **4187.7** | 48.9% | **97.8%** |

Both kernels moved a long way from the initial port: intra_wave through the
opt1-opt5 sequence in [`intra_wave/note.md`](intra_wave/note.md), inter_wave
through the 8-region redesign in [`inter_wave/README.md`](inter_wave/README.md).
The LDS bank-conflict fix in §4.4 is common to both and was worth -9.9% loop
cycles on the 8-wave kernel.

### 3.5 Why TFLOPS understates the 8-wave kernel

The clearest case is fp16 K=8256 in §3: the 8-wave kernel runs **8.7% fewer
cycles per K-step** (4187.7 vs 4587.7) at **8.5 points** higher per-SIMD MFMA
efficiency, and still measures **lower** TFLOPS — 598.3 against 614.3. The 750 W
cap is why. A first capture pair taken
back-to-back without a cooldown recorded:

| | cycles | wall | clock |
|---|---|---|---|
| intra_wave (ran first, cool) | 901,420 | 581.7 us | 1.55 GHz |
| inter_wave (ran second, hot) | 707,824 | 568.6 us | **1.24 GHz** |

a 27% cycle advantage almost exactly cancelled by a 20% clock deficit.
Reversing the run order moved the deficit to the other kernel, so it tracks
thermal history rather than the kernel -- but the effect is real and it is what
the do_bench numbers are measuring: **the better pipeline draws more power and
gets clocked down for it**, which is the same DIDT dynamic the tutorial's v7
describes.

So read §3 as "what you get on a hot part in a benchmark loop" and §3.4 as "how
good the pipeline actually is". Cycle counts and MFMA efficiency are the stable
measurements here; wall-clock TFLOPS are not comparable between kernels unless
the clock is recorded with them (`realtime.json` in every ATT capture carries
`[gfx_clock, realtime_clock]` samples for exactly this).

## 4. What CDNA3 forces to change

Everything in this section applies to **both** kernels — it is the platform, not
the schedule. What each kernel then does about it is in its own README.

### 4.1 No `buffer_load_to_shared` — the register round-trip

gfx950 streams HBM → LDS with `gl.amd.cdna4.async_copy.buffer_load_to_shared`
and synchronises with an async counter (`wait_group(n)`). CDNA3's Gluon surface
(`gl.amd.cdna3`) has `buffer_load`, `buffer_store` and `mfma` and no async copy,
so the path is

```
buffer_load  (HBM -> VGPR)  ->  local_store (VGPR -> LDS)  ->  local_load (LDS -> VGPR)  ->  mfma
```

Two consequences, and they drive both kernels:

1. **32–64 VGPRs of staging** that gfx950 does not pay, which is what makes the
   register budget tight enough to shape the schedule.
2. **The LDS producer/consumer hazard is closed by real `s_barrier`s** inserted by
   Triton's membar pass, rather than by an async counter. `wait_group(n)` is a
   per-wave ordered wait and does not serialise anything; `s_barrier` stops every
   wave in the CTA. This is the single largest cost CDNA3 adds, and the two
   kernels differ mainly in how they hide it.

### 4.2 Half the LDS — 64 KB is exactly one stage

gfx950 has 160 KB of LDS per CU and double-buffers a 256×256×64 stage (2 × 64 KB).
gfx942 has **64 KB**, i.e. exactly one stage. Double buffering at this tile size
is impossible, so on gfx942 **LDS cannot be the pipeline buffer** — the registers
holding in-flight global loads are, and LDS is refilled in place. Each kernel
schedules that refill differently; see their READMEs.

**Why not `BLOCK_K=32` + double buffering?** It fits (8 × 8 KB = 64 KB) and it
does reduce the barrier count, which is what you would expect to want. It is
measurably worse — 400 TFLOPS vs 476 — and the reason is global coalescing, not
LDS. With `BLOCK_K=64` a row of the tile is 128 B, so eight lanes cover a whole
cache line and one `buffer_load_dwordx4` touches 8 back-to-back lines. At
`BLOCK_K=32` a row is 64 B, the same instruction touches 16 lines, and TCP
processing time per instruction roughly doubles. Ablating the `buffer_load`s out
of a `BLOCK_K=32` kernel moves it from 386 to 503 TFLOPS; ablating them out of the
`BLOCK_K=64` kernel changes nothing. Deepening the global staging to two K-steps
did not help either (v4, 400 TFLOPS) — it is the cache-line count, not the
latency. **Both kernels therefore keep `BLOCK_K=64` and slice only on the LDS-read
side.**

### 4.3 MFMA shape

CDNA3's widest 16×16 f16/bf16 intrinsic is `v_mfma_f32_16x16x16_f16`
(`AMDMFMALayout(version=3, instr_shape=[16,16,16])`), **16 cycles**, versus CDNA4's
`..._16x16x32_f16` at 32. `k_width=8` still yields `ds_read_b128`: the dot-operand
K tile is `8 × (64/16) = 32`, which Triton lowers to two K=16 MFMAs.

Same total matrix work, twice the instructions — which matters less than it
sounds. A K-step is ~2048 cycles of MFMA per wave either way, so the *rate* at
which a schedule must feed the matrix unit is unchanged; only the instruction
count inside each region doubles.

CDNA3's LDS is also **128 B/clk against CDNA4's 256**, so every `ds_read_b128` /
`ds_write_b128` occupies the port twice as long relative to the same MFMA work.

### 4.4 LDS layout — `SwizzledSharedLayout(8, 1, 8)`

Both kernels use the same shared layout, and both had to co-design it with the
global-load layout. gfx950's `PaddedSharedLayout([[512, 16]])` spends LDS to
sidestep bank conflicts; with LDS full to the byte that option is gone, so the
swizzle has to be exactly right.

The parameters place element `(r, c)` at column
`((c / vec) ^ phase) * vec + (c % vec)`, with `phase = (r / per_phase) % max_phase`:

* `vec = 8` — swizzle granularity; 8 fp16 = 16 B, which keeps `ds_read_b128` /
  `ds_write_b128` legal.
* `per_phase = 1` — one row per phase.
* `max_phase = 8` — 8 distinct phases before the pattern repeats.

**`per_phase = 1`, not 2.** The global-load layout has consecutive lanes walking
consecutive rows, so lanes 0–15 of a `b128` access cover *two adjacent rows*. With
`per_phase = 2` those two rows share a phase, land on the same 32 banks, and every
access replays. Measured with [`tools/lds_conflict.py`](tools/lds_conflict.py),
which reads `SQ_LDS_BANK_CONFLICT` and `SQ_LDS_IDX_ACTIVE`:

| | conflict ratio | cyc / LDS instr | intra_wave `IDX_ACTIVE` | inter_wave `IDX_ACTIVE` |
|---|---|---|---|---|
| `(8, 2, 8)` | 0.667 / 0.750 | 13.33 / 14.00 | 2.510e9 | 3.514e9 |
| **`(8, 1, 8)`** | **0.000** | **8.00** | **1.506e9** | **2.008e9** |

8.00 cyc/instr is the floor (1024 B at 128 B/clk). Worth −1.0% loop cycles on the
4-wave kernel but **−9.9%** on the 8-wave one, whose LDS duty cycle is far higher
at 2 waves/SIMD.

Getting the `local_store` side conflict-free also required **changing the
global-load layout** away from the tutorial's. The gfx950 layout has lanes 0–7
cover row `M0` and lanes 8–15 cover row `M0+16`; with a 128 B row stride those two
rows land on the same banks and *no swizzle can separate them*, because the
swizzle only permutes chunks within a row. Remapping so consecutive lanes walk
consecutive rows makes lanes 0–15 cover two adjacent 128 B rows = 256 B = all 64
banks exactly once, and improves the global side too (one instruction now reads
1024 contiguous bytes).

> [!WARNING]
> Earlier versions of this port used `(8, 2, 8)` and described it as
> "bank-conflict free in both directions", on the strength of
> [`tools/layout_check.py`](tools/layout_check.py)'s analytical model. **The model
> is wrong** and still reports `(8, 2, 8)` as clean. Use
> [`tools/lds_conflict.py`](tools/lds_conflict.py), which measures the hardware
> counter and can sweep `(vec, per_phase, max_phase)`:
> ```bash
> python tools/lds_conflict.py --kernel inter_wave --sweep
> ```

### 4.5 Where the two kernels diverge

Given §4.1–4.4, the remaining question is *who hides the barriers and the LDS
writes*, and that is the whole difference between the two kernels:

| | [`intra_wave`](intra_wave/) (4 waves, 1/SIMD) | [`inter_wave`](inter_wave/) (8 waves, 2/SIMD) |
|---|---|---|
| hides memory behind | MFMA in the **same** instruction stream | the **other wave group** |
| needs | the LLIR scheduler plugin to interleave (§5) | `warp_pipeline_stage` and a barrier per region |
| accumulators | AGPR — separate capacity at 1 wave/SIMD | VGPR — 2 waves split one 512-register file |
| LDS refill | per half-tile, one region after its last read | whole tensors, late in the body |
| ceiling | `ds_write` issue cost the MFMA stream cannot absorb | ~98% per-SIMD; barriers and write turnaround |

The 4-wave kernel is the more instructive read and the 8-wave one is the faster
kernel. Both READMEs assume this section.

## 5. On the tutorial's out-of-tree plugins

The tutorial's `llirSched` plugin, `TRITON_FORCE_MFMA_AGPR`, and the `amdgcnas`
peephole were all tried here.

`TRITON_FORCE_MFMA_AGPR` is **required** by the 4-wave kernel (it is what
supplies `amdgpu-agpr-alloc=256`; `bench.py` sets it).

`libLlirSched.so` initially appeared to do nothing on gfx942 (467→472 TFLOPS at
4096²×8192, within noise). It was in fact a **silent no-op**: its MFMA table held
only CDNA4 shape names, so `getMFMACycles()` returned 0 for
`mfma.f32.16x16x16f16` and the pass dropped every region. Teaching it the CDNA3
shapes and deriving the LDS cover from bandwidth (128 B/clk on CDNA3 against
CDNA4's 256) made it fire, and it is now **required** for the 4-wave numbers in
§3 — from 69.7% to 76.7% in-loop MFMA efficiency, then 84.0% once surplus MFMAs
are parked between a region's last `local_load` and its first `local_store`.
Both changes are in `plugins/llir_scheduler/LlirSchedPlugin.cpp`; see
[`intra_wave/note.md`](intra_wave/note.md) §1-2. Run it with:

```bash
LLVM_PASS_PLUGIN_PATH=<repo>/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 python bench.py -k intra_wave
```

The 8-wave kernel does not use the plugin — its schedule is expressed directly
with `warp_pipeline_stage`. `amdgcnas` remains untried on gfx942.

## 6. Layout

```
kernels/gemm/gfx942/
  README.md                     this file
  bench.py                      correctness + TFLOPS for both kernels,
                                --sweep-gm, --rocprof
  intra_wave/matmul_kernel.py   4-wave kernel
             README.md          its design + differences from gfx950 v9
             note.md            the opt1-opt5 optimization log
  inter_wave/matmul_kernel.py   8-wave kernel
             README.md          its design + differences from gfx950
             images/            the pipeline diagram used by that README
  tools/layout_check.py         analytical LDS layout sweep (model is unreliable,
                                see 4.3 -- prefer lds_conflict.py)
  tools/lds_conflict.py         SQ_LDS_BANK_CONFLICT counter + swizzle sweep
  tools/run_att.py              rocprofv3 --att capture, verifies the ui_* dir
  tools/bench_prepared.py       prepared-launch rocprofv3 timing, multi-GPU sweep
```

`get_pids` (XCD remap + `GROUP_SIZE_M` swizzle) is reused unchanged from the
repository's shared [`kernels/gemm/utils/common.py`](../utils/common.py) — gfx942
and gfx950 have the same 8-XCD topology, so the L2-locality argument in
[`v9_beyond_hotloop`](../intra_wave/a16w16/v9_beyond_hotloop/README.md) carries
over as-is.
