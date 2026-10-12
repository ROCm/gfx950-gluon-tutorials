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
| §5–6 | the MFMA scheduler on gfx942, and where the files are |

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
git clone https://github.com/triton-lang/triton -b gfx950-tutorial-v3.1 /path/to/triton
pip uninstall -y triton
cd /path/to/triton && TRITON_EXT_ENABLED=1 pip install -e .
```

`gfx950-tutorial-v3.1` is the annotated tag these gfx942 kernels pin (the MI355X
kernels in the rest of the repository stay on `gfx950-tutorial-v3.0`): plain upstream
`main` at `b575c0c38`, the merge of
[triton-lang/triton#12229](https://github.com/triton-lang/triton/pull/12229), which
runs the MFMA scheduler on gfx942 (§5). It reports itself as Triton 3.9.0 and
declares Python >= 3.11 (on a 3.10 environment, add `--ignore-requires-python`: the
floor is metadata only, and nothing in Triton's Python needs 3.11). Neither kernel
loads a plugin, and `TRITON_EXT_ENABLED=1` is only for the gfx950 attention
kernels' plugin.

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

MI300X (gfx942), 304 CUs, ROCm 7.2, Triton `gfx950-tutorial-v3.1`.

**One methodology for every number below**, so the tables are comparable with
each other: M=4096, N=4864 with K swept (see §3.1 for why that shape family), on
**GPU 3** — the fastest of the node's eight, by median over three sweeps, spread
3.8%. TFLOPS come from the repository's plain timing path, as for the gfx950
tables: `bench.py --rocprof` runs 1000 dispatches over 512 MB of rotating buffers
under `rocprofv3 --kernel-trace` with `AMD_SERIALIZE_KERNEL=3`, and
[`scripts/calc_kernel_time.py`](../../../scripts/calc_kernel_time.py) averages the
kernel duration over every dispatch:

```bash
AMD_SERIALIZE_KERNEL=3 rocprofv3 --kernel-trace --output-format csv -d trace \
    -- python bench.py --rocprof -k inter_wave --K 8256 --dtype fp16
python ../../../scripts/calc_kernel_time.py trace/*/*_kernel_trace.csv a16w16_inter_wave_gfx942
```

`torch` is hipBLASLt through `torch.matmul` on the same operands (`-k torch`; its
kernel names start with `Cijk`). Each cell is the median of 3 rounds (5 at
K=8256), and a round runs all configurations of one shape back to back. Per-SIMD
MFMA efficiency is from ATT, decoded with the repo's
[`scripts/process_json.py`](../../../scripts/process_json.py), the median of 3
traces (7 at K=8256).

**fp16**

| K | intra_wave TF | eff | inter_wave TF | eff | torch TF |
|---|---|---|---|---|---|
| 2112 | 587.7 | 90.89% | **609.0** | 99.63% | 593.6 |
| 4160 | 655.6 | 90.88% | **679.5** | 99.72% | 609.3 |
| 8256 | **653.7** | 90.84% | 637.8 | 98.07% | 589.3 |
| 16448 | 654.8 | 90.86% | **662.6** | 99.78% | 608.6 |

**bf16**

| K | intra_wave TF | eff | inter_wave TF | eff | torch TF |
|---|---|---|---|---|---|
| 2112 | 605.8 | 90.89% | **636.7** | 99.63% | 617.9 |
| 4160 | 692.4 | 90.88% | **713.5** | 99.72% | 634.0 |
| 8256 | 694.1 | 90.87% | **701.2** | 98.63% | 641.0 |
| 16448 | 698.5 | 90.86% | **705.5** | 99.78% | 646.0 |

`eff` is in-loop **per-SIMD** MFMA efficiency, the comparable figure: the matrix
unit is a per-SIMD resource, so the 8-wave kernel's per-wave fraction doubles for
its two co-resident waves. At 98.1–99.8% the 8-wave loop is essentially at the
matrix-unit ceiling; the 4-wave one sits 7–9 points back, at 90.8–90.9%, and
[`intra_wave/README.md`](intra_wave/README.md) shows via ablation that almost all
of that gap is `ds_write` the MFMA stream cannot hide.

Efficiency moves with neither dtype nor K: fp16 and bf16 compile to the same loop,
and both kernels run the same cycles per iteration at every K — 4506–4509 for the
4-wave loop, 4105–4111 for the 8-wave one — except K=8256, where the 8-wave loop
runs 1–2% slower (4153–4177) and the traces vary most (§3.3).

**inter_wave wins 7 of 8 points** and beats hipBLASLt on all 8, by 2.1% to 12.4%;
it loses fp16 K=8256 (637.8 vs 653.7), the noisiest shape (§3.3). intra_wave beats
hipBLASLt on 6 of 8, by up to 10.8%, and loses at K=2112 (by 1.4% in fp16 and 2.4%
in bf16).

For scale, Triton's own `BlockPingpong` schedule at 4096×4864×8256
(`python/tutorials/03-matrix-multiplication.py`, tile 256×256×64, `num_warps=8`,
with the loop-variant K mask removed so the pass fires) reaches 4207.5 cyc/iter
and 97.35% per-SIMD at 560.5 TFLOPS, through the same timing path. It uses
`v_mfma_f32_32x32x8_f16`, which is more power-dense than the `16x16x16` used here
and runs at 1.10 GHz against inter_wave's 1.33 (fp16, K=8256) — so inter_wave is
ahead on cycles (4176.6), on efficiency, and by 13.8% on wall clock. See §3.5.

Cycles per iteration per wave behind the efficiency column (fp16; bf16 within 0.6%),
for reference:

| K | intra_wave | inter_wave |
|---|---|---|
| 2112 | 4506.5 | 4111.2 |
| 4160 | 4507.2 | 4107.6 |
| 8256 | 4509.1 | 4176.6 |
| 16448 | 4507.8 | 4104.9 |

Codegen: intra_wave 256 VGPR + 256 AGPR, 1 wave/SIMD, with 2 VGPRs spilled outside
the loop (12 B of scratch, §5); inter_wave 240 VGPR, 0 AGPR, 0 spills, **2
waves/SIMD**. Both use exactly 65536 B of LDS, and both measure
`SQ_LDS_BANK_CONFLICT` = **0**.

> [!NOTE]
> `bench.py` without `--rocprof` reports `do_bench` medians on the *default* device
> instead, so its numbers will not match these: a different GPU (up to 4% across
> the node) and a timer that includes the Python launch path rather than kernel
> duration alone. Use it for correctness checking and quick iteration; use the
> rocprofv3 path above for reporting.

### 3.1 Why these shapes

Both dimensions of the benchmark are chosen for gfx942 specifically. Same harness,
fp16, median of 3 rounds:

| M×N×K | intra_wave TF | inter_wave TF |
|---|---|---|
| 4096×4096×4096 | 598.7 | 572.7 |
| 4096×4864×4096 | 649.3 | 643.0 |
| 4096×4864×4160 | 656.4 | 677.4 |
| 4096×4096×8192 | 614.3 | 591.1 |
| 4096×4864×8192 | 654.5 | 639.9 |
| 4096×4864×8256 | 615.9 | 614.8 |

**N = 4864, not 4096.** With a 256x256 tile, 4096x4864 is a 16 x 19 = **304
workgroup** grid, exactly the MI300X CU count: one workgroup per CU and no tail
wave. 4096x4096 gives 256 workgroups, so 48 of 304 CUs sit idle for the whole
kernel. (LDS is full at 64 KB, so occupancy is 1 workgroup/CU and the grid must
match the machine exactly.) N = 4864 is worth +8.5% / +6.5% to intra_wave and
+12.3% / +8.3% to inter_wave (K = 4096 / 8192) — less than the 19% more work per
pass, because the 48 idle CUs leave power under the 750 W cap and the busy ones
clock higher: inter_wave runs at 1.48 GHz at 4096×4096×8192 and 1.34 GHz at
4096×4864×8192.

**K = 4160 or 8256, not 4096 or 8192.** A power-of-two K makes A's row stride a
power of two (K=4096 -> 8192 B), so every row of a tile lands in the same L1 set
and the loads hotspot. `4160 = 65 x 64` and `8256 = 129 x 64` are still multiples
of `BLOCK_K` but make the stride an *odd* multiple of the 128 B line, spreading
the rows across sets. The effect is clear only at K ≈ 4096: +5.3% for inter_wave
and +1.1% for intra_wave. At K ≈ 8192 it is inside K = 8256's noise (§3.3): these
rounds read 615-616 for both kernels at K = 8256, against 638-654 in §3's.

Together, inter_wave goes from 572.7 TFLOPS at 4096×4096×4096 to 677.4 at
4096×4864×4160 (+18.3%), and intra_wave from 598.7 to 656.4 (+9.6%).

### 3.2 GROUP_SIZE_M

At this grid each XCD gets 304/8 = 38 workgroups, so the tutorial's model
(minimise `GM + ceil(38/GM)`) predicts GM = 5, 6 or 8 at f = 13, with GM = 4 just
behind at 14 and GM = 1/2/16 far worse at 39/21/19. Measured with
`bench.py --sweep-gm`, median of 4 runs at 4096x4864x4160 fp16:

| GM | f(GM) | intra_wave | inter_wave |
|---|---|---|---|
| 1 | 39 | 648.6 | 671.8 |
| 2 | 21 | 648.8 | 672.7 |
| 4 | 14 | 657.7 | 681.6 |
| 5 | 13 | 655.5 | 678.2 |
| 6 | 13 | 657.5 | 683.3 |
| 8 | 13 | 656.5 | 681.6 |
| 16 | 19 | 650.1 | 673.7 |

The model separates the bad choices correctly — GM = 1, 2 and 16 cost 1.2-1.7% —
but GM = 4 through 8 are within 0.8% of each other. **`GROUP_SIZE_M = 4` is
kept**; anything in 4-8 is equivalent here. (These are `do_bench` numbers, which
read higher than §3's; see the note above.)

### 3.3 Reading the numbers

The timing path leaves a short gap between dispatches (`AMD_SERIALIZE_KERNEL=3` and
a Python launch per call), so how close the part runs to its 750 W cap depends on
how long each kernel is. During the GEMM, inter_wave (fp16) draws ~610 W at 1.48
GHz at K = 4160, ~705 W at 1.33 GHz at K = 8256, and the full 746 W at 1.16 GHz at
K = 16448, against a 2.1 GHz peak. Launching back to back with no gaps (a prepared
launch: arguments bound once, the launch stub entered directly) pins the part at
750 W and reads 1.9-10.1% lower, the most at short K; the ranking of the kernels
is the same either way. K = 8256 is the noisiest point: rounds there spread by up
to 5.7%, and an occasional ATT trace (2 of 28) lands 15-18% high, which is why
that row takes 5 rounds and 7 traces.

### 3.4 ATT traces

The repository's shared [`scripts/run_att.py`](../../../scripts/run_att.py) runs a command under
`rocprofv3 --att` and decodes the trace with
[`scripts/process_json.py`](../../../scripts/process_json.py), as for the gfx950 kernels. It reads
the trace configuration from an `att_matmul.json` in the working directory; this one matches
either gfx942 kernel:

```bash
bash ../../../scripts/install_att_decoder.sh       # once
cat > att_matmul.json <<'EOF'
{"jobs": [{"kernel_include_regex": "a16w16_.*_gfx942", "kernel_exclude_regex": "",
           "kernel_iteration_range": "[15]", "advanced_thread_trace": true,
           "att_target_cu": 0, "att_shader_engine_mask": "0xF", "att_simd_select": "0xF",
           "att_buffer_size": "0x60000000"}]}
EOF
python ../../../scripts/run_att.py --att-output att_inter \
    python bench.py -k inter_wave --K 8256 --dtype fp16 --no-torch
```

`process_json.py` reports in-loop MFMA efficiency per wave; for the 8-wave kernel the per-SIMD
figure is twice that, for its two co-resident waves.

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
| intra_wave, as first ported (`v1.1`) | 256 | 5877 | 69.7% | **69.7%** (1 wave/SIMD) |
| intra_wave, current | 256 | 4509.1 | 90.8% | **90.8%** |
| inter_wave, as first ported (`v1.1`) | 128 | 4770.6 | 42.9% | **85.9%** (2 waves/SIMD) |
| inter_wave, current | 128 | **4176.6** | 49.0% | **98.1%** |

Both kernels moved a long way from the initial port: intra_wave through the
opt1-opt5 sequence in [`intra_wave/README.md`](intra_wave/README.md#optimization-history) and then Triton's
MFMA scheduler (§5), inter_wave through the 8-region redesign in
[`inter_wave/README.md`](inter_wave/README.md). The LDS bank-conflict fix in §4.4
is common to both and is worth −16.2% loop cycles on the 4-wave kernel and −10.8%
on the 8-wave one.

### 3.5 Why TFLOPS understates the 8-wave kernel

The clearest case is K=16448, the longest kernel and the steadiest measurement: the
8-wave kernel runs **8.9% fewer cycles per K-step** (4104.9 vs 4507.8) at **8.9
points** higher per-SIMD MFMA efficiency, and is only **1.2% faster** — 662.6
against 654.8 TFLOPS in fp16, 705.5 against 698.5 in bf16. The 750 W cap is why.
During the run, fp16 K=16448:

| fp16, K=16448 | cycles/iter | per-SIMD eff | clock | power | TFLOPS (§3) |
|---|---|---|---|---|---|
| intra_wave | 4507.8 | 90.86% | 1.26 GHz | 746 W | 654.8 |
| inter_wave | **4104.9** | **99.78%** | 1.16 GHz | 746 W | 662.6 |

Both kernels sit on the cap, and the 8.9% cycle advantage is almost cancelled by an
8.0% clock deficit: **the better pipeline draws more power per cycle and gets
clocked down for it**, the same DIDT dynamic the tutorial's v7 describes.
`BlockPingpong`'s `32x32x8` MFMA is denser still, and runs lower still: 1.10 GHz at
K=8256 against inter_wave's 1.33.

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
access replays. Measured with three hardware counters, fp16 K=8256, per dispatch:

```bash
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE SQ_INSTS_LDS --output-format csv -d lds \
    -- python bench.py --rocprof -k inter_wave --K 8256 --dtype fp16
```

conflict ratio = `SQ_LDS_BANK_CONFLICT / (SQ_LDS_IDX_ACTIVE - SQ_LDS_BANK_CONFLICT)`
(replay cycles per useful cycle) and cycles per LDS instruction =
`SQ_LDS_IDX_ACTIVE / SQ_INSTS_LDS`. Both kernels take the layout from
`GFX942_SWZ_VEC` / `GFX942_SWZ_PER_PHASE` / `GFX942_SWZ_MAX_PHASE`, so
`GFX942_SWZ_PER_PHASE=2` measures `(8, 2, 8)`; give each layout its own
`TRITON_CACHE_DIR`, because the swizzle is not part of Triton's cache key.

| | conflict ratio | cyc / LDS instr | intra_wave `IDX_ACTIVE` | inter_wave `IDX_ACTIVE` |
|---|---|---|---|---|
| `(8, 2, 8)` | 0.667 / 0.750 | 13.33 / 14.00 | 1.004e8 | 1.406e8 |
| **`(8, 1, 8)`** | **0.000** | **8.00** | **6.02e7** | **8.03e7** |

8.00 cyc/instr is the floor (1024 B at 128 B/clk). In loop cycles (ATT, fp16
K=4160) the fix is worth **−16.2%** on the 4-wave kernel (5375.3 → 4507.2) and
**−10.8%** on the 8-wave one (4603.6 → 4107.6). The 4-wave figure was only −1.0%
under the `v1.1` plugin, because the plugin parked ~25 surplus MFMAs between each
region's last LDS read and its first LDS write, and that slack absorbed the replays.
Triton's MFMA scheduler leaves no such slack. Measured on a build with the parking
added back (fp16 K=4160, cycles per iteration):

| | in-tree schedule | + parking (`v1.1` opt 2) |
|---|---|---|
| `(8, 2, 8)` | 5379.3 | 4559.6 |
| **`(8, 1, 8)`** | **4507.3** | 4603.2 |

With the conflicting layout, parking recovers most of the loss; with the
conflict-free one it costs 2.1%. The right layout plus the plain schedule is the
fastest of the four, which is why the parking was not carried into Triton.

Getting the `local_store` side conflict-free also required **changing the
global-load layout** away from the tutorial's. The gfx950 layout has lanes 0–7
cover row `M0` and lanes 8–15 cover row `M0+16`; with a 128 B row stride those two
rows land on the same banks and *no swizzle can separate them*, because the
swizzle only permutes chunks within a row. Remapping so consecutive lanes walk
consecutive rows makes lanes 0–15 cover two adjacent 128 B rows = 256 B = all 64
banks exactly once, and improves the global side too (one instruction now reads
1024 contiguous bytes).

### 4.5 Where the two kernels diverge

Given §4.1–4.4, the remaining question is *who hides the barriers and the LDS
writes*, and that is the whole difference between the two kernels:

| | [`intra_wave`](intra_wave/) (4 waves, 1/SIMD) | [`inter_wave`](inter_wave/) (8 waves, 2/SIMD) |
|---|---|---|
| hides memory behind | MFMA in the **same** instruction stream | the **other wave group** |
| needs | Triton's MFMA scheduler to interleave (§5) | `warp_pipeline_stage` and a barrier per region |
| accumulators | AGPR — separate capacity at 1 wave/SIMD | VGPR — 2 waves split one 512-register file |
| LDS refill | per half-tile, one region after its last read | whole tensors, late in the body |
| ceiling | `ds_write` issue cost the MFMA stream cannot absorb | ~99.7% per-SIMD; barriers and write turnaround |

The 4-wave kernel is the more instructive read and the 8-wave one is the faster
kernel. Both READMEs assume this section.

## 5. The MFMA scheduler and accumulator pins on gfx942

The MFMA <-> memory interleave that the 4-wave kernel depends on is a pass in Triton
itself ([triton-lang/triton#12209](https://github.com/triton-lang/triton/pull/12209),
`v3.0`), enabled per launch with `schedule_hint="mfma-schedule"`, and since `v3.1`
it runs on gfx942 too
([triton-lang/triton#12229](https://github.com/triton-lang/triton/pull/12229)).
`intra_wave`'s `matmul()` passes the hint by default and the tools take
`--schedule-hint ""` to turn it off; the tutorial's `llirSched` plugin now carries
only the attention co-execution model and is not used here. Accumulators are pinned
to AGPRs in the kernel with `cd_regclass="a"`, which replaces `TRITON_FORCE_MFMA_AGPR`.

**What gfx942 needed from the pass.** Until `v3.1`, `make_llir` ran it only when
`options.arch == "gfx950"`, so on gfx942 the hint was a **silent no-op**: the hsaco
was byte-identical with and without it. The pass already knew the CDNA3 MFMA shapes
(`mfma.f32.16x16x16f16`, `16x16x16bf16.1k`); what it lacked was the LDS bandwidth. It
priced an LDS access at `Bits / 8` cycles, CDNA4's 256 B/clk; CDNA3 serves 128 B/clk,
so with all four SIMDs issuing it a `ds_read_b128` needs 32 cycles of MFMA cover, not
16. #12229 runs the pass on gfx942 and prices LDS by the target's bandwidth. On the
4-wave kernel (bf16; ATT cycles per K-iteration / per-SIMD in-loop MFMA efficiency):

| | K=2112 | K=4160 | K=8256 | K=16448 |
|---|---|---|---|---|
| scheduler off (`--schedule-hint ""`) | 5867.2 / 69.8% | 5870.9 / 69.8% | 5872.6 / 69.7% | 5873.2 / 69.7% |
| pass run on gfx942 with CDNA4's LDS cost | 5156.1 / 79.4% | 5159.1 / 79.4% | 5319.2 / 77.0% | 5159.8 / 79.4% |
| **`v3.1`: CDNA3's LDS cost** | **4506.4 / 90.9%** | **4507.3 / 90.9%** | **4507.6 / 90.9%** | **4507.8 / 90.9%** |

The middle row was measured on an ablation build while #12229 was prepared (one
trace per cell; read its K=8256 cell loosely, §3.3); the other two are `v3.1`,
and they reproduce the cycle counts of that build exactly. Over the unscheduled
build, the scheduler is worth +11.1% to +12.2% TFLOPS in fp16 and +12.8% to +14.1% in
bf16 (paired within each round of §3), and it runs about 1% fewer cycles per
iteration than the `v1.1` plugin did.

The plugin's other gfx942 change, parking surplus MFMAs between a region's last
`local_load` and first `local_store` (opt 2 in [`intra_wave/README.md`](intra_wave/README.md#optimization-history)),
is **not** in Triton: under the in-tree pass, which fences every memory op, it costs
2.1% more cycles per iteration at every K. It only paid while the LDS layout had
bank conflicts (§4.4).

The in-tree pass also fences each pinned MFMA, so it keeps the MFMAs in program
order: four-deep dependent chains on one accumulator, with an `s_nop 0` at each
chain switch (the plugin's loop round-robined accumulators instead). The nops sit
inside the MFMA shadow, and the cycle counts above already include them.

The pins cost a little spill outside the loop: 2 VGPRs (12 B of scratch) in the
scheduled build, 1 (8 B) unscheduled, stored in the prologue and reloaded in the
epilogue, with no scratch access in the loop. Adding `amdgpu-agpr-alloc=256` does
not change the binary.

The 8-wave kernel uses neither the scheduler nor AGPRs: its schedule is expressed
directly with `warp_pipeline_stage`, and its hot loop is instruction-for-instruction
the loop it had on `v1.1` (compared on the bf16 build). `amdgcnas` remains untried on gfx942.

## 6. Layout

```
kernels/gemm/gfx942/
  README.md                     this file
  bench.py                      correctness + TFLOPS for both kernels,
                                --sweep-gm, --rocprof (also -k torch)
  intra_wave/matmul_kernel.py   4-wave kernel
             README.md          its design + differences from gfx950 v9
  inter_wave/matmul_kernel.py   8-wave kernel
             README.md          its design + differences from gfx950
             images/            the pipeline diagram used by that README
```

`get_pids` (XCD remap + `GROUP_SIZE_M` swizzle) is reused unchanged from the
repository's shared [`kernels/gemm/utils/common.py`](../utils/common.py) — gfx942
and gfx950 have the same 8-XCD topology, so the L2-locality argument in
[`v9_beyond_hotloop`](../intra_wave/a16w16/v9_beyond_hotloop/README.md) carries
over as-is.
