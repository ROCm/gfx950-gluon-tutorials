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
grid of 128×128 quadrants, the `B_left → A_top → A_bot → B_right` region order,
and the XCD-aware PID remap with `GROUP_SIZE_M` swizzling — and change only what
CDNA3 forces them to change.

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

MI300X (gfx942), 304 CUs, ROCm 7.2, Triton `gfx950-tutorial-v1.1`, `do_bench`
median of 3 runs. `torch` is hipBLASLt via `torch.matmul`.

**fp16**

| M | N | K | intra_wave | inter_wave | torch | inter/torch |
|---|---|---|---|---|---|---|
| 4096 | 4864 | 2112 | 534 | 572 | 579 | 0.99 |
| 4096 | 4864 | 4160 | 567 | **622** | 620 | 1.00 |
| 4096 | 4864 | 8256 | 561 | **575** | 572 | 1.01 |
| 4096 | 4864 | 16448 | 575 | **611** | 586 | 1.04 |

**bf16**

| M | N | K | intra_wave | inter_wave | torch | inter/torch |
|---|---|---|---|---|---|---|
| 4096 | 4864 | 2112 | 565 | **609** | 595 | 1.02 |
| 4096 | 4864 | 4160 | 586 | **651** | 647 | 1.01 |
| 4096 | 4864 | 8256 | 567 | **609** | 602 | 1.01 |
| 4096 | 4864 | 16448 | 600 | **659** | 623 | 1.06 |

TFLOPS. The 8-wave kernel matches or beats hipBLASLt on every shape.

Codegen: intra_wave 256 VGPR + 256 AGPR, 0 spills, 1 wave/SIMD; inter_wave 224
VGPR, 0 AGPR, 0 spills, **2 waves/SIMD**. Both use exactly 65536 B of LDS.

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

Point ATTViewer at the `ui_output_agent_*_dispatch_*` directory it produces.
Two things the traces settle:

**Occupancy is visible in the file list.** The 4-wave kernel produces 16
wave-slot files (4 SE x 4 SIMD x `sl0`); the 8-wave kernel produces 32, adding
`sl1` -- the second resident wave group the warp pipeline needs.

**Per-SIMD MFMA efficiency**, decoded with the tutorial's `process_json.py`
(its `MFMA_CYCLE_MAP` already gives 16 cycles, correct for CDNA3's
`v_mfma_f32_16x16x16_f16`):

| | mfma/iter/wave | cycles/iter/wave | per-wave | **per-SIMD** |
|---|---|---|---|---|
| intra_wave | 256 | 5877 | 69.7% | **69.7%** (1 wave/SIMD) |
| inter_wave | 128 | 4737 | 43.2% | **86.5%** (2 waves/SIMD) |

At matched clock the whole dispatch is 856k cycles / 599 us for intra_wave vs
715k / 493 us for inter_wave -- **16.5% fewer cycles, 17.8% less wall time.**

### 3.5 The TFLOPS tables understate the 8-wave kernel

That 17.8% does not appear in the do_bench tables in §3, which show the two
kernels ~2% apart at K=8256. The 750 W cap is why. A first capture pair taken
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

### 4.1 No `buffer_load_to_shared` — the register round-trip

gfx950 streams HBM → LDS with `gl.amd.cdna4.async_copy.buffer_load_to_shared`
and synchronises with an async counter (`wait_group(n)`). CDNA3's Gluon surface
(`gl.amd.cdna3`) has `buffer_load`, `buffer_store` and `mfma` and no async copy,
so the path is

```
buffer_load  (HBM -> VGPR)  ->  local_store (VGPR -> LDS)  ->  local_load (LDS -> VGPR)  ->  mfma
```

Two consequences. It costs 32–64 VGPRs of staging that gfx950 does not pay. And
the LDS producer/consumer hazard is now closed by real `s_barrier`s inserted by
Triton's membar pass rather than by an async counter — which turns out to be the
single largest cost in the 4-wave kernel (§4.4).

### 4.2 Half the LDS — 64 KB is exactly one stage

gfx950 has 160 KB of LDS per CU and double-buffers a 256×256×64 stage (2 × 64 KB).
gfx942 has **64 KB**, i.e. exactly one stage. Double buffering at this tile size
is impossible.

The kernels do *not* respond by shrinking `BLOCK_K` to 32. Instead each of the
four half-tiles (`A_top`, `A_bot`, `B_left`, `B_right`) owns a single 16 KB slot
that is **refilled one region after its last read**:

```
region 0:  DOT C_tl | LR A_bot(k)    | LW B_left(k+1)  | GR B_left(k+2)
region 1:  DOT C_bl | LR B_right(k)  | LW A_top(k+1)   | GR A_top(k+2)
region 2:  DOT C_tr | LR B_left(k+1) | LW A_bot(k+1)   | GR A_bot(k+2)
region 3:  DOT C_br | LR A_top(k+1)  | LW B_right(k+1) | GR B_right(k+2)
```

Every slot's write lands strictly between its previous and next read, one region
apart on both sides, so one barrier per region is sufficient — and this buys the
same pipeline depth as gfx950's double buffer: `GR(k+2) → LW(k+1)` is a full
K-step (~4096 cycles) of HBM latency hiding, `LR(k+1) → DOT(k+1)` is one region
(~1024 cycles) of LDS latency hiding.

**Why not `BLOCK_K=32` + double buffering?** It fits (8 × 8 KB = 64 KB) and it
does reduce the barrier count to one per K-step, which is what you would expect
to want. It is measurably worse — 400 TFLOPS vs 476 — and the reason is global
coalescing. With `BLOCK_K=64` a row of the tile is 128 B, so eight lanes cover a
whole cache line and one `buffer_load_dwordx4` touches 8 back-to-back lines. At
`BLOCK_K=32` a row is 64 B, the same instruction touches 16 lines, and TCP
processing time per instruction roughly doubles. Ablating the `buffer_load`s out
of a `BLOCK_K=32` kernel moves it from 386 to 503 TFLOPS; ablating them out of
the `BLOCK_K=64` kernel changes nothing. Deepening the global staging to two
K-steps did not help either (v4, 400 TFLOPS) — it is the cache-line count, not
the latency.

The 8-wave kernel needs `BLOCK_K=32`-sized *operands* for register reasons, and
gets them without giving up the 128 B loads by keeping the LDS tile at
`BLOCK_K=64` and slicing only the **read** side (`smem.slice(0, 32, k_dim)`).

### 4.3 MFMA shape and LDS layout

CDNA3's widest 16×16 f16/bf16 intrinsic is `v_mfma_f32_16x16x16_f16`
(`AMDMFMALayout(version=3, instr_shape=[16,16,16])`), 16 cycles, versus CDNA4's
`..._16x16x32_f16`. `k_width=8` still yields `ds_read_b128`: the dot-operand K
tile is `8 × (64/16) = 32`, which Triton lowers to two K=16 MFMAs.

The tutorial's `PaddedSharedLayout([[512, 16]])` is not affordable — the four
slots already fill LDS to the byte — so both operands use
`SwizzledSharedLayout(8, 2, 8)` (`(8, 4, 4)` at `BLOCK_K=32`), which costs no
LDS and is bank-conflict free in **both** directions.

Getting the `local_store` side conflict-free also required changing the
global-load layout. The tutorial's layout has lanes 0-7 cover row `M0` and lanes
8-15 cover row `M0+16`; with a 128 B row stride those two rows land on the same
banks and no swizzle can separate them (the swizzle only permutes chunks *within*
a row), giving a fixed 2× conflict. Remapping so consecutive lanes walk
**consecutive rows** makes lanes 0-15 cover two adjacent 128 B rows = 256 B = all
64 banks exactly once, and improves the global side too (one instruction now
reads 1024 contiguous bytes).

`gl.bank_conflicts()` asserts on AMD shared layouts in this Triton build, so
[`tools/layout_check.py`](tools/layout_check.py) reconstructs the same analysis from
`gl.to_linear_layout()` plus the CDNA3 LDS model (64 banks × 4 B = 256 B/cycle,
`256/vec_bytes` lanes per phase). Running that file
sweeps the candidates and is what picked the layouts above.

### 4.4 Where the time actually goes (4-wave)

Ablating the hot loop of the 4-wave kernel (4096²×8192 fp16, the shape used while developing):

| in-loop ops | ms | vs MFMA-only |
|---|---|---|
| mfma only | 0.379 | 1.00× |
| + `local_load` | 0.372 | free |
| + `buffer_load` | 0.373 | free |
| + `local_store` | 0.417 | +10% |
| `local_load` + `local_store` | 0.536 | **+41%** |
| all three | 0.582 | +54% |

Neither memory op costs anything on its own. The +41% appears only when both are
present — i.e. it is entirely the **barriers** that the load/store pair forces
membar to insert, plus the `s_waitcnt lgkmcnt(0)` each one drags along. In the
generated assembly the backend clusters the MFMAs and lands two barriers ~6 MFMAs
apart, so an 8-deep `ds_read` burst has to drain with only ~96 cycles of compute
to hide it.

Two things that did *not* fix it: regrouping the stores to cut barriers from 3 to
2 per K-step (worse — 428 TFLOPS, the clustering costs more than the barrier),
and splitting into 8 shorter regions (worse — 438). What does fix it is the
8-wave kernel: when one wave group stalls on a barrier the other has MFMA work
queued, which is the entire point of warp-pipelining.

### 4.5 The 8-wave register wall

Two waves per SIMD split the unified 512-register file, so each wave gets **256
registers, VGPR and AGPR together** — AGPRs buy no capacity at 2 waves/SIMD,
which is why the 8-wave kernel runs with `amdgpu-agpr-alloc=0,0` while the 4-wave
one reserves 256 AGPRs. A 256×256×64 stage at 8 waves needs

```
accumulators   4 × [128×128] f32 / (8 × 64)  = 128
dot operands   4 live half-tiles             =  96
global staging 4 half-tiles                  =  32
                                               ---
                                               256   before addressing
```

which compiles to 256 VGPRs **with 28 spill slots** and 279 TFLOPS. Reading the
operands as K=32 slices of the same `BLOCK_K=64` LDS tile halves the operand
term to 48, lands at 224 VGPR / 0 spills / 2 waves per SIMD, and takes the kernel
to 510 TFLOPS. It also doubles the region count to 8, which puts a cluster
boundary every ~512 cycles — the same barrier cadence as the gfx950 8-wave
kernel, where 4 regions of a 2×-faster MFMA give the same interval.

## 5. On the tutorial's out-of-tree plugins

The tutorial's `llirSched` plugin, `TRITON_FORCE_MFMA_AGPR`, and the `amdgcnas`
peephole were all tried here. `TRITON_FORCE_MFMA_AGPR` is **required** by the
4-wave kernel (it is what supplies `amdgpu-agpr-alloc=256`; `bench.py` sets it).
`libLlirSched.so` loads and runs on gfx942 but changes nothing measurable
(467→472 TFLOPS at 4096²×8192, within noise) — unsurprising, since its throughput model is
built around CDNA4's MFMA and async-copy anchors, and it has no `local_store`
anchor class tuned for this pipeline. Closing the remaining gap to hipBLASLt
would most likely mean an equivalent scheduling pass taught about the
`buffer_load → local_store → local_load` chain.

## 6. Layout

```
kernels/gemm/gfx942/
  README.md                     this file
  bench.py                      correctness + TFLOPS for both kernels,
                                --sweep-gm, --rocprof
  intra_wave/matmul_kernel.py   4-wave kernel  (+ README.md)
  inter_wave/matmul_kernel.py   8-wave kernel  (+ README.md)
  tools/layout_check.py         CDNA3 LDS bank-conflict model + layout sweep
  tools/run_att.py              rocprofv3 --att capture, verifies the ui_* dir
```

`get_pids` (XCD remap + `GROUP_SIZE_M` swizzle) is reused unchanged from the
repository's shared [`kernels/gemm/utils/common.py`](../utils/common.py) — gfx942
and gfx950 have the same 8-XCD topology, so the L2-locality argument in
[`v9_beyond_hotloop`](../intra_wave/a16w16/v9_beyond_hotloop/README.md) carries
over as-is.
