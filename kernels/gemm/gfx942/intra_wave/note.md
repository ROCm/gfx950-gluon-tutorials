# intra_wave optimization log — gfx942 / MI300X

Working notes for optimizing the 4-wave kernel. Each section is one step: what
was measured, what it means, what to do next.

---

## 0. Baseline (as committed, no LLIR scheduler)

MI300X (gfx942), 304 CUs, ROCm 7.2, Triton `gfx950-tutorial-v1.1`,
`TRITON_FORCE_MFMA_AGPR=1`, `bench.py -k intra_wave --reps 3`.

### Throughput

| M | N | K | fp16 | bf16 | torch fp16 | fp16 / torch |
|---|---|---|---|---|---|---|
| 4096 | 4864 | 2112 | 534 | 562 | 574 | 0.93 |
| 4096 | 4864 | 4160 | 572 | 596 | 624 | 0.92 |
| 4096 | 4864 | 8256 | 529 | 595 | 572 | 0.93 |
| 4096 | 4864 | 16448 | 580 | 605 | 588 | 0.99 |

TFLOPS, `do_bench` median of 3, GPU at 1980–2075 MHz. K=8256 is the noisiest
point (±5%). Read these against §3.3 of the [gfx942 README](../README.md): the
750 W cap means absolute TFLOPS move with thermal history, so the cycle-level
numbers below are the stable ones.

### Codegen

| | |
|---|---|
| VGPR | 256 (arch) |
| AGPR | 256 (accumulators, `amdgpu-agpr-alloc=256`) |
| VGPR spills | **0** |
| SGPR spills | 0 |
| LDS | 65536 B (100% of the CU) |
| occupancy | 1 wave/SIMD (LDS-bound) |

Hot loop, per K-step per wave (`.LBB0_1`, 424 instructions):

| op | count |
|---|---|
| `v_mfma_f32_16x16x16_f16` | 256 |
| `ds_read_b128` | 32 |
| `ds_write_b128` | 16 |
| `buffer_load_dwordx4` | 16 |
| `s_barrier` | 3 |
| `s_waitcnt` | 6 |
| `v_accvgpr` copies | 0 |

### In-loop MFMA efficiency — 69.7%

From the ATT capture at 4096x4864x8256 fp16 (`tools/run_att.py intra_wave`),
decoded with `scripts/process_json.py`:

| | |
|---|---|
| MFMA per iteration per wave | 256 |
| ideal cycles per iteration | 256 x 16 = 4096 |
| **measured cycles per iteration** | **5877** |
| **in-loop MFMA efficiency** | **69.7%** |
| loop / prologue / epilogue split | 95.2% / 1.9% / 2.9% |
| iterations (K=8256) | 127 |

Two independent confirmations from the same trace, reading per-instruction issue
clocks out of `se0_sm0_sl0_wv0.json`:

* `v_mfma_f32_16x16x16_f16` **median issue gap = 16.0 cycles** — the matrix pipe
  really is 16 cycles/MFMA on gfx942 for this shape, so the 4096-cycle ideal is
  right. (Cross-check: 1307 TFLOPS / (304 CU x 2.1 GHz) = 2048 flop/clk/CU =
  512/SIMD; a 16x16x16 MFMA is 8192 flop; 8192/512 = 16.)
* Same opcode's **mean gap = 23.0 cycles**. 16/23 = 69.6%, i.e. the efficiency
  number falls straight out of the issue-gap distribution: MFMAs issue
  back-to-back at 16 when they can, and the deficit is entirely stall.

So **~30% of the matrix pipe is idle**, and it is idle in bursts, not uniformly.

### Where the 30% goes

Ablating the hot loop (4096²x8192 fp16, the shape used during development):

| in-loop ops | ms | vs MFMA-only |
|---|---|---|
| mfma only | 0.379 | 1.00x |
| + `local_load` | 0.372 | free |
| + `buffer_load` | 0.373 | free |
| + `local_store` | 0.417 | +10% |
| `local_load` + `local_store` | 0.536 | **+41%** |
| all three | 0.582 | +54% |

Neither memory op costs anything alone. The cost appears only when both are
present — it is the **barriers** membar must insert for the write-after-read /
read-after-write on each LDS slot, plus the `s_waitcnt lgkmcnt(0)` each drags
along. In the emitted assembly the backend clusters the MFMAs and lands two of
the three barriers only ~6 MFMAs apart, so an 8-deep `ds_read` burst has to
drain with ~96 cycles of compute to hide it.

The instruction-order skeleton of the loop makes it obvious (M=mfma, R=ds_read,
W=ds_write, G=buffer_load, B=barrier, w=s_waitcnt):

```
w B R4 M1 R4 M3 w B M1 R3 M1 w W1 w W2 w W1 G2 M1 G2 W4 M1 G4 M23 R1 M1 R1 M46
R1 M1 R1 M1 R1 w B W4 G3 W4 G5 M72 R3 M7 R3 M65 R3 M12 R3 M10 R3 M2 R1 M8
```

Memory ops arrive in clumps (`R4`, `W4`, `G5`) separated by long MFMA runs
(`M72`, `M65`, `M46`), instead of being spread through the compute. Two things
already ruled out as fixes: regrouping the stores to cut barriers 3→2 is *worse*
(428 vs 476 TFLOPS — the resulting 16-store/16-read clumping costs more than the
barrier saved), and splitting into 8 shorter regions is also worse (438).

**Conclusion: this is a scheduling problem, not a bandwidth wall.** §2 below
shows the memory ops demand only 44–63% of the MFMA stream as cover, so there is
enough compute to hide all of it if it is interleaved.

---

## 1. Optimization 1 — enable the LLIR scheduler

### 1.1 Status: it is currently a silent no-op on gfx942

Loading the plugin changes nothing (467 → 472 TFLOPS at 4096²x8192, inside
noise). The reason is not subtle:

```
$ LLVM_PASS_PLUGIN_PATH=.../libLlirSched.so LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 ...
sched.barrier count in LLIR: 0          # <- with the plugin loaded
mfma intrinsic name: ['mfma.f32.16x16x16f16']
```

Zero `sched.barrier` are emitted, so the pass ran and bailed. Tracing it:

`Utils::getMFMACycles()` matches the intrinsic name against a fixed table
(`LlirSchedPlugin.cpp:203-213`) that contains **only CDNA4 shapes**:

```
mfma.f32.16x16x32.f16  16    mfma.f32.32x32x16.f16   32
mfma.f32.16x16x32.bf16 16    mfma.f32.32x32x16.bf16  32
mfma.i32.16x16x64.i8   16    mfma.i32.32x32x32.i8    32
```

gfx942 emits `mfma.f32.16x16x16f16`, which matches nothing, so `getMFMACycles`
returns 0, and the schedulability check at `LlirSchedPlugin.cpp:728` —

```cpp
if (MFMACycles == 0 || !HasAnchor)
  continue;                       // leave the region to LLVM's schedulers
```

— skips every region. **The pass is doing exactly what it was written to do:
refuse to schedule a shape it does not model.**

> [!IMPORTANT]
> Note the naming trap. CDNA4 intrinsics are `mfma.f32.16x16x32.f16` (dot before
> the type); the CDNA3 ones are `mfma.f32.16x16x16f16` — **no dot**. And bf16 on
> CDNA3 is `mfma.f32.16x16x16bf16.1k`, with the `.1k` suffix. A table entry
> written by analogy with the CDNA4 rows will silently fail to match, and the
> failure mode is *no error and no speedup*. Verified against
> `IntrinsicsAMDGPU.h` in the pinned LLVM (`850a2b1b`).

### 1.2 Everything else in the pass is already architecture-neutral

Worth stating explicitly, because it means the port is small:

* **Instruction classification** (`classifySchedInst`, :133) keys off
  `buffer.load` / `ds.read` / `ds.load` intrinsic names and LDS address space 3
  for plain load/store. Our kernel's `buffer_load` → GR, `local_load` → LR,
  `local_store` → LW all classify correctly with no change.
* **Region formation** (`analyzeBB`, :321) opens a new region at every MFMA that
  follows a memory op. Our source order per quadrant is
  `[64 MFMA][8 LR][4 LW][4 GR]`, so it forms exactly 4 regions per K-step, each
  `64 MFMA + 16 anchors`. This is the intended shape.
* **Dependency safety** holds by construction: region *k*'s MFMAs consume
  operands read in region *k-1* (that is the kernel's LR-one-region-ahead
  schedule), so reordering MFMAs among region *k*'s anchors cannot move an MFMA
  above its producer.
* **Barriers are not region boundaries** — `s_barrier` classifies as `Other`.
  MFMAs can therefore be placed on both sides of a barrier, which is precisely
  what is needed to cover the `s_waitcnt lgkmcnt(0)` that the barrier drags in.
  This is the mechanism by which the pass should attack the §0 problem.
* **`sched.barrier(0)` pinning** (:685) stops the machine scheduler from
  re-clustering the MFMAs afterwards — without it the interleave would be undone
  later in codegen, which is the failure mode §0 is showing today.

### 1.3 Plan (as written before implementing; kept for the record)

**Step 1 — teach `getMFMACycles` the CDNA3 shapes.** Add to `kFixedCycles`:

| intrinsic (LLVM name) | shape | flop | cycles |
|---|---|---|---|
| `mfma.f32.16x16x16f16` | 16x16x16 fp16 | 8192 | **16** |
| `mfma.f32.16x16x16bf16.1k` | 16x16x16 bf16 | 8192 | **16** |
| `mfma.f32.32x32x8f16` | 32x32x8 fp16 | 16384 | **32** |
| `mfma.f32.32x32x8bf16.1k` | 32x32x8 bf16 | 16384 | **32** |

Cycles = flop / 512 flop-per-clk-per-SIMD, the same derivation the CDNA4 rows
use. The 16-cycle figure for the shape we actually emit is **measured**, not
assumed (§0: median issue gap 16.0).

Substring-matching check (the table uses `Name.contains`): `16x16x16f16` does
not match `16x16x16bf16.1k` or vice versa (the `b` intervenes), and neither
collides with the existing CDNA4 rows or with the fp8 shapes
`mfma.f32.16x16x32.fp8.fp8`. Safe to add.

This alone should make the pass fire. It is a ~4-line change and is the whole of
optimization 1's first cut.

**Step 2 — check the GR cover constant.** `mfmaPerGR = ceil(64 / mfmaCycles)`
(:576). The 64 comes from the gfx950 TCP model: 16 cycles of TCP processing per
`buffer_load_dwordx4` x 4 waves sharing the TCP. gfx942 intra_wave also runs
**4 waves/CU** through the same 32 KB TCP, so 64 should carry over unchanged and
`mfmaPerGR = 4`. No change expected; worth confirming against a counter run
rather than assuming.

(For an 8-wave kernel the constant would be 8 x 16 = 128 — noted only so the
constant is not silently reused out of context. The inter_wave kernel is
warp-pipelined and is not a target for this pass.)

**Step 3 — settle the LDS cover constant. This is the one real unknown.**

`getLDSCoverCycles` (:235) returns `Bits / 8`, so a `ds_read_b128` demands 16
cycles of cover. That number is the gfx950 LDS: 64 banks x 4 B = 256 B/clk/CU,
so four SIMDs each issuing a b128 (4 x 1024 B) take 4096/256 = 16 cycles, i.e.
one b128 per SIMD per 16 cycles.

**If gfx942's LDS is 128 B/clk/CU instead, the correct divisor is `Bits / 4`
(32 cycles per b128) and the current constant under-spaces every LDS access by
2x.** I could not settle this from the existing ATT trace: this kernel runs LDS
at only ~20% utilization (192 KB per K-step per CU over 5877 cycles ≈ 33 B/clk),
so the port never saturates and the trace shows queueing behaviour, not the
peak — the measured `ds_read_b128` median issue gap of 12 cycles is the SP-to-LDS
FIFO absorbing an 8-deep burst (`docs/lds_throughput.md` §1.6), not the
steady-state rate.

Settle it with a saturation microbenchmark, mirroring
`experiments/lds_throughput_validation/`: a Gluon kernel issuing back-to-back
`ds_read_b128` with **no** MFMA and a conflict-free swizzle, 4 waves, one
workgroup per CU, and measure cycles/instruction from ATT. Take the slope
between two variants with different read counts so loop and VALU overhead
cancel. Expected answer is 16 or 32; either way it is a one-line change and it
should be parameterized per ISA family rather than hardcoded.

**Step 4 — what the model predicts for this kernel.** Per region (64 MFMA;
anchors 8 LR, 4 LW, 4 GR; `mfmaCycles=16`, `mfmaPerGR=4`, no GR→LR adjacency):

| | LDS = `Bits/8` (16 cyc/b128) | LDS = `Bits/4` (32 cyc/b128) |
|---|---|---|
| `grBudget` | 4 x 4 = 16 | 16 |
| `ldsBudget` | 12 x 16 / 16 = 12 | 12 x 32 / 16 = 24 |
| `needed` (+2 tail) | 30 | 42 |
| leftover of 64 | 34 → 17 head / 17 tail | 22 → 11 head / 11 tail |

Both fit inside the region's 64 MFMAs with room to spare. Across the K-step that
is 4 x 28 = 112 or 4 x 40 = 160 MFMAs of cover demanded out of 256 available —
**44% to 63%**. The throughput model therefore says the loop has enough compute
to hide all of its memory traffic, which is the quantitative version of §0's
conclusion. If the pass fires and the model is right, in-loop MFMA efficiency
should move from 69.7% toward the 90s.

**Step 5 — validate.** In order, stopping at whatever breaks:

1. `sched.barrier` count in the LLIR is non-zero (the §1.1 check — this is the
   go/no-go).
2. `LLVM_DEBUG` region dump shows 4 regions/K-step at 64 MFMA each, and the
   MFMA-insertion summary matches the Step 4 table.
3. Loop instruction mix is unchanged (256/32/16/16) — the pass must reorder, not
   duplicate or drop.
4. Correctness: `bench.py -k intra_wave` all shapes, fp16 + bf16.
5. VGPR/AGPR/spills unchanged — interleaving lengthens live ranges, and this
   kernel has **zero** headroom (256 VGPR + 256 AGPR, exactly full). Spills here
   would erase the win. This is the most likely way Step 1 backfires.
6. ATT re-capture: in-loop MFMA efficiency and the `v_mfma` mean issue gap. The
   gap should fall from 23.0 toward 16.
7. TFLOPS last, at matched clock, medians — it is the noisiest signal, not the
   diagnostic.

### 1.4 Result — implemented, +5-8% TFLOPS, MFMA efficiency 69.7% -> 76.9%

Implemented in `plugins/llir_scheduler/LlirSchedPlugin.cpp` and rebuilt
(`libLlirSched.so`, LLVM `850a2b1b`). Two changes:

1. **Instruction table** — added the CDNA3 shapes with an explicit derivation
   (`cycles = flop / flop_per_clk_per_SIMD`), and a comment about the naming
   trap. The f16/bf16 rows are measured; the fp8/bf8/i8 rows are derived from
   the same rule and marked as not yet measured.
2. **Throughput model** — `getLDSCoverCycles` no longer hardcodes `Bits / 8`.
   It is now `Bits * 32 / LDSBytesPerClk`, where `32 = 64 lanes x 4 SIMDs / 8`
   and `LDSBytesPerClk` is **128 on CDNA3, 256 on CDNA4**. A `ds_read_b128`
   therefore costs 16 cycles of cover on gfx950 and **32 on gfx942**.

   For CDNA4 the new expression is *arithmetically identical* to the old one
   (`Bits * 32 / 256 == Bits / 8` for every width), so gfx950 scheduling is
   bit-for-bit unchanged. That was the design constraint.

The ISA family is derived from the MFMA opcode, because at `OptimizerLast`
Triton has set no `target-cpu` on the function and the module triple is a bare
`amdgcn-amd-amdhsa` — the opcode is the only arch signal available. Triton's
`MfmaGroup` picks a distinct shape per family for every dtype, so this is
unambiguous in practice; `LLIR_SCHED_LDS_BYTES_PER_CLK` overrides it.

**Go/no-go:** `sched.barrier` in the emitted LLIR went **0 -> 65**, for fp16
(`mfma.f32.16x16x16f16`) and bf16 (`mfma.f32.16x16x16bf16.1k`) alike.

**The schedule now matches the model.** Loop skeleton before and after
(M=mfma, R=ds_read, W=ds_write, G=buffer_load, B=barrier, w=s_waitcnt):

```
before:  w B R4 M1 R4 M3 w B M1 R3 M1 w W1 w W2 w W1 G2 M1 G2 W4 M1 G4 M23 R1 M1
         R1 M46 R1 M1 R1 M1 R1 w B W4 G3 W4 G5 M72 R3 M7 R3 M65 R3 M12 ...

after:   w M1 B1 M1 R1 M10 R1 M2 R1 M2 R1 M2 R1 M2 R1 M2 R1 M2 R1 M2 w B M1 w W1
         M1 w W1 M2 W1 M2 w W1 M1 G1 M2 G1 M4 G1 M4 G1 M5 R1 M28 R1 M2 R1 M2 ...
```

Each `ds_read` now draws **2 MFMAs** (32 cycles of LDS occupancy / 16 per MFMA)
and each `buffer_load` **4** (64 / 16) — exactly the constants above, so the
emitted spacing is direct evidence the cost model is being applied. The clumps
(`R4`, `W4`, `G5`, `M72`) are gone.

**Codegen unchanged** — 256 VGPR + 256 AGPR, **0 spills**, 65536 B LDS, and the
loop instruction mix is still exactly 256/32/16/16, i.e. a pure reorder. The
register-pressure risk in §1.5 did not materialise.

**Throughput** (median of 3, `--reps 3`):

| K | fp16 base | fp16 +llir | | bf16 base | bf16 +llir | | torch fp16 |
|---|---|---|---|---|---|---|---|
| 2112 | 532 | **561** | +5.5% | 564 | **595** | +5.4% | 583 |
| 4160 | 572 | **602** | +5.2% | 599 | **625** | +4.5% | 624 |
| 8256 | 528 | **573** | +8.4% | 551 | **591** | +7.1% | 570 |
| 16448 | 579 | **605** | +4.5% | 604 | **633** | +4.7% | 588 |

With the scheduler the 4-wave kernel now **beats hipBLASLt at K=8256 and
K=16448** in both dtypes, where before it trailed at every shape.

**ATT re-measurement** (4096x4864x8256 fp16, same method as §0):

| | baseline | +llir |
|---|---|---|
| cycles / iteration / wave | 5877 | **5327** (-9.4%) |
| in-loop MFMA efficiency | 69.7% | **76.9%** |
| `v_mfma` mean issue gap | 23.0 | **20.9** |
| `v_mfma` median issue gap | 16.0 | 16.0 |

The three move consistently (16/20.9 = 76.6% ≈ 76.9%), which is the same
cross-check as §0.

**Still 23% idle.** The predicted budget (§1.3 step 4, `Bits/4` column) said the
memory ops need only 63% of the MFMA stream as cover, so a well-scheduled loop
should get much closer to 90%. The remainder is most likely the three barriers:
the pass spaces memory ops against compute, but it does not move work across the
`s_waitcnt lgkmcnt(0)` that each `ttg.barrier` drags in, and the skeleton above
still shows `w B` pairs with only 1-2 MFMAs adjacent. Next steps to test, in
order of expected value:

1. `amdgcnas` post-assembly peephole on top (it interleaves MFMA with the scalar
   ops that `llirSched` structurally cannot reach).
2. Look at whether the region head/tail leftover split (`M10`, `M28` runs)
   should instead be distributed — those are ~40 MFMAs per K-step sitting in
   clumps.
3. The barrier itself: a finer `lgkmcnt(N)` instead of `lgkmcnt(0)` would let
   LDS traffic stay in flight across it. That is a Triton membar change.

### 1.5 Risks

* **Register pressure (was the main risk; did not materialise).** The kernel is
  at exactly 256 VGPR + 256 AGPR with 0 spills, and stayed there. Interleaving MFMAs with loads extends live ranges; if the
  pass causes any spill the win is gone. Mitigation if it happens: the pass only
  reorders within a region, so reducing `headLeftover`/`tailLeftover` (keeping
  more MFMAs clustered at the region head) trades interleave quality for
  pressure.
* **The `.so` is ABI-locked** to LLVM `850a2b1b`. Rebuilding after editing the
  source needs the same LLVM the Triton pin uses — `plugins/llir_scheduler/README.md`
  has the `g++` line. Cannot be edited and shipped as Python.
* **`amdgcnas` is a separate lever** and untested here; the peephole is
  post-assembly and independent of this change.

---

## 2. Backlog (not yet attempted)

* Close the remaining 23% — see the three candidates at the end of §1.4.
* `amdgcnas` post-assembly peephole on gfx942.
* Measure the fp8/bf8/i8 CDNA3 cycle counts added to the scheduler's table on
  faith (derived, not measured). Needed before porting a8w8 to gfx942.
* Epilogue: currently stores from the mfma layout (`buffer_store_dwordx2`).
  Converting to a blocked layout would give `dwordx4` but needs LDS scratch,
  and LDS is full — would require freeing a slot first.
* The `s_waitcnt lgkmcnt(0)` that each membar barrier drags in is coarser than
  needed; a finer `lgkmcnt(N)` would let some LDS traffic stay in flight across
  the barrier. This is a Triton membar/AMD-backend change, not a kernel change.
