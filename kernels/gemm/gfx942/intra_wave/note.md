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

**Conclusion: this is a scheduling problem, not a bandwidth wall.** The
throughput model says the memory ops demand only ~63% of the MFMA stream as
cover (see Opt 1 §1.4), so there is enough compute to hide all of it if it is
interleaved. Opt 1 acts on exactly that.

---

## Opt 1 — teach the LLIR scheduler CDNA3  ✅ done

**in-loop MFMA efficiency 69.7% → 76.7%, +5–8% TFLOPS, 0 spills.**

### 1.1 The problem: the pass was a silent no-op on gfx942

Loading the plugin changed nothing (467 → 472 TFLOPS at 4096²x8192, inside
noise), because it never ran on a single region. The check:

```
$ LLVM_PASS_PLUGIN_PATH=.../libLlirSched.so LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 ...
sched.barrier count in LLIR: 0
mfma intrinsic name: ['mfma.f32.16x16x16f16']
```

`Utils::getMFMACycles()` matched intrinsic names against a table holding **only
CDNA4 shapes**, so gfx942's `mfma.f32.16x16x16f16` matched nothing, the function
returned 0, and the schedulability check dropped every region:

```cpp
if (MFMACycles == 0 || !HasAnchor)
  continue;                       // leave the region to LLVM's schedulers
```

The pass was doing exactly what it was written to do — refuse to schedule a
shape it does not model. Everything *else* in it was already
architecture-neutral, which is why the fix is small: instruction classification
(`buffer.load` / `ds.read` / LDS addrspace 3), region formation (a new region at
every MFMA following a memory op → 4 regions/K-step of `64 MFMA + 16 anchors`
here), dependency safety (region *k*'s MFMAs consume operands read in *k-1*),
and — critically — `s_barrier` is **not** a region boundary, so MFMAs can be
placed on both sides of one.

> [!IMPORTANT]
> Naming trap. CDNA4 spells the type with a separating dot
> (`mfma.f32.16x16x32.f16`); the CDNA3 legacy ops do not
> (`mfma.f32.16x16x16f16`), and CDNA3 bf16 carries a `.1k` suffix. A CDNA3 row
> written by analogy with a CDNA4 row silently fails to match, and the failure
> mode is *no error and no speedup*. Names checked against `IntrinsicsAMDGPU.h`
> in the pinned LLVM (`850a2b1b`).

### 1.2 What was extended

Two changes in `plugins/llir_scheduler/LlirSchedPlugin.cpp` (`.so` rebuilt).

**1. Instruction list** — added the CDNA3 shapes.
`cycles = flop / flop_per_clk_per_SIMD` (512 for f16/bf16, 1024 for fp8/bf8/i8),
the same rule the CDNA4 rows follow:

| intrinsic | cycles | basis |
|---|---|---|
| `mfma.f32.16x16x16f16` | 16 | **measured** (ATT median issue gap 16.0) |
| `mfma.f32.16x16x16bf16.1k` | 16 | measured |
| `mfma.f32.32x32x8f16` / `...8bf16.1k` | 32 | derived |
| `mfma.f32.{16x16x32,32x32x16}.{fp8,bf8}` | 16 / 32 | derived, **not measured** |
| `mfma.i32.{16x16x32,32x32x16}.i8` | 16 / 32 | derived, **not measured** |

**2. Throughput model** — `getLDSCoverCycles` hardcoded `Bits / 8`, which is the
CDNA4 LDS. It is now bandwidth-derived:

```
cover_cycles   = Bits * 32 / LDSBytesPerClk      // 32 = 64 lanes * 4 SIMDs / 8
LDSBytesPerClk = 128 (CDNA3)  |  256 (CDNA4)
```

so a `ds_read_b128` costs **32 cycles** of MFMA cover on gfx942 versus 16 on
gfx950 — the old constant under-spaced every LDS access by 2x. **gfx950 is
untouched**: `Bits * 32 / 256 == Bits / 8` exactly, for every access width that
occurs. That identity was the design constraint.

The ISA family is read off the MFMA opcode, because at `OptimizerLast` Triton
has set no `target-cpu` on the function and the module triple is a bare
`amdgcn-amd-amdhsa` — the opcode is the only arch signal available.
`LLIR_SCHED_LDS_BYTES_PER_CLK` overrides it.

### 1.3 Result

`sched.barrier` in the emitted LLIR: **0 → 65**, for fp16 and bf16 alike.

The loop is now interleaved instead of clumped — same instruction counts (256
mfma / 32 ds_read / 16 ds_write / 16 buffer_load / 3 barriers), i.e. a pure
reorder. From the traced code objects (M=mfma, R=ds_read, W=ds_write,
G=buffer_load, B=barrier):

```
before:  B1 R4 M1 R4 M3 B1 M1 R3 M1 W4 G2 M1 G2 W4 M1 G4 M23 ... M46 ... M72 ... M65
after:   M1 B1 M1 R1 M10 R1 M2 R1 M2 R1 M2 ... W1 M2 W1 M2 ... G1 M4 G1 M4 G1 M5
```

Each `ds_read` draws **2** MFMAs (32 cycles LDS / 16 per MFMA) and each
`buffer_load` **4** (64 / 16) — the cost model visible in the ISA.

**ATT** — 4096x4864x8256 fp16, published traces in
`/data/att_gfx942_intra_wave_4096x4864x8256_fp16{,_llirsched}/`:

| | baseline | +llirsched |
|---|---|---|
| cycles / iteration / wave | 5877 | **5343** (−9.1%) |
| in-loop MFMA efficiency | 69.7% | **76.7%** |
| `v_mfma` mean issue gap | 23.0 | **20.9** |
| `v_mfma` median issue gap | 16.0 | 16.0 |
| whole dispatch | 856,144 cyc / 599.3 µs @ 1.43 GHz | 785,656 cyc / 580.7 µs @ 1.35 GHz |

Cycles are the comparable figure (−8.2% for the dispatch); wall time differs by
only 3% because the two captures clocked differently. A second capture agreed
within noise (5327 cyc/iter, 76.9%).

**Throughput** — `bench.py -k intra_wave --reps 3`, TFLOPS:

| K | fp16 base | fp16 +llir | Δ | bf16 base | bf16 +llir | Δ | torch fp16 |
|---|---|---|---|---|---|---|---|
| 2112 | 532 | **561** | +5.5% | 564 | **595** | +5.4% | 583 |
| 4160 | 572 | **602** | +5.2% | 599 | **625** | +4.5% | 624 |
| 8256 | 528 | **573** | +8.4% | 551 | **591** | +7.1% | 570 |
| 16448 | 579 | **605** | +4.5% | 604 | **633** | +4.7% | 588 |

The kernel now **beats hipBLASLt at K=8256 and K=16448 in both dtypes**, where
before it trailed at every shape. (torch was re-measured in the same loop for
the +llir runs and moved <1.5%.)

**Codegen unchanged** — 256 VGPR + 256 AGPR, **0 spills**, 65536 B LDS. Register
pressure was the main risk of interleaving into a register file that is exactly
full; it did not materialise.

### 1.4 What is still on the table

23% of the matrix pipe is still idle. The throughput model says the memory ops
need only 63% of the MFMA stream as cover (per region: `grBudget` 4x4=16 +
`ldsBudget` 12x32/16=24 + 2 tail = 42 of 64 MFMAs), so a perfectly scheduled
loop should reach the 90s. Remaining suspects, in order of expected value:

1. **The barriers.** The pass spaces memory against compute but does not move
   work across the `s_waitcnt lgkmcnt(0)` each `ttg.barrier` drags in; the
   skeleton above still shows `B1` with only 1–2 adjacent MFMAs.
2. **The head/tail leftover split.** ~40 MFMAs per K-step still sit in clumps
   (`M10`, `M28`, `M21`) at region heads rather than being distributed.
3. **`amdgcnas`** post-assembly peephole on top — it interleaves MFMA with the
   scalar ops `llirSched` structurally cannot reach.

### 1.5 Risks (carried forward)

* **Register pressure** — did not materialise here, but the kernel has zero
  headroom (256 VGPR + 256 AGPR, 0 spills), so any future change to the
  interleave has to be re-checked for spills before its TFLOPS are believed.
* **The `.so` is ABI-locked** to LLVM `850a2b1b`; rebuild it after editing the
  source (`plugins/llir_scheduler/README.md` has the `g++` line).
* **gfx950 is unverified by measurement.** The CDNA4 path is unchanged by
  arithmetic (`Bits*32/256 == Bits/8`), but the committed `.so` is a rebuild and
  no gfx950 was available to spot-check it.

## Opt 2 — park the surplus MFMAs between the last LR and the first LW  ✅ done

**in-loop MFMA efficiency 76.7% → 84.0%, +6.8% work per clock, +2.2% sustained
TFLOPS** (the gap between those last two is the point — see §2.3).

### 2.1 The change

Opt 1 left ~40 MFMAs per K-step sitting in clumps at region heads (`M10`, `M28`,
`M21`, `M27` in the §1.3 skeleton) — the scheduler's "leftover", the surplus
compute beyond what the anchors demand cover for, which it split evenly between
the region head and tail.

Instead, put all of it **between the region's last LR and its first LW**:

```cpp
// last LR that precedes the first LW in the anchor list
int splitIdx = -1;
for (size_t j = 0; j < firstLW; ++j)
  if (Anchors[j].Kind == SchedKind::LR) splitIdx = j;

if (splitIdx >= 0) splitLeftover = leftover;      // all of it, at that boundary
else               { tailLeftover = leftover/2;   // fallback: old head/tail split
                     headLeftover = leftover - tailLeftover; }
```

Two reasons that boundary and not another:

* **LDS port and FIFO.** Reads and writes share one LDS issue port and one
  SP-to-LDS FIFO, so a `ds_write` at the head of the queue blocks the `ds_read`s
  behind it (`docs/lds_throughput.md` §1.6). Draining the read burst before the
  writes start keeps the two from interleaving in the FIFO.
* **It is where the barrier is.** In an LDS-slot-recycling kernel the membar
  barrier lands exactly at the read→write boundary (the WAR hazard on the slot),
  and it carries an `s_waitcnt lgkmcnt(0)` that retires every outstanding
  `ds_read`. Surplus compute placed there is what the 8-deep read burst drains
  behind — candidate 1 from §1.4.

Regions with no LR-then-LW boundary keep the old even split.

### 2.2 Emitted schedule

Head clumps collapse (28 → 6, 27 → 6) and ~25 MFMAs move to the boundary:

```
opt1:  M28 R1 M2 R1 M2 ... R1 M2  W1 M2 W1 M2 W1 M2 W1 M1 G1 M2 G1 M4 G1 M4 G1 M5
opt2:  M6  R1 M2 R1 M2 ... R1 M25 W1 M1 W1 M2 W1 M2 W1 M1 G1 M2 G1 M4 G1 M4 G1 M5
```

**The surplus block needs fencing on both sides.** `insertSchedBarrier` pins an
anchor by emitting a `sched.barrier` *after* it, which gives

```
LR(last) ; sched.barrier ; [25 surplus MFMAs] ; LW(first) ; ...
```

— the near side only. With the far side open, LLVM's machine scheduler hoists
`LW(first)` up to just below that `sched.barrier`, over the whole block, and the
surplus ends up *after* the first write instead of before it: exactly the
separation the optimization exists to create, undone. So
`scheduleMFMAWithSpacing` reports the first-LW instruction and `scheduleBB`
fences it with `insertSchedBarrierBefore`, closing the block on both sides. The
extra fence is compile-time only — no code is emitted.

Worth knowing because the failure is invisible in the pass's own accounting:
`LLIR_SCHED_DEBUG=1` still reports `anchors=RRRRRRRRWWWWGGGG splitIdx=7` and the
IR is built correctly; only the emitted assembly shows it. It cost about half of
opt2's benefit before it was caught.

Counts are unchanged (256 mfma / 32 ds_read / 16 ds_write / 16 buffer_load /
3 barriers), registers unchanged (256 VGPR + 256 AGPR, **0 spills**), LDS 65536 B.

### 2.3 Result — and why TFLOPS understates it

ATT, 4096x4864x8256 fp16:

| | baseline | opt1 | **opt2** |
|---|---|---|---|
| cycles / iteration / wave | 5877 | 5343 | **4874** |
| in-loop MFMA efficiency | 69.7% | 76.7% | **84.0%** |
| `v_mfma` mean issue gap | 23.0 | 20.9 | **19.0** |
| whole dispatch (cyc) | 856,144 | 785,656 | **733,548** |

**17.1% fewer cycles than baseline**, 8.8% fewer than opt1.

Wall clock tells a different story, because the part is on its 750 W cap.
Sustained 12 s loops at 4096x4864x8256 fp16, three alternating runs per config,
median:

| | TFLOPS | clock | power | work / clock |
|---|---|---|---|---|
| opt1 | 579.4 | 1379 MHz | 750 W | 0.4202 |
| **opt2** | **592.0** | **1319 MHz** | 750 W | **0.4488  (+6.8%)** |

opt2 delivers **+2.2% TFLOPS while running 4.3% slower** — i.e. **+6.8% work per
clock**, an independent confirmation of the ATT cycle win (+9.6% work per cycle;
the two differ by about the run-to-run noise, which is large here: individual
sustained samples ranged 551–608 TFLOPS).

So opt2 is a real scheduling improvement that this part converts mostly into
lower frequency rather than throughput: a denser MFMA stream draws more power,
and at the cap that is paid in clock. On hardware that is not power-limited the
full ~9% would be visible. The ATT number is not a TFLOPS claim here.

> [!NOTE]
> **gfx950 is unaffected.** The policy only engages when a region has an LW
> anchor, and the gfx950 kernels fill LDS with `buffer_load_to_shared`, which
> classifies as **GR** (`buffer.load.lds`), not LW. With no LW, `splitIdx` stays
> −1 and the old even head/tail split is used — bit-for-bit unchanged. The one
> exception in this repo is `intra_wave/a4w4/v0_sliceN`, which stages MXFP4
> *scales* through LDS with a real `local_store`; its successor `v1_sliceMN`
> does not. Not re-measured (no gfx950 available).

### 2.4 What is still on the table

| candidate (from §1.4) | status |
|---|---|
| 2. head/tail leftover clumps | **done — this optimization** |
| 1. barriers / `lgkmcnt(0)` | largely addressed — in Region 0 the surplus now lands directly after the barrier, so the read burst drains behind 23 MFMAs |
| 3. `amdgcnas` post-assembly peephole | open, untried on gfx942 |

At 84.0% the 4-wave kernel is within 2.5 points of the 8-wave `inter_wave`
kernel (86.5%). Opt 3 closes the gap and passes it.

## Opt 3 — drain the prologue's LDS reads before the loop  ✅ done

**in-loop MFMA efficiency 84.0% → 89.0%, +9.4% work per clock, and the 4-wave
kernel now beats hipBLASLt on every shape and dtype.** One line of kernel code.

### 3.1 The finding: a first-iteration hazard paid on every iteration

The loop carried **3 `s_barrier` + 3 `s_waitcnt lgkmcnt(0)`** per K-step, and the
`lgkmcnt(0)`s cost 5.8% of all issue slots (mean 94 cycles, max 260). Hand-running
membar over the loop body's LDS order predicts only **2** barriers:

```
op0 LR A_bot | op1 LW B_left | op2 LR B_right | op3 LW A_top
op4 LR B_left | op5 LW A_bot | op6 LR A_top  | op7 LW B_right
```

* before `op0` — loop-carried RAW on A_bot (`op5` writes it)
* before `op4` — RAW on B_left (`op1` writes it)

The third came from the **prologue**. It ends with

```python
b_left = smemB_left.load(dotOpLayoutB)
a_top  = smemA_top.load(dotOpLayoutA)
```

and those reads are still pending in membar's `BlockInfo` at the loop header.
That set is joined into the loop-body input, where it collides with `op1
LW B_left` as a **WAR hazard that can only occur on the first iteration**.
membar resolves it the only way its algorithm allows — a barrier inside the loop
body — so every K-step pays for a one-time condition.

Confirmed by construction: adding two explicit `gl.barrier()` at the points the
hazard analysis says are sufficient still left 3 barriers, because the prologue
hazard is independent of both. Draining the prologue instead fixes it.

### 3.2 The change

```python
b_left = smemB_left.load(dotOpLayoutB)
a_top  = smemA_top.load(dotOpLayoutA)

gl.barrier()      # <-- close the prologue's pending LDS reads

for k in range(0, iterMax - 2):
```

One barrier in the prologue (paid once) removes one from every K-step.

### 3.3 The waits go fine-grained

With the redundant barrier gone, the loop's sync profile changes character
completely — `SIInsertWaitcnts` can finally do its job:

| | opt2 | opt3 |
|---|---|---|
| `s_barrier` | 3 | **2** |
| `s_waitcnt lgkmcnt(0)` | 3 | **1** |
| `s_waitcnt lgkmcnt(N>0)` | 0 | **9** — `lgkm9`, `lgkm12`, `lgkm13`, `lgkm14` |

The blanket drains are replaced by precise dependency waits. Those are what one
would *expect* to see before the consuming MFMAs; they were absent before only
because the barrier's full drain had already retired everything. Measured cost
of all 479 of them in the ATT window: **median 4 cycles, max 4** — free.

Instruction counts, registers (256 VGPR + 256 AGPR, **0 spills**) and LDS are
unchanged.

### 3.4 Result

ATT, 4096x4864x8256 fp16:

| | baseline | opt1 | opt2 | **opt3** |
|---|---|---|---|---|
| cycles / iteration / wave | 5877 | 5343 | 4874 | **4601** |
| in-loop MFMA efficiency | 69.7% | 76.7% | 84.0% | **89.0%** |
| whole dispatch (cyc) | 856,144 | 785,656 | 733,548 | **701,952** |

**21.7% fewer cycles than baseline.** Where the issue slots went:

| | opt2 | opt3 |
|---|---|---|
| `v_mfma` | 73.1% | **77.5%** |
| `ds_write` | 10.6% | 10.2% |
| `s_waitcnt lgkmcnt(0)` | **5.8%** | **0.1%** |
| `s_barrier` | 2.3% | 2.9% (fewer, but mean 37 → 65 cyc) |
| `s_waitcnt lgkmcnt(N>0)` | – | 0.8% |
| total sync | **8.7%** | **4.3%** |

Sustained 12 s loops, 3 alternating runs per config, both pinned at 750 W:

| | TFLOPS | clock | work / clock |
|---|---|---|---|
| opt2 | 584.6 | 1348 MHz | 0.4313 |
| **opt3** | **605.3** | **1283 MHz** | **0.4718  (+9.4%)** |

`do_bench` median of 3 — **ahead of hipBLASLt everywhere**, which the 4-wave
kernel never was before:

| K | fp16 | torch | | bf16 | torch |
|---|---|---|---|---|---|
| 2112 | **582** | 572 | +1.9% | **613** | 610 |
| 4160 | **629** | 615 | +2.4% | **666** | 646 |
| 8256 | **588** | 563 | +4.4% | **660** | 622 |
| 16448 | **616** | 575 | +7.1% | **666** | 614 |

### 3.5 It exposed the vmcnt conservatism, as predicted

`SIInsertWaitcnts` emits `vmcnt(3)/(1)/(0)` before region 0's `ds_write`s. The
correct values are `vmcnt(15)/(13)/(12)`: those writes consume GR#1–#4, the
*oldest* 4 of 16 loads in flight, so GR#5–#16 could stay outstanding. The pass
counts only region 0's own 4 loads and loses the rest across the back edge.
(Not a forced flag — Triton never sets `amdgpu-waitcnt-load-forcezero`.)

Before opt3 this measured as free (`vmcnt(0)`: max gap **4 cycles**) — but only
because the `lgkmcnt(0)` stall immediately before it was already covering the
HBM latency. With that stall gone, `vmcnt(0)`'s max gap is **168 cycles**. Still
only 0.2% of the window today, but the mechanism is now live and will grow as
the remaining sync cost comes down.

> [!NOTE]
> This section originally concluded that fixing it "needs a MachineIR-level
> change, which the LLIR plugin cannot reach". That was wrong. The cause is a
> preheader/loop-body ordering mismatch that a single `gl.barrier()` in the
> prologue removes — see §5. The `vmcnt(15)/(13)/(12)` predicted here is
> exactly what that fix produces.

### 3.6 What is still on the table

| candidate | status |
|---|---|
| barrier count | **done** — 3 → 2, the minimum for this slot schedule |
| `ds_write` 10.2% | at the CDNA3 LDS port limit (32 cyc = 4 SIMD x 1024 B / 128 B per clk); only reducible by writing less to LDS |
| `vmcnt` conservatism | **done** — see §5, one prologue barrier; perf-neutral but removes the latent drain |
| `s_barrier` 2.9% | 2 barriers is the minimum given the loop-carried WAR/RAW on each slot |
| `amdgcnas` peephole | open, untried on gfx942 |

At 89.0% the 4-wave kernel has now passed the 8-wave `inter_wave` kernel's
86.5%. Re-running opt1-3 against `inter_wave` is probably the highest-value next
step — it has the same prologue pattern.

## Opt 4 — the vmcnt conservatism, first attempt: register form  ❌ not landed

First attempt at the conservative `vmcnt` from Opt 3 §3.5.

> [!IMPORTANT]
> **This section identifies a trigger, not the cause.** Flipping
> `amdgpu-mfma-vgpr-form` does change the emitted waits, but it is not why they
> are conservative, and there is no LLVM bug here to report. The actual root
> cause — and a fix that costs one prologue barrier instead of the whole
> register-allocation strategy — is in §5. Kept for the reasoning trail and
> because the `llc` reproducer technique below is reusable.

### 4.1 What it looked like: self-inflicted, not an LLVM loop-analysis limit

`TRITON_FORCE_MFMA_AGPR=1` makes Triton do **two** things
(`python/src/llvm.cc:349`):

1. the kernel passes `llvm_fn_attrs="amdgpu-agpr-alloc=256"` — reserves the AGPRs
2. Triton sets the **global** LLVM option `amdgpu-mfma-vgpr-form=false`

Only (1) is needed for the accumulators. (2) additionally pushes the
**global-staging registers** out of AGPRs into VGPRs, and that is what makes
`SIInsertWaitcnts` conservative:

```
vgpr-form=true  (LLVM default)   buffer_load_dwordx4 a[64:67]
                                 s_waitcnt vmcnt(12)/(10)/(9)  at all 4 regions
vgpr-form=false (Triton)         buffer_load_dwordx4 v[88:91]
                                 s_waitcnt vmcnt(3)/(1)/(0)    at region 0 only
```

`vmcnt(0)` drains all 16 in-flight `buffer_load`s where retiring the oldest 4
would do (theoretical minimum `vmcnt(15)/(13)/(12)`; LLVM's default
`(12)/(10)/(9)` is within 3 of it).

**Minimal reproducer, no Triton required** — same IR, same LLVM, one flag apart,
with **byte-identical `lgkmcnt`**; only `vmcnt` degrades:

```bash
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 kernel.ll -o good.s
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -O3 -amdgpu-mfma-vgpr-form=false kernel.ll -o bad.s
```

Found without rebuilding LLVM: the `llc` shipped with the Triton LLVM pin is the
same commit built *with assertions*, so it can be run directly on Triton's
emitted IR. It produced the correct waits, which located the problem in Triton's
codegen configuration rather than in the pass.

### 4.2 The fix works — and is a net loss here

Dropping the env var and carrying `amdgpu-agpr-alloc=256` unconditionally in
`llvm_fn_attrs` leaves register allocation **identical** (intra_wave 512 VGPR /
256 AGPR / 0 spills / 1 wave per SIMD; inter_wave 236 / 0 / 0 / 2 waves per
SIMD) and produces the precise waits. Alternating ATT captures, two samples
each, 4096x4864x8256 fp16:

| | cyc/iter | MFMA eff | `vmcnt` count | `vmcnt` total | `vmcnt` **max** |
|---|---|---|---|---|---|
| opt3 (conservative) | 4647, 4650 | 88.1% | 159 | 4384, 5896 | **996, 1276** |
| opt4 (precise) | 4698, 4698 | 87.2% | 527 | 2108 | **4** |

The stall is genuinely eliminated — max 1276 → 4 cycles, total halved. But
**cycles per iteration get 1.1% worse**, reproducibly (two identical samples per
config). Precise waits cost 368 extra `s_waitcnt` instructions in the measured
window; at ~4 cycles of issue each that is ~1500 cycles, and the remaining
difference is most likely the loss of a secondary effect: `vmcnt(0)` empties the
TCP queue once per iteration, so the following 16 `buffer_load`s issue into a
drained queue instead of one that still holds 9-12 requests.

`do_bench` had said the same thing earlier (~1-2% slower) and was wrongly
dismissed as clock noise. The two agree.

> [!WARNING]
> An earlier version of this section claimed opt4 was a win. That was measured
> on a mis-captured trace: `tools/run_att.py` was still forcing
> `TRITON_FORCE_MFMA_AGPR=1` internally at the time, so the "opt4" capture was
> really a second sample of opt3. The giveaway is the `vmcnt` instruction count
> in the traced window — 159 for the conservative build, 527 for the precise
> one. Worth checking that number on any future capture.

### 4.3 What to do with it

* **Kernel: keep opt3.** `TRITON_FORCE_MFMA_AGPR=1` stays required.
* **Do not file upstream.** The follow-up in §5 shows `SIInsertWaitcnts` is
  emitting the minimal *safe* wait for the bracket it is handed; the register
  form only perturbs an input to that bracket. Artifacts (`.ll` / `.s` triple)
  stay in `/data/att_gfx942_intra_wave_..._opt4_rejected/` as the reproducer for
  §5's analysis.

## Opt 5 — the real root cause: preheader load order  ✅ landed (perf-neutral)

The conservative `vmcnt` is a **preheader/loop-body ordering mismatch**, and
neither LLVM nor the register form is at fault.

### 5.1 The mechanism

`SIInsertWaitcnts` joins the loop's two incoming paths with
`WaitcntBrackets::mergeScore` (`SIInsertWaitcnts.cpp:2698`):

```cpp
unsigned MyShifted    = Score      <= M.OldLB   ? 0 : Score      + M.MyShift;
unsigned OtherShifted = OtherScore <= M.OtherLB ? 0 : OtherScore + M.OtherShift;
Score = std::max(MyShifted, OtherShifted);
```

Both brackets are right-aligned to a common upper bound and merged per register
with `max`. Higher score = younger, and `wait = UB - score`, so `max` is the
*stronger* wait — correct and necessary, because on the preheader path the
register really does have that many younger loads outstanding.

The damage is what the two paths disagree about. The 16 loop-carried
`buffer_load`s, by preheader position of body-load #1…#16:

| | preheader position of body load #1…#16 | rotation |
|---|---|---|
| AGPR staging (opt4) | `3,4,5,…,15,0,1,2` | 3 |
| VGPR staging (opt3, shipped) | `12,13,14,15,0,1,…,11` | **12** |

Rotation 3 keeps `max(rank_body, rank_pre)` monotone for loads #1–#13, so ages
survive. Rotation 12 gives the **four oldest** loads preheader ranks 13–16, so
`max` promotes them to the top of the bracket — the oldest in-flight load is
scored as the youngest, and `wait = UB - score` collapses.

The model `merged = max(rank_body, rank_pre)` reproduces the pass's printed
brackets exactly (constant −1 UB offset) and every emitted wait in both builds:

| `ds_write` src | merged score | predicted | emitted |
|---|---|---|---|
| `a[64:67]` (oldest) | 3 | `vmcnt(12)` | `vmcnt(12)` |
| `a[68:71]` | 5 | `vmcnt(10)` | `vmcnt(10)` |
| `a[60:63]` | 6 | `vmcnt(9)` | `vmcnt(9)` |
| `v[88:91]` (oldest) | 12 | `vmcnt(3)` | `vmcnt(3)` |
| `v[92:95]` | 14 | `vmcnt(1)` | `vmcnt(1)` |
| `v[84:87]` | 15 | `vmcnt(0)` | `vmcnt(0)` |

The rotation is introduced by the **MachineScheduler**, not by Triton: the
emitted LLVM IR issues the loads in identical order in the preheader and the
body (`%78,%80,%82,%84 / %87… / %95… / %103…`). The scheduler sinks the first
tile's four loads to the end of the preheader.

### 5.2 The change

Gluon exposes no `sched.barrier`, but `s_barrier` has unmodeled side effects, so
`ScheduleDAGInstrs` chains memory ops to it — it fences the MachineScheduler.
One barrier between the first and second prologue load is enough:

```python
gB_left = buffer_load(ptr=b_base, offsets=b_left_offsets)
gl.barrier()                       # pin prologue load order == loop-body order
gA_top = buffer_load(ptr=a_base, offsets=a_top_offsets)
gA_bot = buffer_load(ptr=a_base, offsets=a_bot_offsets)
gB_right = buffer_load(ptr=b_base, offsets=b_right_offsets)
```

| | preheader→body permutation | in-loop `vmcnt` waits |
|---|---|---|
| opt3 | `[12,13,14,15,0,…,11]` | `3, 1, 0` — region 0 only |
| **opt5** | `[0,1,2,…,15]` identity | `15, 13, 12` — **all 4 regions** |

That is exactly the `vmcnt(15)/(13)/(12)` optimum predicted in §3.5, and better
than LLVM's own AGPR-form output of `12/10/9`. Register allocation is unchanged
(512 VGPR / 256 AGPR / 0 spills), and the loop keeps opt3's 2 barriers and 1
`lgkmcnt(0)`. Cost: +1 prologue barrier, +9 in-loop `s_waitcnt`.

### 5.3 Result — the stall is eliminated, the cycles are a wash

Alternating ATT captures, 3 samples each, 4096×4864×8256 fp16, 90 s cooldowns:

| | cyc/iter | median | MFMA eff | `vmcnt` n | `vmcnt` sum | `vmcnt` **max** | `s_barrier` sum |
|---|---|---|---|---|---|---|---|
| opt3 | 4618.7, 4621.7, 4638.0 | **4621.7** | 88.3–88.7% | 3 | 6096–10084 | **4–560** | 43040–43988 |
| opt5 | 4616.1, 4621.5, 4621.9 | **4621.5** | 88.6–88.7% | 12 | 24384 | **4** | 37092–38440 |

Medians differ by 0.2 cycles — **perf-neutral**. The stall is genuinely gone:
every `vmcnt` in opt5 is a pure 4-cycle issue slot, deterministically, whereas
opt3's varies with memory timing (max 4 in one sample, 560 in another). The
barrier stall also drops ~13%. Those gains are cancelled by the 9 extra
`s_waitcnt` issue slots (24384 vs ~8000 cycles on the `vmcnt` line).

All 8 shape/dtype combos correct, still ahead of hipBLASLt.

### 5.4 Why land a neutral change

* It removes a **latent** hazard, not a live one. Today `vmcnt(0)` mostly hides
  behind the barrier and `lgkmcnt` stalls; as those come down there would not be
  enough region-0 MFMA to cover HBM latency, and the drain would become the
  binding constraint.
* It stops the kernel depending on a **scheduler accident**. Whether the waits
  are `12/10/9` or `3/1/0` currently turns on how far the MachineScheduler
  happens to rotate the preheader (3 vs 12). The barrier makes it invariant.
* It makes the `vmcnt` cost deterministic and bounded rather than
  memory-timing-dependent.

The counter-argument is real: 9 extra in-loop instructions for zero measured
gain. If the loop is ever re-tuned toward instruction-issue limits, re-measure.

## 6. `ds_write` ablation — where the last 11% is, and why it stays

Assembly-level ablation: dump the `.amdgcn`, delete the 16 `ds_write_b128` from
the loop body, reassemble via `TRITON_KERNEL_OVERRIDE`. Everything else is held
constant — same register allocation, same occupancy, same 256 MFMA / 32
`ds_read` / 16 `buffer_load` / 2 `s_barrier`, same 12 `s_waitcnt vmcnt`. The
result is numerically wrong (LDS is never refilled); this is a timing probe.

| | cyc/iter | in-loop MFMA eff | loop instrs |
|---|---|---|---|
| opt5 (full) | 4621.5 | 88.63% | 357 |
| `ds_write` removed | 4255.0 | **96.26%** | 341 |
| delta | **−366.5 (−7.9%)** | **+7.63 pts** | −16 |

Time attribution inside the loop:

| | opt5 (full) | ablated |
|---|---|---|
| mfma | 77.71% | 87.73% |
| **`ds_write_b128`** | **10.62%** | — |
| `ds_read_b128` | 4.55% | 5.03% |
| `buffer_load_dwordx4` | 1.96% | 2.25% |
| `s_waitcnt` | 1.92% | 2.09% |
| `s_barrier` | 1.59% | 1.10% |

So `ds_write` is **~two thirds of all remaining inefficiency**. Two details:

* Gross cost is 486.7 cyc/iter ÷ 16 = **30.4 cycles per `ds_write_b128`**, which
  confirms the 32-cycle LDS-port model in §3.6 (4 SIMD × 1024 B ÷ 128 B/clk).
  Net saving is only 22.9 per write, so ~3/4 of it is on the critical path.
* `s_barrier` cost drops 37% as a side effect — part of the 10.62% is really
  "`ds_write` plus the synchronisation it forces".

**This is the limit of the intra_wave design on MI300.** A `ds_write_b128`
takes ~20 cycles to issue (4 for the address, 16 for the data), and a
`v_mfma_f32_16x16x16_f16` covers only 16 — one MFMA cannot hide one `ds_write`.
16 × 20 = 320 unhideable cycles per iteration, against the 366.5 measured. Going
further means writing *less* to LDS (e.g. feeding one MFMA operand from
registers instead of round-tripping A through LDS), which is a different
pipeline, not a tuning knob.

> [!WARNING]
> TFLOPS is useless for this comparison. The ablated build ran at **1.266 GHz vs
> 1.387** — 8.7% slower clock, because a denser MFMA stream draws more power at
> the 750 W cap — so it measured *slower* by median wall-clock (626.2 vs 638.5)
> despite executing 7.9% fewer cycles. An isolated first measurement caught a
> cool run and read 677.2 vs 631.5, overstating the case in the other direction.

## 7. Final measurement

`tools/bench_prepared.py`: prepared launch (compile once, pre-bind arguments,
enter the launch stub directly), 3 rotating tensor sets, 1000 dispatches, mean
of the **last 100**, timed with `rocprofv3 --kernel-trace`.

Three alternating-order sweeps over all 8 GPUs, per-GPU median TFLOPS
(4096×4864×8256 fp16):

| GPU | 0 | 1 | 2 | **3** | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| median | 610.4 | 602.1 | 615.7 | **618.1** | 615.9 | 603.6 | 612.2 | 594.3 |

**GPU 3 is fastest and is the reference device from here on.** Read the ranking
carefully: the top four (3, 4, 2, 6) are within 1% and reorder between sweeps;
only the bottom is solid — GPUs 7, 1 and 5 are slow in all three orderings,
which looks like real device variation. Total spread 4.0%.

Final state, GPU 3, dedicated runs:

| | TFLOPS | µs/dispatch | in-loop MFMA eff | cyc/iter | clock |
|---|---|---|---|---|---|
| fp16 | 616.1 | 533.97 | **88.43%** | 4631.9 | 1.425 GHz |
| bf16 | **652.1** | 504.47 | — | — | — |

Trace: `/data/att_gfx942_intra_wave_4096x4864x8256_fp16_final_gpu3/`.

> [!CAUTION]
> **The Triton cache key does not include `TRITON_KERNEL_OVERRIDE`.** The
> hand-edited `ds_write`-ablated kernel from §6 persisted in `~/.triton/cache`
> and silently poisoned a full 8-GPU sweep *and* an ATT capture — both reported
> the ablated build as if it were the shipped kernel (~630-649 TFLOPS instead of
> ~610-620). The tell was `instrs=341` instead of 357 in the ATT decode.
> `bench_prepared.py` now verifies the kernel output against a torch reference
> *before* timing and aborts if it fails. Always `rm -rf ~/.triton/cache` after
> any override run, and check loop instruction counts on every capture — this is
> the same class of mis-capture as the opt4 trace in §4.

## 3. Backlog (not yet attempted)

* Close the remaining 11% — see the table in §3.6.
* `amdgcnas` post-assembly peephole on gfx942.
* Measure the fp8/bf8/i8 CDNA3 cycle counts added to the scheduler's table on
  faith (derived, not measured). Needed before porting a8w8 to gfx942.
* Epilogue: currently stores from the mfma layout (`buffer_store_dwordx2`).
  Converting to a blocked layout would give `dwordx4` but needs LDS scratch,
  and LDS is full — would require freeing a slot first.
* The `s_waitcnt lgkmcnt(0)` that each membar barrier drags in is coarser than
  needed; a finer `lgkmcnt(N)` would let some LDS traffic stay in flight across
  the barrier. This is a Triton membar/AMD-backend change, not a kernel change.
