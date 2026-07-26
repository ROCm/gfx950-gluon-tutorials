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
