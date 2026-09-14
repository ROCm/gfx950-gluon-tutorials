# v9 with MFMA accumulators pinned to AGPRs

This experiment pins every MFMA accumulator of the
[a16w16 v9 kernel](../../kernels/gemm/intra_wave/a16w16/v9_beyond_hotloop/matmul_kernel.py) to AGPRs
with inline-asm register-class constraints. The question is whether this can replace `force-agpr`,
which is the pair of LLVM settings `amdgpu-mfma-vgpr-form=0` (process-wide LLVM option) +
`amdgpu-agpr-alloc=256` (kernel function attribute); see
[docs/performance_philosophy.md](../../docs/performance_philosophy.md).

It uses two extensions:

- **`cd_regclass`** (Triton): `gl.amd.cdna4.mfma(a, b, acc, cd_regclass="a")`, and the same argument
  on `gl.amd.cdna4.mfma_scaled`, make the compiler pin each MFMA's C and D next to the MFMA
  instruction.
- **llirSched** (this repo's [plugins/llir_scheduler](../../plugins/llir_scheduler)): the plugin keeps
  those pins attached to their MFMAs when it reorders the loop.

**Background.** An earlier attempt placed the pins from Gluon source instead, wrapping the
accumulator before and after every `mfma()` call with `inline_asm_elementwise`, using
[triton-lang/triton#10337](https://github.com/triton-lang/triton/pull/10337) (Lei Zhang's branch
`antiagainst/triton:pr-10337-amd-register-classes`). It does not work on v9. A Gluon-level pin covers
the whole accumulator tensor, so each `mfma()` call becomes 16 pins, then 32 MFMAs, then 16 more pins.
The loop kept 480 copy instructions and only 176 of its 256 MFMAs in AGPR form, and pinning only D
spilled 147 VGPRs. Lei's example on that branch only has one MFMA per `mfma()` call (a single wave,
one 16×16×32 or 32×32×16 tile, no register pressure), so a tensor-level pin sits next to its MFMA by
construction and the problem cannot show up there. Those files are in this directory's git history
(commit `efaa3ad`); `cd_regclass` does not need PR #10337.

## Triton compiler

- Branch `gfx950-tutorial-v2.1-cd-regclass` on top of the tutorial's pin
  [`gfx950-tutorial-v2.1`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v2.1)
  (`e346b8a741cf5366a5bc9207a44c99d7e70fa828`), so everything the tutorial relies on is there too:
  - `98dc3eeb7e7bb2b2fbba3c224fefbfb12041ffa2` adds `cd_regclass` to `mfma`.
  - `96f4a1e2d` adds it to `mfma_scaled`.
- LLVM `b010a18d2b648cab83c83967ff26b8fde11acdc6`, the same pin as `gfx950-tutorial-v2.1`.
- Built with default visibility so the llirSched plugin can load in-process:
  `TRITON_EXT_ENABLED=1 TRITON_BUILD_PROTON=OFF TRITON_APPEND_CMAKE_ARGS=-DTRITON_BUILD_UT=OFF pip install --no-build-isolation .`

Flow: no amdgcnas and no force-agpr unless stated. As in `gfx950-tutorial-v2.1`, the compiler passes
`amdgpu-use-amdgpu-trackers` on gfx950.

## Files

- [`matmul_kernel_cd_regclass.py`](matmul_kernel_cd_regclass.py): v9 with every
  `gl.amd.cdna3.mfma(a, b, acc)` written as `gl.amd.cdna4.mfma(a, b, acc, cd_regclass="a")`. A diff
  against v9 shows only the 16 call sites and the kernel renamed to `v9_cd_regclass`.
- [`run.py`](run.py): correctness check and timing, 4096×4096×8192 fp16.
- [`ir_dumps/`](ir_dumps/): the LLVM IR (`.llir`) and gfx950 assembly (`.amdgcn`) Triton generated,
  `v9_cd_regclass.*` without llirSched and `v9_cd_regclass_llirsched.*` with it.

```bash
python run.py                                         # without llirSched
LLVM_PASS_PLUGIN_PATH=$PWD/../../plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 python run.py  # with llirSched
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/dump python run.py  # regenerate ir_dumps/
```

## The `cd_regclass` extension

```python
acc = gl.amd.cdna4.mfma(a, b, acc, cd_regclass="a")  # "a" = AGPRs, "v" = VGPRs
acc = gl.amd.cdna4.mfma_scaled(a, a_scale, "e2m1", b, b_scale, "e2m1", acc, cd_regclass="a")
```

- **Gluon** (`gl.amd.cdna3.mfma`, re-exported as `gl.amd.cdna4.mfma`, and `gl.amd.cdna4.mfma_scaled`)
  takes an optional `cd_regclass` of `None`, `"a"` or `"v"`, and records it as the
  `amdg.cd_regclass` attribute on the `tt.dot` / `tt.dot_scaled`.
- **MFMA lowering** (`third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp`, both the plain and
  the scaled path) wraps each MFMA tile's accumulator in an empty asm whose output is tied to its
  input (`"=a,0"` / `"=v,0"`): once before the tile's first K-step (C) and once after its last (D).
  The asm emits no instruction, but the value must pass through an AGPR (or VGPR) tuple at that
  point. It uses the MFMA's own vector type, so a 32×32 accumulator is pinned as one 16-register
  tuple.
- **Lit test** `test/TritonGPU/amd/mfma-cd-regclass.mlir`: pin order with two K-steps, no pins
  without the attribute, a 32×32 tile pinned as `vector<16xf32>`, a scaled-dot case, and an error for
  invalid values.

From `ir_dumps/v9_cd_regclass.llir` (the loop starts at line 370):

```llvm
%728 = tail call <4 x float> asm "", "=a,0"(<4 x float> %727)  ; pin C
%729 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %688, <8 x half> %648, <4 x float> %728, i32 0, i32 0, i32 0)
%730 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %693, <8 x half> %653, <4 x float> %729, i32 0, i32 0, i32 0)
%731 = tail call <4 x float> asm "", "=a,0"(<4 x float> %730)  ; pin D
```

## The llirSched extension

Before this branch, llirSched reverted any block containing the pins: it moved MFMAs but left the
pins behind, which broke the IR. The plugin on this branch (`LlirSchedPlugin.cpp`) now:

- **Moves each pin with its MFMA.** When the scheduler moves an MFMA, the C pin goes directly before
  it and the D pin directly after it. Its hoisting of MFMA inputs and sinking of MFMA-result
  extracts also looks through pins.
- **Fences each pinned tile.** A `sched.barrier` after every D pin stops LLVM's machine scheduler
  from reordering the pinned MFMAs in the stretches between memory anchors. Without these fences
  that reordering left 200 copies in the v9 loop, because LLVM's post-RA copy-removal pass could not
  find a free AGPR for 17 of the MFMA chains. The fences have a cost, see
  [What ATT shows](#what-att-shows).

Kernels without pins get byte-identical output from the plugin before and after this change:
running `opt -passes=llir-sched` with the old and the new `libLlirSched.so` over the LLVM IR of the
unpinned v9 (16×16 and 32×32), v6, a8w8, a4w4 v0 and a4w4 v1 kernels gives identical files and
identical `sched.barrier` counts. Everything the change adds is reached only through a pin.

## How the numbers are measured

- **Copies in the loop** (`v_accvgpr_read` / `_write` / `_mov`), **AGPR-form MFMAs**, **VGPR spills**,
  **`s_nop`** and **`sched.barrier`s** come from the generated assembly and kernel metadata. The hot
  loop is the loop body containing the MFMAs, from its label to its backward branch.
- **MFMA efficiency** and **cycles per loop iteration** come from `rocprofv3 --att` on the 15th
  dispatch, CU 0 of every shader engine, decoded with
  [`scripts/process_json.py`](../../scripts/process_json.py). All kernels here run 4 warps, i.e. 1
  wave/SIMD, so the per-wave figure the script prints is also the per-SIMD figure (no ×2, unlike the
  8-wave kernels; see [docs/mfma_efficiency.md](../../docs/mfma_efficiency.md)).
- **Cold TFLOPS** come from `rocprofv3 --kernel-trace` over 200 dispatches, median kernel time.
- Both use rotating inputs larger than 512 MB, so every dispatch starts with cold caches, as in
  `bench.py --rocprof`. All 4096×4096×8192. Every configuration was checked against its reference.
- The per-configuration traces, the summary CSVs and the driver scripts are in
  `/data/att_v9_cd_regclass_sweep_20260912/`, and two full ATT traces (row 4 and force-agpr) in
  `/data/att_a16w16-v9_cd_regclass_llir-sched_...` and `..._v9_beyond_hotloop_llir-sched_force-agpr_...`.

## Results

**v9, 16×16×32 fp16** (256 MFMAs in the hot loop)

| variant | llirSched | copies in loop | AGPR-form MFMAs | VGPR spills | `s_nop` (cycles) | `sched.barrier`s | MFMA efficiency | cycles/iter | cold TFLOPS |
|---|---|---|---|---|---|---|---|---|---|
| unpinned | no | 36 | 52/256 | 0 | 1 (1) | 0 | 70.3% | 5827 | 922 |
| unpinned | yes | 119 | 58/256 | 0 | 40 (169) | 96 | 76.7% | 5344 | 936 |
| **`cd_regclass`** | no | **0** | 256/256 | 0 | 47 (47) | 0 | 71.8% | 5703 | 917 |
| **`cd_regclass`** | yes | **0** | 256/256 | 0 | 90 (90) | 224 | **91.4%** | 4481 | 1024 |
| force-agpr | no | 0 | 256/256 | 0 | 1 (1) | 0 | 70.7% | 5793 | 909 |
| force-agpr | yes | 0 | 256/256 | 0 | 0 (0) | 96 | **96.5%** | 4245 | 1012 |

Two things to note. Without llirSched the register placement barely matters: every variant sits near
70% because the schedule, not the accumulator's register file, is the limit. And MFMA efficiency does
not always track end-to-end time: `cd_regclass` is 5 points behind force-agpr here yet slightly ahead
in cold TFLOPS (1024 vs 1012).

### Other shapes and data types

Same measurements on a 32×32×16 variant of v9 and on the tutorial's fp8 and MXFP4 GEMMs, with
`cd_regclass="a"` on every `mfma` / `mfma_scaled` call. These all use K=8192 so they are comparable
with v9 above; that is *not* the shape the tutorial publishes for fp8 and MXFP4, see
[At the tutorial's headline shapes](#at-the-tutorials-headline-shapes).

**v9, 32×32×16 fp16** (128 MFMAs in the loop)

| variant | llirSched | copies in loop | AGPR-form | VGPR spills | MFMA efficiency | cycles/iter | cold TFLOPS |
|---|---|---|---|---|---|---|---|
| unpinned | no | 112 | 24/128 | 0 | 62.6% | 6540 | 811 |
| unpinned | yes | 144 | 32/128 | 0 | 85.3% | 4801 | 876 |
| **`cd_regclass`** | no | **0** | 128/128 | 0 | 63.7% | 6427 | 815 |
| **`cd_regclass`** | yes | **0** | 128/128 | 0 | 92.9% | 4411 | 897 |
| force-agpr | no | 0 | 128/128 | **27** | 16.3% | 25094 | 289 |
| force-agpr | yes | 0 | 128/128 | 0 | 96.6% | 4241 | 915 |

**a8w8, fp8 e5m2, 16×16×128 scaled** (128 MFMAs in the loop)

| variant | llirSched | copies in loop | AGPR-form | VGPR spills | MFMA efficiency | cycles/iter | cold TFLOPS |
|---|---|---|---|---|---|---|---|
| unpinned | no | 33 | 32/128 | 0 | 67.3% | 6083 | 1803 |
| unpinned | yes | 49 | 34/128 | 0 | 91.8% | 4460 | 1954 |
| **`cd_regclass`** | no | **0** | 128/128 | 0 | 72.7% | 5636 | 1852 |
| **`cd_regclass`** | yes | **0** | 128/128 | 0 | 95.2% | 4304 | 2014 |
| force-agpr | no | 0 | 128/128 | 0 | 70.8% | 5789 | 1858 |
| force-agpr | yes | 0 | 128/128 | 0 | 96.9% | 4225 | 2056 |

**a4w4 v0, MXFP4, 16×16×128 scaled** (256 MFMAs in the loop)

| variant | llirSched | copies in loop | AGPR-form | VGPR spills | MFMA efficiency | cycles/iter | cold TFLOPS |
|---|---|---|---|---|---|---|---|
| unpinned | no | 68 | 68/256 | 0 | 50.5% | 8107 | 2633 |
| unpinned | yes | 238 | 62/256 | **186** | 7.8% | 52742 | 709 |
| **`cd_regclass`** | no | **0** | 256/256 | **60** | 56.1% | 7299 | 2637 |
| **`cd_regclass`** | yes | **0** | 256/256 | **28** | 70.2% | 5837 | 2920 |
| force-agpr | no | 0 | 256/256 | 0 | 55.4% | 7395 | 2742 |
| force-agpr | yes | 0 | 256/256 | 0 | 60.7% | 6745 | 2992 |

**a4w4 v1, MXFP4, 16×16×128 scaled** (256 MFMAs in the loop)

| variant | llirSched | copies in loop | AGPR-form | VGPR spills | MFMA efficiency | cycles/iter | cold TFLOPS |
|---|---|---|---|---|---|---|---|
| unpinned | no | 79 | 48/256 | 0 | 64.0% | 6404 | 2775 |
| unpinned | yes | 220 | 82/256 | 12 | 40.1% | 10226 | 2363 |
| **`cd_regclass`** | no | **0** | 256/256 | 0 | 64.2% | 6379 | 2861 |
| **`cd_regclass`** | yes | **0** | 256/256 | 0 | 67.8% | 6040 | 3072 |
| force-agpr | no | 0 | 256/256 | 0 | 66.2% | 6188 | 2833 |
| force-agpr | yes | 0 | 256/256 | 0 | 67.5% | 6067 | 3050 |

`cd_regclass` removes every loop copy and puts every loop MFMA in AGPR form in all five kernels. With
llirSched it beats force-agpr's MFMA efficiency on a4w4 v0 (70.2% vs 60.7%) and matches it on a4w4 v1,
while trailing it on v9 (91.4% vs 96.5%, 92.9% vs 96.6%) and a8w8 (95.2% vs 96.9%) because of the
per-tile fences. It also rescues the two configurations where llirSched alone collapses: a4w4 v0
(7.8% → 70.2%) and a4w4 v1 (40.1% → 67.8%).

### At the tutorial's headline shapes

The tutorial publishes MFMA efficiency at each precision's headline shape (fp16 K=8192, BF8 K=16384,
MXFP4 K=32768), with amdgcnas in its last column
([intra_wave README](../../kernels/gemm/intra_wave/README.md)). Measured here the same way,
force-agpr reproduces those numbers, which is also a check that the pin-aware llirSched leaves
unpinned kernels alone:

| kernel | source | `llir` | `llir + force-agpr` | `+ amdgcnas` |
|---|---|---|---|---|
| a8w8, K=16384 | measured here | 91.8% | **98.0%** | 99.2% |
| a8w8, K=16384 | tutorial | 91.6% | **98.1%** | 99.2% |
| a4w4 v1, K=32768 | measured here | 43.1% | **88.3%** | 94.6% |
| a4w4 v1, K=32768 | tutorial | 45.1% | **88.5%** | 93.7% |

`cd_regclass` at those same shapes: a8w8 95.8% with llirSched and 99.5% with amdgcnas on top (the best
of all a8w8 configurations measured); a4w4 v1 83.7% and 92.7%; v9 fp16 at K=8192 96.6% with amdgcnas
against force-agpr's 97.9%.

Absolute TFLOPS on this machine run about 30% below the published figures in every column, including
the unmodified `llir` and `+amdgcnas` ones, so that offset belongs to the machine and not to any
variant: the shapes (4096×4096) and the GPU (256 CUs, SPX) match, and ATT cycles ÷ kernel time put
these loops at ~2.0 GHz, while the published TFLOPS would need ~3 GHz at the same cycle counts. MFMA
efficiency is a ratio and independent of clock, which is why it reproduces exactly.

## `s_nop` generation and overhead

The pins add `s_nop 0` (1 wait state each) because LLVM's hazard recognizer
(`GCNHazardRecognizer.cpp`) cannot see inside inline asm and assumes the worst about it. Two rules
fire:

- **Partial-register writes (not a real hazard here).** On gfx950, an instruction that writes only
  part of a register (SDWA `dst_sel` or `op_sel` on the destination) must be followed by 1 wait
  state before the next instruction touching that register. LLVM assumes any inline asm might be
  such a write, so it pads between a pin and the next instruction that uses the pinned registers.
  The pins are empty, so this padding is never needed.
  - v9 without llirSched: 46 of the 47, between the D pin that ends one `mfma()` call and the C pin
    that starts the next call on the same accumulator, which LLVM's machine scheduler hoists right up
    to it.
  - v9 with llirSched: 60 of the 90, between a C pin and the MFMA that reads it. The fences keep each
    C pin next to its own MFMA, and LLVM counts MFMAs as VALU instructions.
- **M0 before an LDS-DMA load (a real hazard).** Writing `m0` (`s_mov_b32 m0, ...`) and then issuing
  a `buffer_load ... lds` needs 1 wait state. Inline asm doesn't count toward that distance, so when
  only a pin sits between the two, LLVM has to add the nop. This accounts for the other 30.

**Overhead: negligible.** Each `s_nop 0` takes 4 cycles of the wave's issue time, 360 cycles per
iteration with llirSched, 8% of the loop's summed latency in ATT. But deleting the 60 fake-hazard
nops from the assembly (keeping the 30 real ones) does not speed the loop up: MFMA efficiency stays
at 90.8% against 91.2%, because the MFMAs' stall time grows by the same amount. Cold end-to-end it
gains about 0.5%. The nops sit in time the wave would otherwise spend waiting on the MFMA pipe.

**Removing them anyway.** Two options, neither done yet:

- In LLVM, teach the hazard recognizer that an inline asm with an empty string emits no instruction.
  That removes the partial-register-write padding (46 without llirSched, 60 with it). The 30 M0 nops
  stay unless the schedule puts a real instruction between the `m0` write and the load.
- In Triton, drop redundant pins. A C pin whose input is already the previous call's D pin in the
  same register class changes nothing.

## What ATT shows

The MFMA-efficiency gap to force-agpr on v9 comes from the per-tile fences, not from the `s_nop`s.
It is repeatable across dispatches:

| traced dispatch | `cd_regclass` + llirSched | force-agpr + llirSched |
|---|---|---|
| 15 | 4491 cycles/iter, 91.2% | 4226, 96.9% |
| 25 | 4490, 91.2% | 4212, 97.3% |
| 40 | 4494, 91.1% | 4239, 96.6% |

### Where the gap comes from

In the LLVM IR llirSched hands to the backend, 128 of the loop's 192 dependent MFMA pairs are already
consecutive — for the unpinned kernel as well, the same order. Without pins LLVM's schedulers pull
every one of those pairs apart again in the final assembly; with pins the per-tile fences stop that,
and 123 of the 192 stay consecutive.

A fence only forbids *crossing* it, so this is not simply the IR order being frozen. Each fenced
window here holds a C pin, the tile's first K-step and one memory op, and inside the window the
pre-RA scheduler still hoists the memory op above the MFMA — which pushes that MFMA to the window's
end, right against the next window's dependent MFMA. In the IR the tile reads `pin, mfma k0, ds_read,
[fence], mfma k1`; built with `-enable-misched=false` the assembly keeps that (`mfma, ds_read, mfma`),
and with the scheduler on it becomes `ds_read, mfma, mfma`. That hoisting is worth about a point
(91.9% vs 90.9% below). The other five points are the post-RA scheduler, which the fences stop from
interleaving independent tiles at all.

Separating the passes shows it (v9 16×16 with `cd_regclass` pins throughout; the assembly built by
`llc` from llirSched's IR and measured with the same ATT recipe):

| pin fences | pre-RA `misched` | loop ins | copies | `s_nop` | adjacent dependent pairs | MFMA eff | cycles/iter |
|---|---|---|---|---|---|---|---|
| every tile (shipped) | on | 519 | 0 | 90 | 123/192 | 90.9% | 4508 |
| every tile | off | 520 | 0 | 90 | 118/192 | 91.9% | 4456 |
| none | on | 643 | **200** | 24 | 5/184 | 83.2% | 4921 |
| none | off, and post-RA off too | 585 | 0 | 152 | 58/192 | 93.2% | 4395 |
| **none** | **off** | **428** | **0** | **8** | **0/192** | **96.9%** | **4229** |
| *force-agpr, no pins* | on | 422 | 0 | 0 | 0/192 | 96.5% | 4245 |

The pins themselves cost nothing: drop the fences and stop only the **pre-RA** scheduler from touching
the pinned MFMAs, and the **post-RA** scheduler spreads the dependent MFMAs, the hazard `s_nop`s all
but disappear and `cd_regclass` lands on force-agpr's number (96.9% vs 96.5%, 4229 vs 4245
cycles/iter, in a loop of 428 instructions against force-agpr's 422). The fences are only needed
against the pre-RA scheduler — without them it reorders the pinned MFMAs and register allocation
strands 200 copies in the loop (row 3). But `sched.barrier(0)` blocks *both* schedulers, so it also
blocks the pass that does the spreading. That is the 6 points.

The `s_nop`s are a symptom of the same adjacency rather than a separate cost (90 with the fences, 8
without), which is also why deleting the 60 that guard the fake hazard changed nothing.

Three things that do not close the gap:

- **Fewer fences.** One per 2 / 4 / 8 tiles instead of per tile leaves 80 / 132 / 164 copies in the
  loop (994 / 969 / 951 TFLOPS warm, against 1063 for one per tile).
- **A masked fence.** `sched.barrier` with every class except MFMA allowed to cross behaves like no
  fence at all: 200 copies.
- **Emitting the K-steps apart.** Reordering llirSched's MFMA list so that a tile's K-steps are
  separated (round-robin over accumulator chains, at any group size) does remove the adjacency, but
  then register allocation cannot hold the pins: 332–452 copies, 84–106 MFMAs back in VGPR form,
  810–846 TFLOPS.

### Fencing before the anchor instead

llirSched's output is the same with and without pins: dropping the pins and fences, the loop's MFMAs,
LDS reads, async copies and stores come in the same order, and all 96 anchor fences sit at the same
positions. The pinned version only adds a fence after each of the 128 D pins.

The anchor fence goes *after* its anchor, so a window runs from one tile's MFMA to the next anchor
(`mfma, ds_read | mfma`), and the scheduler is free to hoist that ds_read in front of the MFMA. Without
pin fences that is harmless: the next window also starts with a hoisted load, so a load still lands
between every two MFMAs. With them, 120 of the 192 MFMA windows hold no memory op at all. Putting the
anchor fence *before* the anchor (`LLIR_SCHED_ANCHOR_FENCE=before` in an experimental build of the
plugin) makes each window `ds_read, mfma` with the load already in front, and the D-pin fence then
coincides with the next anchor's fence. Two ATT runs each, v9 16×16:

| anchor fence | variant | loop ins | `s_nop` | adjacent dependent pairs | MFMA eff |
|---|---|---|---|---|---|
| after (shipped) | force-agpr | 423 | 0 | 0/192 | 96.7% / 96.4% |
| before | force-agpr | 420 | 0 | 0/192 | 97.2% / 97.2% |
| after (shipped) | `cd_regclass` | 520 | 90 | 123/192 | 90.9% / 91.3% |
| before | `cd_regclass` | 528 | 98 | 59/192 | **95.6% / 95.2%** |
| before, pin `s_nop`s deleted | `cd_regclass` | 429 | 0 | 59/192 | 95.9% / 95.9% |

All five keep 0 copies in the loop and all 256 MFMAs in AGPR form. Fence-before recovers most of
`cd_regclass`'s gap and helps force-agpr slightly too; the pin fences are still needed (without them:
316 copies).

The remaining 1.6 points against force-agpr are 56 tiles that still sit alone in a window
(`pin, s_nop, mfma, mfma, pin`). They come from the blocks of 4 MFMAs llirSched places after each
async copy: 4 consecutive MFMAs from 2-step chains always contain a whole tile, and with no memory op
inside the block only interleaving two tiles would separate its K-steps. Doing that
(`LLIR_SCHED_PIN_LOCAL_SPREAD=1`, alternating chains within each block) strands the pins again: 312
copies, 52 MFMAs back in VGPR form. So pinned tiles apparently must not overlap each other, and that
is what the last points cost.

The `s_nop` row emulates LLVM's hazard recognizer ignoring the pins (see Limitations): it deletes
each `s_nop 0` that sits between an empty asm and an MFMA. It is worth a few tenths of a point here,
but it shortens the loop to force-agpr's length.

Closing the gap completely would need either a fence that holds the pins through the pre-RA
scheduler without freezing the post-RA one — LLVM has no such barrier today, and
`-enable-misched=false` is process-wide, though so is force-agpr's own `amdgpu-mfma-vgpr-form=0` — or
the register class attached to the MFMA at instruction selection instead of as an asm hint, which is
what force-agpr and the in-progress `RewriteMFMAFormStage` do.

Note that this per-CU loop metric does not decide end-to-end time on its own: on v9 16×16
`cd_regclass` trails force-agpr by 5 points of MFMA efficiency but is 1% ahead in cold TFLOPS, while
on a4w4 v0 it leads by 9 points and is 2% behind. The 96.9% variant above is another case — it gains
6 points of MFMA efficiency over the shipped one and runs the same cold (1013 vs 1016 TFLOPS), the
loop's extra MFMA-pipe idle time being covered by memory waits. Judge a change by both.

## LLVM fix for the pin `s_nop`s

A local LLVM change (`[AMDGPU] Don't assume a hazard for empty inline asm`, on top of `b010a18d`)
makes `GCNHazardRecognizer` skip inline asm with an empty asm string, both as a hazard producer and
as a consumer. A hazard carried by the pinned value is still found at the instruction that produced
it, since wait-state counting already walks past inline asm. It adds the lit test
`llvm/test/CodeGen/AMDGPU/inlineasm-empty-no-hazard.ll`, which fails on the stock LLVM.

Triton (this experiment's branch) was rebuilt against it, with llirSched rebuilt against the same
headers. Checks:

- The AMDGPU CodeGen lit suite passes (4949 tests, 0 failures).
- The unpinned force-agpr kernels (v9 16×16 and 32×32, a8w8, a4w4 v0 and v1) compile to the same
  assembly as with the stock LLVM.
- In the pinned kernels every remaining loop `s_nop` guards the real `m0` → LDS-DMA hazard.

`cd_regclass` + llirSched, 6 interleaved ATT runs per row (median MFMA efficiency), stock → patched
LLVM:

| kernel | anchor fence | loop instructions | loop `s_nop` | MFMA eff |
|---|---|---|---|---|
| v9 16×16 | after (shipped) | 519 → 429 | 90 → 0 | 91.2% → 93.0% |
| v9 16×16 | before | 527 → 431 | 98 → 0 | 95.3% → 95.1% |
| v9 32×32 | after | 318 → 304 | 24 → 8 | 92.8% → 93.0% |
| v9 32×32 | before | 306 → 296 | 16 → 0 | 93.8% → 94.7% |
| a8w8 | after | 450 → 328 | 154 → 32 | 95.0% → 95.6% |
| a8w8 | before | 418 → 296 | 122 → 0 | 95.3% → 96.4% |
| a4w4 v0 | after | 585 → 491 | 126 → 32 | 70.5% → 71.7% |
| a4w4 v0 | before | 573 → 459 | 114 → 0 | 71.6% → 73.9% |
| a4w4 v1 | after | 560 → 472 | 96 → 8 | 81.9% → 84.7% |
| a4w4 v1 | before | 566 → 470 | 102 → 0 | 83.1% → 85.3% |

All rows keep 0 copies in the loop and every loop MFMA in AGPR form. On v9 with the shipped fence
placement the real fix gains more than deleting the `s_nop`s from the stock assembly did (+1.8 against
+0.3 points), because the hazard recognizer no longer reshapes the schedule around the pins. Cold
end-to-end kernel time (`--kernel-trace`, median of 200 dispatches, 3 runs) moves by less than 2% and
within run-to-run spread on every kernel.

The a8w8 and a4w4 traces are noisy on this shared machine: single runs occasionally drop 10–20
points for stock and patched alike, and the same assembly re-run gives both results. Read these
kernels by the median, not a single run. Builds, patch and raw results:
`/data/llvm-hazardfix/INFO.txt`.

## Best MFMA efficiency

All configurations at the tutorial's headline shapes (fp16 K=8192, BF8 K=16384, MXFP4 K=32768), 6
interleaved ATT runs each, median MFMA efficiency (range in brackets). llirSched is on in every row.
force-agpr uses the stock LLVM, since the LLVM fix leaves its assembly unchanged; `cd_regclass` uses the
LLVM with the fix. "Fence before" is the experimental `LLIR_SCHED_ANCHOR_FENCE=before` (with
`LLIR_SCHED_PIN_FENCE=1` for `cd_regclass`); "after" is the shipped plugin. Every row has 0 copies in
the loop and every loop MFMA in AGPR form.

| kernel | best setting | MFMA eff | tutorial's best (force-agpr + llirSched + amdgcnas) |
|---|---|---|---|
| a16w16 v9 | `cd_regclass` + LLVM fix + fence before + amdgcnas | **98.4%** (98.1–98.6) | 97.7% (97.4–97.9) |
| a8w8 | `cd_regclass` + LLVM fix + amdgcnas, either fence | **99.5%** (99.5–99.5) | 99.2% (99.2–99.2) |
| a4w4 v1 | `cd_regclass` + LLVM fix + fence before + amdgcnas | **93.7%** (92.9–94.5) | 93.7% (92.7–93.8) |

All eight combinations:

| placement | anchor fence | amdgcnas | v9 | a8w8 | a4w4 v1 |
|---|---|---|---|---|---|
| force-agpr | after (tutorial) | off | 96.7% (96.3–96.9) | 97.9% (97.6–98.2) | 87.8% (86.8–88.7) |
| force-agpr | after (tutorial) | on | 97.7% (97.4–97.9) | 99.2% (99.2–99.2) | **93.7%** (92.7–93.8) |
| force-agpr | before | off | 97.2% (97.0–97.8) | 97.7% (97.7–97.7) | 88.2% (88.1–88.3) |
| force-agpr | before | on | 97.1% (96.7–97.1) | 98.3% (98.2–98.3) | 92.5% (91.7–92.5) |
| `cd_regclass` + LLVM fix | after | off | 93.1% (92.7–93.3) | 96.7% (96.6–96.7) | 86.2% (85.9–86.3) |
| `cd_regclass` + LLVM fix | after | on | 96.6% (96.5–96.6) | **99.5%** (99.5–99.5) | 90.9% (90.8–91.7) |
| `cd_regclass` + LLVM fix | before | off | 95.3% (94.8–95.4) | 96.9% (96.9–96.9) | 87.6% (87.1–87.8) |
| `cd_regclass` + LLVM fix | before | on | **98.4%** (98.1–98.6) | **99.5%** (99.5–99.5) | **93.7%** (92.9–94.5) |

- amdgcnas is the largest single gain in every configuration.
- Fence before helps `cd_regclass` everywhere, but hurts force-agpr once amdgcnas is on.
- Without amdgcnas, force-agpr stays ahead of `cd_regclass`.
- End-to-end kernel time was not measured for this table; the earlier sweeps moved it by less than 2%.

The winning setting needs three things outside the tutorial: the Triton `cd_regclass` branch, the local
LLVM commit, and the experimental plugin (`/data/llvm-hazardfix/plugins/`). Raw runs:
`/data/llvm-hazardfix/results/sweep_best.txt`.

### Without amdgcnas

`cd_regclass` + LLVM fix + llirSched with fence before, against the tutorial's llirSched + force-agpr
(from the table above):

| kernel | tutorial: llirSched + force-agpr | `cd_regclass` + LLVM fix + llirSched, fence before | difference |
|---|---|---|---|
| a16w16 v9 | 96.7% (96.3–96.9), 4238 cycles/iter | 95.3% (94.8–95.4), 4300 cycles/iter | −1.4 points |
| a8w8 | 97.9% (97.6–98.2), 4184 | 96.9% (96.9–96.9), 4226 | −1.0 |
| a4w4 v1 | 87.8% (86.8–88.7), 4665 | 87.6% (87.1–87.8), 4676 | −0.2, within the spread |

The remaining gap is the tiles that llirSched places whole inside a block of MFMAs with no memory op,
which the pin fences keep from interleaving (see
[Fencing before the anchor instead](#fencing-before-the-anchor-instead)).

### Without llirSched and amdgcnas

With LLVM's own scheduler only. Same shapes, 6 interleaved runs each:

| kernel | placement | loop ins | copies in loop | AGPR / VGPR-form MFMAs | loop `s_nop` | MFMA eff | cycles/iter |
|---|---|---|---|---|---|---|---|
| a16w16 v9 | unpinned (tutorial kernel) | 462 | 36 | 52 / 204 | 1 | 70.3% (70.3–70.3) | 5828 |
| a16w16 v9 | force-agpr | 421 | 0 | 256 / 0 | 1 | 70.7% (70.6–70.8) | 5790 |
| a16w16 v9 | `cd_regclass` + LLVM fix | 421 | 0 | 256 / 0 | 1 | **71.8%** (71.7–71.8) | 5708 |
| a8w8 | unpinned | 324 | 33 | 32 / 96 | 0 | 68.0% (68.0–68.2) | 6021 |
| a8w8 | force-agpr | 299 | 0 | 128 / 0 | 9 | 70.7% (70.6–70.8) | 5794 |
| a8w8 | `cd_regclass` + LLVM fix | 296 | 0 | 128 / 0 | 6 | **72.3%** (72.2–72.4) | 5666 |
| a4w4 v1 | unpinned | 560 | 79 | 48 / 208 | 8 | 63.4% (63.3–63.8) | 6462 |
| a4w4 v1 | force-agpr | 483 | 0 | 256 / 0 | 15 | 66.5% (66.5–66.6) | 6158 |
| a4w4 v1 | `cd_regclass` + LLVM fix | 477 | 0 | 256 / 0 | 6 | **68.1%** (68.0–68.2) | 6016 |

With the stock scheduler, `cd_regclass` gets the same register placement as force-agpr (0 copies, all
MFMAs in AGPR form) and is 1.1–1.6 points ahead of it on all three kernels, with no spread overlap. Why
it is ahead was not investigated; on v9 the two loops have the same instruction and `s_nop` counts. All
three configurations stay near 70%, 20–25 points below any llirSched configuration: without llirSched
the schedule, not the register placement, limits MFMA efficiency. (a8w8 reports a tolerance mismatch in
every configuration, including the unmodified tutorial kernel: max error 0.5 against a fixed 0.1
tolerance, an fp8 accumulation effect at K=16384.) Raw runs: `/data/llvm-hazardfix/results/sweep_nollir.txt`.

## Limitations

- **The pins never change instruction selection.** At this LLVM, AGPR-form MFMA needs both
  `amdgpu-mfma-vgpr-form=0` and `amdgpu-agpr-alloc`, so the MFMAs are selected in VGPR form, the pins
  add AGPR↔VGPR copies, and LLVM's copy-removal passes take them out after register allocation. They
  do so only when each pin sits next to its MFMA and the pinned MFMAs are not reordered.
- **Spills at the register limit.** Because register allocation still sees VGPR-form MFMAs, it has to
  find VGPRs for values in transit before the copies are removed. On a4w4 v0 (256 VGPRs + 256 AGPRs)
  that spills 60 VGPRs without llirSched and 28 with it, all after the hot loop; force-agpr does not
  spill there.
- **All-AGPR is not always best.** On a4w4 v1 without llirSched, force-agpr's MFMA efficiency (66.2%)
  beats both the unpinned kernel (64.0%) and `cd_regclass` (64.2%) only slightly, and on v9 without
  llirSched none of the three differ much: the schedule dominates.
- **llirSched's fences serialize dependent MFMAs**, worth ~6 points of MFMA efficiency on v9; the
  pins are what force the fences (see [Where the gap comes from](#where-the-gap-comes-from)).
- **`s_nop` padding.** `GCNHazardRecognizer::checkVALUHazards` (and `checkInlineAsmHazards`) assume
  any inline asm has a dst-sel forwarding hazard, so an MFMA reading a pinned tuple right after its
  pin gets 1 wait state. The pin emits nothing. Fixed in a local LLVM build, see
  [LLVM fix for the pin `s_nop`s](#llvm-fix-for-the-pin-s_nops).
- Measured on v9 (16×16×32 and 32×32×16 fp16), a8w8 and a4w4 v0/v1, at one shape
  (4096×4096×8192) each.

## Performance impact of the LLVM fix (llvm/llvm-project#223526)

The fix as in [llvm/llvm-project#223526](https://github.com/llvm/llvm-project/pull/223526) (head
`c9c0db98e`), ported onto LLVM `b010a18d`, the version `gfx950-tutorial-v2.1` uses. "Before" is stock
`b010a18d`; "after" is the same LLVM with the fix, with Triton rebuilt against it.

- Every MFMA accumulator is pinned with `cd_regclass="a"`, and no force-agpr settings are used.
- gfx950, 4096×4096×8192, cold rotating inputs.
- Loop `s_nop`: counted in the generated assembly.
- MFMA efficiency: `rocprofv3 --att`, median of 6 interleaved runs (min–max in brackets).
- TFLOPS: `rocprofv3 --kernel-trace`, median kernel time over 200 dispatches, then the median of 3
  runs (min–max in brackets).

### With llirSched (anchor fence before)

| kernel | loop `s_nop` (before → after) | MFMA eff, before | MFMA eff, after | TFLOPS, before | TFLOPS, after | TFLOPS change |
|---|---|---|---|---|---|---|
| a16w16 v9 (16×16) | 98 → 0 | 95.2% (95.0–95.8) | 95.0% (94.8–95.3) | 1029 (1021–1036) | 1034 (1029–1035) | +0.5% |
| a8w8 | 122 → 0 | 95.4% (93.6–95.6) | 96.4% (96.0–96.8) | 2036 (2035–2050) | 2048 (2030–2070) | +0.6% |
| a4w4 v1 | 102 → 0 | 83.4% (75.4–84.5) | 85.5% (69.2–86.8) | 3054 (2998–3064) | 3085 (3054–3085) | +1.0% |

### Without llirSched

| kernel | loop `s_nop` (before → after) | MFMA eff, before | MFMA eff, after | TFLOPS, before | TFLOPS, after | TFLOPS change |
|---|---|---|---|---|---|---|
| a16w16 v9 (16×16) | 47 → 1 | 71.8% (71.8–71.9) | 71.8% (71.7–71.9) | 915 (912–918) | 914 (914–917) | −0.0% |
| a8w8 | 49 → 6 | 72.4% (71.8–72.6) | 72.4% (72.3–72.6) | 1867 (1836–1870) | 1850 (1844–1879) | −0.9% |
| a4w4 v1 | 32 → 6 | 68.3% (67.8–68.6) | 67.7% (66.9–67.9) | 2857 (2851–2869) | 2843 (2843–2893) | −0.5% |

- Every configuration keeps 0 copies in the loop and every loop MFMA in AGPR form.
- Every `s_nop` left after the fix guards the real `m0` → LDS-DMA hazard.
- TFLOPS moves by −0.9% to +1.0%, within run-to-run spread.
- With llirSched, MFMA efficiency rises on a8w8 (+1.0) and a4w4 v1 (+2.1) and is flat on v9. Without
  llirSched the schedule limits efficiency, and removing the `s_nop`s changes nothing.
- a4w4 v1 ATT runs occasionally drop to 66–78% regardless of the build: swapping the generated assembly
  between builds reproduces the drop with either version. Read that kernel by its median.
- Raw runs and scripts: `/data/llvm-hazardfix/results/` (`perf_impact_pr223526.md`, `perf_att.txt`,
  `perf_kt.txt`).

