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

The fences stop LLVM's machine scheduler from interleaving tiles, so all 128 second-K-step MFMAs in
the loop issue right behind the MFMA whose result they read and wait for it. With force-agpr only 33
of the 128 do; the rest have 1 to 5 independent MFMAs in between. Interleaving two tiles' K-steps
before fencing them should close the gap.

Note that this per-CU loop metric does not decide end-to-end time on its own: on v9 16×16
`cd_regclass` trails force-agpr by 5 points of MFMA efficiency but is 1% ahead in cold TFLOPS, while
on a4w4 v0 it leads by 9 points and is 2% behind. Judge a change by both.

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
- **llirSched's fences serialize dependent MFMAs** (see [What ATT shows](#what-att-shows)).
- **`s_nop` padding**, negligible in practice (see above).
- Measured on v9 (16×16×32 and 32×32×16 fp16), a8w8 and a4w4 v0/v1, at one shape
  (4096×4096×8192) each.
