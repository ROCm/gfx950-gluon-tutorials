# v9 with MFMA accumulators pinned to AGPRs

This experiment pins every MFMA accumulator of the
[a16w16 v9 kernel](../../kernels/gemm/intra_wave/a16w16/v9_beyond_hotloop/matmul_kernel.py) to AGPRs
with inline-asm register-class constraints. The question is whether this can replace `force-agpr`,
which is the pair of LLVM settings `amdgpu-mfma-vgpr-form=0` (process-wide LLVM option) +
`amdgpu-agpr-alloc=256` (kernel function attribute); see
[docs/performance_philosophy.md](../../docs/performance_philosophy.md).

It uses two extensions:

- **`cd_regclass`** (Triton): `gl.amd.cdna4.mfma(a, b, acc, cd_regclass="a")` and the same argument
  on `gl.amd.cdna4.mfma_scaled` make the compiler pin each MFMA's C and D next to the MFMA
  instruction.
- **llirSched** (this repo's [plugins/llir_scheduler](../../plugins/llir_scheduler)): the plugin keeps
  those pins attached to their MFMAs when it reorders the loop.

**Background.** An earlier attempt placed the pins from Gluon source instead, wrapping the
accumulator before and after every `mfma()` call with `inline_asm_elementwise`, using
[triton-lang/triton#10337](https://github.com/triton-lang/triton/pull/10337) (Lei Zhang's branch
`antiagainst/triton:pr-10337-amd-register-classes`). It does not work on v9. A Gluon-level pin covers
the whole accumulator tensor, so each `mfma()` call becomes 16 pins, then 32 MFMAs, then 16 more pins.
The loop kept 480 copy instructions and only 176 of its 256 MFMAs in AGPR form (849 TFLOPS vs 943 for
v9), and pinning only D spilled 147 VGPRs. Lei's example on that branch only has one MFMA per `mfma()`
call (a single wave, one 16×16×32 or 32×32×16 tile, no register pressure), so a tensor-level pin sits
next to its MFMA by construction and the problem cannot show up there. Those files are in this
directory's git history (commit `efaa3ad`); `cd_regclass` does not need PR #10337.

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
- [`run.py`](run.py): correctness check and `do_bench` timing, 4096×4096×8192 fp16.
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
  that reordering left 200 copies in the loop, because LLVM's post-RA copy-removal pass could not
  find a free AGPR for 17 of the MFMA chains. The fences have a cost, see
  [What ATT shows](#what-att-shows).

Kernels without pins get byte-identical output from the plugin before and after this change.

## Results

**The number to look at is the copy instructions inside the hot loop**: `v_accvgpr_read`,
`v_accvgpr_write` and `v_accvgpr_mov`, which move accumulator values between AGPRs and VGPRs. A clean
kernel keeps the accumulators in AGPRs across the whole loop and has none. The hot loop is the loop
body that contains the MFMAs, from its label to its backward branch.

All rows are compiled and run in-process with the Triton build above; llirSched is loaded with
`LLVM_PASS_PLUGIN_PATH` and `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1`. TFLOPS are only a rough
reference: `triton.testing.do_bench` (warm cache, not the tutorial's cold-cache rocprof methodology)
on gfx950, 4096×4096×8192. Run-to-run spread is about 1%. All outputs match their reference.

| # | `cd_regclass` | llirSched | copies in loop | AGPR-form MFMAs in loop (x/y) | VGPR spills | `s_nop` in loop (wait states) | `sched.barrier`s | TFLOPS (do_bench, reference) |
|---|---|---|---|---|---|---|---|---|
| 1 | no | no | 36 | 52/256 | 0 | 1 (1) | 0 | 942 |
| 2 | **yes** | no | **0** | 256/256 | 0 | 47 (47) | 0 | 949 |
| 3 | no | yes | 119 | 58/256 | 0 | 40 (169) | 96 | 963 |
| 4 | **yes** | yes | **0** | 256/256 | 0 | 90 (90) | 224 | 1065 |

For reference, force-agpr on the same build also gives 0 copies and 256/256 in both flows, at 939
TFLOPS without llirSched and 1062 with it. With cold caches (`rocprofv3 --kernel-trace`, 200
dispatches on rotating inputs, median of two runs) row 4 and llir + force-agpr are also level:
1022 vs 1019 TFLOPS (row 3: 934).

- **copies in loop**: number of `v_accvgpr_read` / `v_accvgpr_write` / `v_accvgpr_mov` instructions in the hot loop.
- **AGPR-form MFMAs in loop (x/y)**: x of the y MFMA instructions in the hot loop use the AGPR form,
  meaning their accumulator input C and result D are AGPRs, e.g.
  `v_mfma_f32_16x16x32_f16 a[252:255], v[88:91], v[96:99], a[252:255]`. The other y − x use the VGPR
  form, where C and D are VGPRs: `v_mfma_f32_16x16x32_f16 v[..], v[..], v[..], v[..]`. C and D always
  share one register class (the instruction has a single bit for both).
- **VGPR spills**: `.vgpr_spill_count` from the kernel metadata.
- **`s_nop` in loop (wait states)**: number of `s_nop` instructions in the hot loop, and the cycles
  they add in total (`s_nop N` waits N+1 cycles).
- **`sched.barrier`s**: `llvm.amdgcn.sched.barrier` calls in the LLVM IR, which llirSched inserts to
  pin its schedule.

### Other MFMA shapes and data types

The same pinning on a 32×32×16 variant of v9 (`instr_shape=[32, 32, 16]`, 128 MFMAs in the loop)
and on the tutorial's fp8 and MXFP4 GEMMs, all at 4096×4096×8192 with `cd_regclass="a"` on every
`mfma` / `mfma_scaled` call. With `cd_regclass`, every loop MFMA is in AGPR form in all of these.

| kernel | llirSched | copies in loop: unpinned / `cd_regclass` | `cd_regclass` VGPR spills | TFLOPS (do_bench): unpinned / `cd_regclass` / force-agpr |
|---|---|---|---|---|
| v9, 32×32×16 fp16 | no | 112 / **0** | 0 | 826 / 830 / 290 ¹ |
| | yes | 144 / **0** | 0 | 900 / 937 / 945 |
| a8w8, fp8 (e5m2) | no | 33 / **0** | 0 | 1982 / 2027 / 2025 |
| | yes | 49 / **0** | 0 | 2203 / 2258 / 2267 |
| a4w4 v0, MXFP4 | no | 68 / **0** | **60** | 3248 / 3025 / 3314 |
| | yes | 238 / **0** | **28** | 497 ² / 2411 / 2466 |
| a4w4 v1, MXFP4 | no | 79 / **0** | 0 | 3045 / 2475 / 2672 |
| | yes | 220 / **0** | 0 | 2165 / 2481 / 2470 |

¹ force-agpr spills 27 VGPRs on this shape. ² The unpinned kernel spills 186 VGPRs here.

## `s_nop` generation and overhead

The pins add `s_nop 0` (1 wait state each) because LLVM's hazard recognizer
(`GCNHazardRecognizer.cpp`) cannot see inside inline asm and assumes the worst about it. Two rules
fire:

- **Partial-register writes (not a real hazard here).** On gfx950, an instruction that writes only
  part of a register (SDWA `dst_sel` or `op_sel` on the destination) must be followed by 1 wait
  state before the next instruction touching that register. LLVM assumes any inline asm might be
  such a write, so it pads between a pin and the next instruction that uses the pinned registers.
  The pins are empty, so this padding is never needed.
  - Row 2: 46 of the 47, between the D pin that ends one `mfma()` call and the C pin that starts the
    next call on the same accumulator, which LLVM's machine scheduler hoists right up to it.
  - Row 4: 60 of the 90, between a C pin and the MFMA that reads it. The fences keep each C pin
    next to its own MFMA, and LLVM counts MFMAs as VALU instructions.
- **M0 before an LDS-DMA load (a real hazard).** Writing `m0` (`s_mov_b32 m0, ...`) and then issuing
  a `buffer_load ... lds` needs 1 wait state. Inline asm doesn't count toward that distance, so when
  only a pin sits between the two, LLVM has to add the nop. This accounts for the other 30 in row 4.

**Overhead.** Each `s_nop 0` takes 4 cycles of the wave's issue time: 360 cycles per iteration in
row 4, 8% of the loop's latency in ATT. It costs almost nothing, though. Deleting the 60 fake-hazard
nops from row 4's assembly (and keeping the 30 real ones) leaves the ATT loop time unchanged: the
MFMAs' stall time grows by the same amount. Cold end-to-end it gains about 0.5% (1027 vs 1022 TFLOPS).

**Removing them.** Two options, neither done yet:

- In LLVM, teach the hazard recognizer that an inline asm with an empty string emits no instruction.
  That removes the partial-register-write padding (46 in row 2, 60 in row 4). The 30 M0 nops stay
  unless the schedule puts a real instruction between the `m0` write and the load.
- In Triton, drop redundant pins. A C pin whose input is already the previous call's D pin in the
  same register class changes nothing, and removing it removes the row-2 pairs.

## What ATT shows

ATT traces for row 4 and for llir + force-agpr are in `/data`:
`att_a16w16-v9_cd_regclass_llir-sched_4096x4096x8192_fp16_mi350x_rotating_v2.1-cd-regclass_20260911`
and `att_a16w16-v9_beyond_hotloop_llir-sched_force-agpr_4096x4096x8192_fp16_mi350x_rotating_v2.1-cd-regclass_20260911`
(cold rotating inputs, CU 0 of every shader engine, `scripts/process_json.py`).

| traced dispatch | row 4: cycles per iteration, MFMA efficiency | llir + force-agpr |
|---|---|---|
| 15 | 4491, 91.2% | 4226, 96.9% |
| 25 | 4490, 91.2% | 4212, 97.3% |
| 40 | 4494, 91.1% | 4239, 96.6% |

On the traced CU the pinned loop is consistently about 6% longer, and the cause is the per-tile
fences, not the `s_nop`s. The fences stop LLVM's machine scheduler from interleaving tiles, so all 128
second K-step MFMAs in the loop issue right behind the MFMA whose result they read and wait for it.
With force-agpr only 33 of the 128 do; the rest have 1 to 5 independent MFMAs in between. End to
end the two still take the same time (cold and warm), and why the traced CU's gap does not show up
there is not understood yet. Interleaving two tiles' K-steps before fencing them should remove the
gap.

## Limitations

- **The pins never change instruction selection.** At this LLVM, AGPR-form MFMA needs both
  `amdgpu-mfma-vgpr-form=0` and `amdgpu-agpr-alloc`, so the MFMAs are selected in VGPR form, the pins
  add AGPR↔VGPR copies, and LLVM's copy-removal passes take them out after register allocation. They
  do so only when each pin sits next to its MFMA and the pinned MFMAs are not reordered.
- **Spills at the register limit.** Because register allocation still sees VGPR-form MFMAs, it has to
  find VGPRs for values in transit before the copies are removed. On a4w4 v0 (256 VGPRs + 256 AGPRs)
  that spills 60 VGPRs without llirSched and 28 with it, all after the hot loop; force-agpr does not
  spill there.
- **All-AGPR is not always best.** On a4w4 v1 without llirSched, keeping every accumulator in AGPRs
  (`cd_regclass` 2475, force-agpr 2672) is slower than the compiler's own mix (3045).
- **llirSched's fences serialize dependent MFMAs** (see [What ATT shows](#what-att-shows)).
- **`s_nop` padding**, mostly harmless (see above).
- Measured on v9 (16×16×32 and 32×32×16 fp16), a8w8 and a4w4 v0/v1 only, and with warm-cache
  `do_bench` except where cold numbers are given.
