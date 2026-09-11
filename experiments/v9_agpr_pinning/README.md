# v9 with MFMA accumulators pinned to AGPRs

This experiment pins every MFMA accumulator of the
[a16w16 v9 kernel](../../kernels/gemm/intra_wave/a16w16/v9_beyond_hotloop/matmul_kernel.py) to AGPRs
with inline-asm register-class constraints. The question is whether this can replace `force-agpr`,
which is the pair of LLVM settings `amdgpu-mfma-vgpr-form=0` (process-wide LLVM option) +
`amdgpu-agpr-alloc=256` (kernel function attribute); see
[docs/performance_philosophy.md](../../docs/performance_philosophy.md).

It uses two extensions:

- **`cd_regclass`** (Triton): `gl.amd.cdna4.mfma(a, b, acc, cd_regclass="a")` makes the compiler pin
  each MFMA's C and D next to the MFMA instruction.
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

- Branch `gfx950-tutorial-v2.1-cd-regclass` at `98dc3eeb7e7bb2b2fbba3c224fefbfb12041ffa2`: one commit
  that adds `cd_regclass` on top of the tutorial's pin
  [`gfx950-tutorial-v2.1`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v2.1)
  (`e346b8a741cf5366a5bc9207a44c99d7e70fa828`), so everything the tutorial relies on is there too.
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
```

The commit changes three files:

- **Gluon `mfma`** (`gl.amd.cdna3.mfma`, re-exported as `gl.amd.cdna4.mfma`) takes an optional
  `cd_regclass` of `None`, `"a"` or `"v"`, and records it as the `amdg.cd_regclass` attribute on
  the `tt.dot`.
- **MFMA lowering** (`third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp`) wraps each MFMA
  tile's accumulator in an empty asm whose output is tied to its input (`"=a,0"` / `"=v,0"`): once
  before the tile's first K-step (C) and once after its last (D). The asm emits no instruction, but
  the value must pass through an AGPR (or VGPR) tuple at that point. It uses the MFMA's own vector
  type, so a 32×32 accumulator is pinned as one 16-register tuple.
- **Lit test** `test/TritonGPU/amd/mfma-cd-regclass.mlir`: pin order with two K-steps, no pins
  without the attribute, a 32×32 tile pinned as `vector<16xf32>`, and an error for invalid values.

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
  find a free AGPR for 17 of the MFMA chains.

Kernels without pins get byte-identical output from the plugin before and after this change.

## Results

**The number to look at is the copy instructions inside the hot loop**: `v_accvgpr_read`,
`v_accvgpr_write` and `v_accvgpr_mov`, which move accumulator values between AGPRs and VGPRs. A clean
kernel keeps the accumulators in AGPRs across the whole loop and has none. The hot loop is the loop
body that contains the MFMAs, from its label to its backward branch.

All rows are compiled and run in-process with the Triton build above; llirSched is loaded with
`LLVM_PASS_PLUGIN_PATH` and `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1`. TFLOPS are only a rough
reference: `triton.testing.do_bench` (warm cache, not the tutorial's cold-cache rocprof methodology)
on gfx950, 4096×4096×8192 fp16. Run-to-run spread is about 1%. All outputs match torch bit for bit.

| # | `cd_regclass` | llirSched | copies in loop | AGPR-form MFMAs in loop (x/y) | VGPR spills | `s_nop` in loop (wait states) | `sched.barrier`s | TFLOPS (do_bench, reference) |
|---|---|---|---|---|---|---|---|---|
| 1 | no | no | 36 | 52/256 | 0 | 1 (1) | 0 | 942 |
| 2 | **yes** | no | **0** | 256/256 | 0 | 47 (47) | 0 | 949 |
| 3 | no | yes | 119 | 58/256 | 0 | 40 (169) | 96 | 963 |
| 4 | **yes** | yes | **0** | 256/256 | 0 | 90 (90) | 224 | 1065 |

For reference, force-agpr on the same build also gives 0 copies and 256/256 in both flows, at 939
TFLOPS without llirSched and 1062 with it.

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

## `s_nop` generation and overhead

The pins add `s_nop 0` (1 wait state each) because LLVM's hazard recognizer
(`GCNHazardRecognizer.cpp`) cannot see inside inline asm and assumes the worst about it. Two rules
fire:

- **Partial-register writes.** On gfx950, an instruction that writes only part of a register (SDWA
  `dst_sel` or `op_sel` on the destination) must be followed by 1 wait state before the next
  instruction touching that register. LLVM assumes any inline asm might be such a write, so it pads
  between a pin and the next instruction that uses the pinned registers.
  - Row 2: 46 of the 47, between the D pin that ends one `mfma()` call and the C pin that starts the
    next call on the same accumulator, which LLVM's machine scheduler hoists right up to it.
  - Row 4: 60 of the 90, between a C pin and the MFMA that reads it. The fences keep each C pin
    next to its own MFMA, and LLVM counts MFMAs as VALU instructions.
- **M0 before an LDS-DMA load.** Writing `m0` (`s_mov_b32 m0, ...`) and then issuing a
  `buffer_load ... lds` needs 1 wait state. Inline asm doesn't count toward that distance, so a pin
  sitting between the two leaves the gap unfilled. This accounts for the other 30 in row 4.

None of these hazards is real, since an empty asm executes nothing; LLVM just can't tell an empty
asm from one that contains a real instruction.

**Overhead.** That is 47 cycles per loop iteration in the default flow and 90 with llirSched, next
to 256 MFMAs, with no visible effect in the TFLOPS column. With llirSched the pinned loop even
spends fewer cycles on `s_nop` than the unpinned one (90 vs 169): the unpinned loop's 119 copies
need longer nops (`s_nop 4`–`s_nop 6` before a `v_accvgpr_read` of a fresh MFMA result).

**Removing them.** Two options, neither done yet:

- In Triton, drop redundant pins. A C pin whose input is already the previous call's D pin in the
  same register class changes nothing, and removing it removes the row-2 pairs.
- In LLVM, teach the hazard recognizer that an inline asm with an empty string emits no instruction.
  That removes all of them.

## What is going on

- **The pins never change instruction selection.** At this LLVM, AGPR-form MFMA needs both
  `amdgpu-mfma-vgpr-form=0` and `amdgpu-agpr-alloc`, so the MFMAs are selected in VGPR form and the
  pins add AGPR↔VGPR copies. LLVM's copy-removal passes take them all out only when each pin sits
  next to its MFMA and the pinned MFMAs are not reordered.
- **`cd_regclass` gets the placement right by construction.** It gives 0 copies and all MFMAs in
  AGPR form in the default flow (row 2), and with the llirSched extension also under llirSched
  (row 4), without either force-agpr setting.
- **Caveats.**
  - `cd_regclass` covers `mfma` only, not `mfma_scaled` yet.
  - Only v9 (16×16×32 fp16) was measured.
