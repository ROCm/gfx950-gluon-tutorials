# v9 with MFMA accumulators pinned to AGPRs

This experiment takes the [a16w16 v9 kernel](../../kernels/gemm/intra_wave/a16w16/v9_beyond_hotloop/matmul_kernel.py)
and pins every MFMA accumulator to AGPRs with inline-asm register-class constraints. The question
is whether this can replace `force-agpr`, which is the pair of LLVM settings
`amdgpu-mfma-vgpr-form=0` (process-wide LLVM option) + `amdgpu-agpr-alloc=256` (kernel function
attribute); see [docs/performance_philosophy.md](../../docs/performance_philosophy.md).

Two ways of placing the pins were tried:

1. **From Gluon source** ([`matmul_kernel.py`](matmul_kernel.py)): wrap the accumulator before and
   after every `mfma()` call with `inline_asm_elementwise`, using
   [triton-lang/triton#10337](https://github.com/triton-lang/triton/pull/10337). This does not work
   on v9 (see [Approach 1](#approach-1-pins-from-gluon-source)).
2. **In the MFMA lowering** ([`matmul_kernel_cd_regclass.py`](matmul_kernel_cd_regclass.py)): a new
   `cd_regclass` argument on `gl.amd.cdna4.mfma`, with which the compiler pins C and D next to each
   MFMA instruction. Together with an llirSched extension, this is the version the
   [results](#results) are about.

Lei Zhang's example on the PR #10337 branch
([`test_cdna4_mfma_register_classes.py`](https://github.com/antiagainst/triton/blob/pr-10337-amd-register-classes/test_cdna4_mfma_register_classes.py))
only has one MFMA. It runs a single wave, and each `mfma()` call is exactly one MFMA instruction
(the tensors are one 16×16×32 or 32×32×16 tile), repeated 1 or 8 times in a loop, with no register
pressure. v9 issues 32 MFMA instructions per `mfma()` call and 256 in the hot loop, close to the
512-register limit.

## Triton compiler

- Approach 1: [`antiagainst/triton` branch `pr-10337-amd-register-classes`](https://github.com/antiagainst/triton/tree/pr-10337-amd-register-classes)
  at `a26d06a750df8e149b6b32a3e113b4d4b31ae35a`: PR #10337 rebased onto triton `main` at
  `4a15f415d8ac4f830ce788c6fca3a5b3b29908e7`, plus a test/doc commit on top.
- Approach 2: branch `amd-mfma-cd-regclass` at `0d9e22ade4cb668db0ca96bee2186d57a625d7fc`, one
  commit on top of `a26d06a7` that adds `cd_regclass` (see
  [the `cd_regclass` extension](#the-cd_regclass-extension)).
- LLVM `b010a18d2b648cab83c83967ff26b8fde11acdc6` (the same pin as `gfx950-tutorial-v2.1`).
- Built with `TRITON_BUILD_PROTON=OFF TRITON_APPEND_CMAKE_ARGS=-DTRITON_BUILD_UT=OFF pip install --no-build-isolation .`

Default Triton flow: no llirSched plugin, no force-agpr, no amdgcnas. Unlike the v2.1 tag, these
upstream-based commits pass `amdgpu-use-amdgpu-trackers` on gfx950. llirSched means the plugin in
[plugins/llir_scheduler](../../plugins/llir_scheduler) on this branch (see
[the llirSched extension](#the-llirsched-extension)).

## Files

- [`matmul_kernel.py`](matmul_kernel.py): approach 1. A diff against v9 shows only the helper block
  and the 16 call sites.
- [`matmul_kernel_cd_regclass.py`](matmul_kernel_cd_regclass.py): approach 2. A diff against v9
  shows only the 16 call sites and the kernel renamed to `v9_cd_regclass`.
- [`run.py`](run.py): correctness check and `do_bench` timing, 4096×4096×8192 fp16; `--kernel`
  picks the kernel.
- [`ir_dumps/`](ir_dumps/): the LLVM IR (`.llir`) and gfx950 assembly (`.amdgcn`) Triton generated
  in the default flow: `v9_beyond_hotloop.*` for approach 1, `v9_cd_regclass.*` for approach 2.

```bash
python run.py                                     # approach 1
python run.py --kernel matmul_kernel_cd_regclass  # approach 2
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/dump python run.py  # regenerate ir_dumps/
```

## Approach 1: pins from Gluon source

```python
@gluon.jit
def reg_class(x, CLASS: gl.constexpr, PACK: gl.constexpr):
    VEC: gl.constexpr = PACK * x.dtype.primitive_bitwidth // 32
    return gl.inline_asm_elementwise("", "=" + CLASS + ",0", [x], dtype=x.dtype, is_pure=True,
                                     pack=PACK, operand_vec_sizes=[VEC], result_vec_sizes=[VEC])

@gluon.jit
def mfma_agpr(a, b, acc):
    acc = reg_class(acc, "a", 4)  # pin C
    acc = gl.amd.cdna3.mfma(a, b, acc)
    return reg_class(acc, "a", 4)  # pin D
```

`"=a,0"` is an empty asm whose output is tied to its input. It emits no instruction, but the value
must pass through an AGPR tuple at that point. With `pack=4`, each asm covers one 16×16 MFMA
accumulator (4 fp32 in 4 registers, passed as `i128`).

In the default flow this leaves 480 copy instructions in the hot loop, with only 176 of the 256
MFMAs in AGPR form (849 TFLOPS vs 943 for v9). Pinning only D spills 147 VGPRs (183 TFLOPS). The
reason is placement: a Gluon-level pin covers the whole accumulator tensor, so each `mfma()` call
becomes 16 asm (one per tile), then 32 MFMAs (16 tiles × 2 K-steps), then 16 asm pinning the
results. See `ir_dumps/v9_beyond_hotloop.llir`: the loop starts at line 370, with C pins at 701–806,
MFMAs at 888–919, and D pins plus the next call's C pins at 921–1080. Moving each of these asm next
to its MFMA, with nothing else changed, removes every loop copy. Approach 2 does that placement in
the compiler.

## The `cd_regclass` extension

```python
acc = gl.amd.cdna4.mfma(a, b, acc, cd_regclass="a")  # "a" = AGPRs, "v" = VGPRs
```

Changes on the Triton branch `amd-mfma-cd-regclass`:

- **Gluon `mfma`** (`gl.amd.cdna3.mfma`, re-exported as `gl.amd.cdna4.mfma`) takes an optional
  `cd_regclass` of `None`, `"a"` or `"v"`, and records it as the `amdg.cd_regclass` attribute on
  the `tt.dot`.
- **MFMA lowering** (`third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp`) wraps each MFMA
  tile's accumulator in the empty `"=a,0"` / `"=v,0"` asm: once before the tile's first K-step (C)
  and once after its last (D). The asm uses the MFMA's own vector type, so a 32×32 accumulator is
  pinned as one 16-register tuple, with no `i128` split and none of PR #10337's arguments needed.
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

TFLOPS are only a rough reference: `triton.testing.do_bench` (warm cache, not the tutorial's
cold-cache rocprof methodology) on gfx950, 4096×4096×8192 fp16. Run-to-run spread is about 1%. All
outputs match torch bit for bit.

| # | `cd_regclass` | llirSched | copies in loop | AGPR-form MFMAs in loop (x/y) | VGPR spills | `s_nop` in loop (wait states) | TFLOPS (do_bench, reference) |
|---|---|---|---|---|---|---|---|
| 1 | no | no | 36 | 52/256 | 0 | 1 (1) | 943 |
| 2 | **yes** | no | **0** | 256/256 | 0 | 47 (47) | 949 |
| 3 | no | yes ¹ | 119 | 58/256 | 0 | 40 (169) | 963 |
| 4 | **yes** | yes ¹ | **0** | 256/256 | 0 | 90 (90) | 1060 |

For reference, force-agpr also gives 0 copies and 256/256 in both flows, at 942 TFLOPS without
llirSched and 1049–1061 with it.

- **copies in loop**: number of `v_accvgpr_read` / `v_accvgpr_write` / `v_accvgpr_mov` instructions in the hot loop.
- **AGPR-form MFMAs in loop (x/y)**: x of the y MFMA instructions in the hot loop use the AGPR form,
  meaning their accumulator input C and result D are AGPRs, e.g.
  `v_mfma_f32_16x16x32_f16 a[252:255], v[88:91], v[96:99], a[252:255]`. The other y − x use the VGPR
  form, where C and D are VGPRs: `v_mfma_f32_16x16x32_f16 v[..], v[..], v[..], v[..]`. C and D always
  share one register class (the instruction has a single bit for both).
- **VGPR spills**: `.vgpr_spill_count` from the kernel metadata.
- **`s_nop` in loop (wait states)**: number of `s_nop` instructions in the hot loop, and the cycles
  they add in total (`s_nop N` waits N+1 cycles).

¹ The Triton builds used here cannot load the plugin in-process (they hide LLVM's symbols, and with
a plugin set they drop the target machine). So rows 3 and 4 run the plugin with
`opt -load-pass-plugin libLlirSched.so -passes=llir-sched` on Triton's optimized LLVM IR, compile
with the pinned `llc -O3 -mcpu=gfx950 -amdgpu-use-amdgpu-trackers`, and swap the result into the
launch. On unmodified IR this `llc` command reproduces Triton's assembly byte for byte.

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
  `amdgpu-mfma-vgpr-form=0` and `amdgpu-agpr-alloc`, so with either approach the MFMAs are selected
  in VGPR form and the pins add AGPR↔VGPR copies. LLVM's copy-removal passes take them all out only
  when each pin sits next to its MFMA and the pinned MFMAs are not reordered.
- **`cd_regclass` gets the placement right by construction.** It gives 0 copies and all MFMAs in
  AGPR form in the default flow (row 2), and with the llirSched extension also under llirSched
  (row 4), without either force-agpr setting.
- **Why the single-MFMA example looks fine.** When an `mfma()` call is a single MFMA instruction,
  as in the example on the PR branch, a tensor-level pin sits right next to that instruction by
  construction. The per-call grouping of approach 1 cannot show up there.
- **Caveats.**
  - `cd_regclass` covers `mfma` only, not `mfma_scaled` yet.
  - Only v9 (16×16×32 fp16) was measured.
