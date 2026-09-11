# v9 with MFMA accumulators pinned to AGPRs

This experiment takes the [a16w16 v9 kernel](../../kernels/gemm/intra_wave/a16w16/v9_beyond_hotloop/matmul_kernel.py)
and pins every MFMA accumulator to AGPRs with inline-asm register-class constraints. The question
is whether this can replace `force-agpr`, which is the pair of LLVM settings
`amdgpu-mfma-vgpr-form=0` (process-wide LLVM option) + `amdgpu-agpr-alloc=256` (kernel function
attribute); see [docs/performance_philosophy.md](../../docs/performance_philosophy.md).

Two ways of placing the pins are compared:

1. **From Gluon source** ([`matmul_kernel.py`](matmul_kernel.py)): wrap the accumulator before and
   after every `mfma()` call with `inline_asm_elementwise`, using
   [triton-lang/triton#10337](https://github.com/triton-lang/triton/pull/10337).
2. **In the MFMA lowering** ([`matmul_kernel_cd_regclass.py`](matmul_kernel_cd_regclass.py)):
   call `gl.amd.cdna4.mfma(a, b, acc, cd_regclass="a")`, and the compiler pins C and D next to each
   MFMA instruction.

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
  commit on top of `a26d06a7` that adds `cd_regclass` (Gluon `mfma`, `DotOpToLLVM/MFMA.cpp`, lit test
  `test/TritonGPU/amd/mfma-cd-regclass.mlir`).
- LLVM `b010a18d2b648cab83c83967ff26b8fde11acdc6` (the same pin as `gfx950-tutorial-v2.1`).
- Built with `TRITON_BUILD_PROTON=OFF TRITON_APPEND_CMAKE_ARGS=-DTRITON_BUILD_UT=OFF pip install --no-build-isolation .`

Default Triton flow: no llirSched plugin, no force-agpr, no amdgcnas. Unlike the v2.1 tag, these
upstream-based commits pass `amdgpu-use-amdgpu-trackers` on gfx950. The llirSched results use the
plugin source in [plugins/llir_scheduler](../../plugins/llir_scheduler) on this branch.

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

## Approach 2: `cd_regclass` in the MFMA lowering

```python
acc = gl.amd.cdna4.mfma(a, b, acc, cd_regclass="a")  # "a" = AGPRs, "v" = VGPRs
```

The argument sets `amdg.cd_regclass` on the `tt.dot`. When lowering it, `MFMA.cpp` wraps each MFMA
tile's accumulator in the same empty `"=a,0"` asm, once before the tile's first K-step and once
after its last. The asm uses the MFMA's own vector type, so a 32×32 accumulator is pinned as one
16-register tuple, with no `i128` split. From `ir_dumps/v9_cd_regclass.llir` (the loop starts at
line 370):

```llvm
%728 = tail call <4 x float> asm "", "=a,0"(<4 x float> %727)  ; pin C
%729 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %688, <8 x half> %648, <4 x float> %728, i32 0, i32 0, i32 0)
%730 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.f16(<8 x half> %693, <8 x half> %653, <4 x float> %729, i32 0, i32 0, i32 0)
%731 = tail call <4 x float> asm "", "=a,0"(<4 x float> %730)  ; pin D
```

## llirSched with pins

The llirSched plugin on this branch handles these pins (kernels without pins get byte-identical
output from the plugin before and after this change):

- **Pins move with their MFMA.** When the scheduler moves an MFMA, the pin on its C goes directly
  before it and the pin on its D directly after it. The scheduler's hoisting of MFMA inputs and
  sinking of MFMA-result extracts also looks through pins.
- **Each pinned tile is fenced.** A `sched.barrier` after every D pin stops LLVM's machine scheduler
  from reordering the pinned MFMAs in the stretches between memory anchors. Without these fences
  that reordering left 200 copies in the loop: LLVM's post-RA copy-removal pass could not find a
  free AGPR for 17 of the MFMA chains. With LLVM's machine scheduler switched off instead, the
  same loop has 0 copies, which is how the fence was found.

Before this change the plugin reverted any block containing the pins, so the loop kept no copies
but lost its schedule.

## Results

**The number to look at is the copy instructions inside the hot loop**: `v_accvgpr_read`,
`v_accvgpr_write` and `v_accvgpr_mov`, which move accumulator values between AGPRs and VGPRs. A clean
kernel keeps the accumulators in AGPRs across the whole loop and has none. The hot loop is the loop
body that contains the MFMAs, from its label to its backward branch.

TFLOPS are only a rough reference: `triton.testing.do_bench` (warm cache, not the tutorial's
cold-cache rocprof methodology) on gfx950, 4096×4096×8192 fp16. Run-to-run spread is about 1%. All
outputs match torch bit for bit.

Default Triton flow:

| variant | copies in loop | AGPR-form MFMAs in loop (x/y) | VGPR spills | TFLOPS (do_bench, reference) |
|---|---|---|---|---|
| v9 unchanged | 36 | 52/256 | 0 | 943 |
| v9 + force-agpr (`amdgpu-mfma-vgpr-form=0` + `amdgpu-agpr-alloc=256`) ¹ | 0 | 256/256 | 0 | 942 |
| approach 1: C and D pinned | 480 | 176/256 | 0 | 849 |
| approach 1: D pinned only | 712 | 118/256 | 147 | 183 |
| approach 1's IR with each asm moved next to its MFMA ¹ | 0 | 256/256 | 0 | 950 |
| **approach 2: `cd_regclass="a"`** | **0** | 256/256 | 0 | 949 |

With the llirSched plugin (the tutorial's `llir` config) ²:

| variant | copies in loop | AGPR-form MFMAs in loop (x/y) | `sched.barrier`s placed | TFLOPS (do_bench, reference) |
|---|---|---|---|---|
| llir | 119 | 58/256 | 96 | 963 |
| llir + force-agpr | 0 | 256/256 | 96 | 1056 |
| llir + approach 2, plugin before this change | 0 | 256/256 | 0 (blocks reverted) | 947 |
| llir + approach 2, pins moved with their MFMA only | 200 | 222/256 | 96 | 983 |
| **llir + approach 2, this plugin (pins moved, pinned tiles fenced)** | **0** | 256/256 | 224 | 1060 |

- **copies in loop**: number of `v_accvgpr_read` / `v_accvgpr_write` / `v_accvgpr_mov` instructions in the hot loop.
- **AGPR-form MFMAs in loop (x/y)**: x of the y MFMA instructions in the hot loop use the AGPR form,
  meaning their accumulator input C and result D are AGPRs, e.g.
  `v_mfma_f32_16x16x32_f16 a[252:255], v[88:91], v[96:99], a[252:255]`. The other y − x use the VGPR
  form, where C and D are VGPRs: `v_mfma_f32_16x16x32_f16 v[..], v[..], v[..], v[..]`. C and D always
  share one register class (the instruction has a single bit for both).
- **VGPR spills**: `.vgpr_spill_count` from the kernel metadata. No variant in the second table spills.
- **`sched.barrier`s placed**: llirSched pins its schedule with `llvm.amdgcn.sched.barrier`; 0 means
  it reverted the blocks and left them unscheduled.

¹ Compiled from Triton's LLVM IR with the pin's `llc -O3 -mcpu=gfx950 -amdgpu-use-amdgpu-trackers`
(plus `-amdgpu-mfma-vgpr-form=0` for force-agpr, on IR carrying `"amdgpu-agpr-alloc"="256"`) and
swapped into the launch. On unmodified IR this command reproduces Triton's assembly byte for byte.

² Triton's LLVM IR run through the plugin with `opt -load-pass-plugin libLlirSched.so -passes=llir-sched`,
then compiled and swapped in as in ¹.

## What is going on

- **Approach 1's pins are grouped per call, not per MFMA.** A Gluon-level pin covers the whole
  accumulator tensor, so each `mfma()` call becomes 16 asm (one per tile) followed by 32 MFMAs (16
  tiles × 2 K-steps). Then come 16 asm pinning the results, then the next call's 16 input pins, and
  so on. See `ir_dumps/v9_beyond_hotloop.llir`: the loop starts at line 370, with C pins at 701–806,
  MFMAs at 888–919, and D pins plus the next call's C pins at 921–1080.
- **The pins never change instruction selection.** At this LLVM, AGPR-form MFMA needs both
  `amdgpu-mfma-vgpr-form=0` and `amdgpu-agpr-alloc`, so with either approach the MFMAs are selected
  in VGPR form and the pins add AGPR↔VGPR copies. LLVM's copy-removal passes take them all out only
  when each pin sits next to its MFMA and the pinned MFMAs are not reordered.
- **Placement is the whole difference.** Moving approach 1's asm next to each MFMA removes every loop
  copy. Approach 2 does that placement in the compiler and matches force-agpr in the default flow (0
  copies, all MFMAs in AGPR form) without either force-agpr setting. With the llirSched changes above
  it also matches llir + force-agpr.
- **Why the single-MFMA example looks fine.** When an `mfma()` call is a single MFMA instruction,
  as in the example on the PR branch, the tensor-level pin sits right next to that instruction by
  construction. The per-call grouping above cannot show up there.
- **Caveats.**
  - The opaque asm makes LLVM insert extra hazard `s_nop`s: 47 in the default-flow loop against 1
    without pins, and 90 in the fenced llirSched loop against 0 for llir + force-agpr.
  - Pinning only D spills (approach 1).
  - `cd_regclass` covers `mfma` only, not `mfma_scaled` yet.
