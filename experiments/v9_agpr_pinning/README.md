# v9 with MFMA accumulators pinned to AGPRs from Gluon source

This experiment takes the [a16w16 v9 kernel](../../kernels/gemm/intra_wave/a16w16/v9_beyond_hotloop/matmul_kernel.py)
and pins every MFMA accumulator to AGPRs with inline-asm register-class constraints, the approach
built on [triton-lang/triton#10337](https://github.com/triton-lang/triton/pull/10337). The question
is whether this can replace `force-agpr`, which is the pair of LLVM settings
`amdgpu-mfma-vgpr-form=0` (process-wide LLVM option) + `amdgpu-agpr-alloc=256` (kernel function
attribute); see [docs/performance_philosophy.md](../../docs/performance_philosophy.md).

Lei Zhang's example on that branch
([`test_cdna4_mfma_register_classes.py`](https://github.com/antiagainst/triton/blob/pr-10337-amd-register-classes/test_cdna4_mfma_register_classes.py))
only has one MFMA. It runs a single wave, and each `mfma()` call is exactly one MFMA instruction
(the tensors are one 16×16×32 or 32×32×16 tile), repeated 1 or 8 times in a loop, with no register
pressure. v9 issues 32 MFMA instructions per `mfma()` call and 256 in the hot loop, close to the
512-register limit.

## Triton compiler

PR #10337 adds `operand_vec_sizes` / `result_vec_sizes` to `inline_asm_elementwise`; the
`gfx950-tutorial-v2.1` tag does not have them. Everything here uses Triton built from:

- [`antiagainst/triton` branch `pr-10337-amd-register-classes`](https://github.com/antiagainst/triton/tree/pr-10337-amd-register-classes)
  at `a26d06a750df8e149b6b32a3e113b4d4b31ae35a`: PR #10337 rebased onto triton `main` at
  `4a15f415d8ac4f830ce788c6fca3a5b3b29908e7`, plus a test/doc commit on top.
- LLVM `b010a18d2b648cab83c83967ff26b8fde11acdc6` (the same pin as `gfx950-tutorial-v2.1`).
- Built with `TRITON_BUILD_PROTON=OFF TRITON_APPEND_CMAKE_ARGS=-DTRITON_BUILD_UT=OFF pip install --no-build-isolation .`

Default Triton flow: no llirSched plugin, no force-agpr, no amdgcnas. Unlike the v2.1 tag, this
upstream-based commit passes `amdgpu-use-amdgpu-trackers` on gfx950.

## Files

- [`matmul_kernel.py`](matmul_kernel.py): v9 with every `gl.amd.cdna3.mfma(a, b, acc)` replaced by
  `mfma_agpr(a, b, acc)`. A diff against v9 shows only the helper block and the 16 call sites.
- [`run.py`](run.py): correctness check and `do_bench` timing, 4096×4096×8192 fp16.
- [`ir_dumps/`](ir_dumps/): the LLVM IR (`.llir`) and gfx950 assembly (`.amdgcn`) Triton generated for `matmul_kernel.py`.

```bash
python run.py
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/dump python run.py  # regenerate ir_dumps/
```

## The pinning

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

## Results

**The number to look at is the copy instructions inside the hot loop**: `v_accvgpr_read`,
`v_accvgpr_write` and `v_accvgpr_mov`, which move accumulator values between AGPRs and VGPRs. A clean
kernel keeps the accumulators in AGPRs across the whole loop and has none. The hot loop is the loop
body that contains the MFMAs, from its label to its backward branch.

TFLOPS are only a rough reference: `triton.testing.do_bench` (warm cache, not the tutorial's
cold-cache rocprof methodology) on gfx950, 4096×4096×8192 fp16, same Triton build. All outputs match
torch bit for bit.

| variant | copies in loop | AGPR-form MFMAs in loop (x/y) | VGPR spills | TFLOPS (do_bench, reference) |
|---|---|---|---|---|
| v9 unchanged | 36 | 52/256 | 0 | 943 |
| v9 + force-agpr (`amdgpu-mfma-vgpr-form=0` + `amdgpu-agpr-alloc=256`) ¹ | 0 | 256/256 | 0 | 942 |
| **this kernel: C and D pinned** | **480** | 176/256 | 0 | 849 |
| D pinned only | 712 | 118/256 | 147 | 183 |
| this kernel's IR with each asm moved next to its MFMA ¹ | 0 | 256/256 | 0 | 950 |

- **copies in loop**: number of `v_accvgpr_read` / `v_accvgpr_write` / `v_accvgpr_mov` instructions in the hot loop.
- **AGPR-form MFMAs in loop (x/y)**: x of the y MFMA instructions in the hot loop use the AGPR form,
  meaning their accumulator input C and result D are AGPRs, e.g.
  `v_mfma_f32_16x16x32_f16 a[252:255], v[88:91], v[96:99], a[252:255]`. The other y − x use the VGPR
  form, where C and D are VGPRs: `v_mfma_f32_16x16x32_f16 v[..], v[..], v[..], v[..]`. C and D always
  share one register class (the instruction has a single bit for both).
- **VGPR spills**: `.vgpr_spill_count` from the kernel metadata.

¹ Compiled from Triton's LLVM IR with the pin's `llc -O3 -mcpu=gfx950 -amdgpu-use-amdgpu-trackers`
(plus `-amdgpu-mfma-vgpr-form=0` for force-agpr, on IR carrying `"amdgpu-agpr-alloc"="256"`) and
swapped into the launch. On unmodified IR this command reproduces Triton's assembly byte for byte.

## What is going on

- **The pins are grouped per call, not per MFMA.** A Gluon-level pin covers the whole accumulator
  tensor, so each `mfma()` call becomes 16 asm (one per tile) followed by 32 MFMAs (16 tiles × 2
  K-steps). Then come 16 asm pinning the results, then the next call's 16 input pins, and so on.
  See `ir_dumps/v9_beyond_hotloop.llir`: the loop starts at line 370, with C pins at 701–806,
  MFMAs at 888–919, and D pins plus the next call's C pins at 921–1080.
- **The pins never change instruction selection.** At this LLVM, AGPR-form MFMA needs both
  `amdgpu-mfma-vgpr-form=0` and `amdgpu-agpr-alloc`; the MFMAs stay in VGPR form and the pins only
  add AGPR↔VGPR copies. LLVM's copy-removal passes clean up part of them, leaving 480 copies in the loop.
- **Placement is the whole difference.** Moving each asm next to its MFMA (same asm, same values)
  removes every loop copy, the same as force-agpr. A Gluon author can't express that placement,
  because `mfma()` works on the whole tensor. It would have to come from the `mfma` op's lowering
  itself, for example a register-class attribute applied per MFMA instruction.
- **Why the single-MFMA example looks fine.** When an `mfma()` call is a single MFMA instruction,
  as in the example on the PR branch, the tensor-level pin sits right next to that instruction by
  construction. The per-call grouping above cannot show up there.
- **Other caveats.**
  - Pinning only D spills.
  - The llirSched plugin reverts any block containing these asm calls (0 `sched.barrier`s instead of 96).
  - The opaque asm makes LLVM insert extra hazard `s_nop`s.
