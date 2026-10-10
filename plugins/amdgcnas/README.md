# amdgcnas — out-of-tree post-assembly peephole

`amdgcnas` is a pure-Python peephole over the final `amdgcn` assembly text: it
applies LICM and interleaves MFMA with scalar instructions to compress the
remaining non-MFMA gaps in the GEMM hot loop. It attaches at the `amdgcn` compile
stage through a stages-inspection hook, so it needs no `.so`, no LLVM symbols, and
no Triton rebuild — it works on stock Triton.

## Files
- `amdgcnas_ext.py` — the peephole itself: the LICM / save-restore /
  loop-scheduling passes, with the VGPR-count directives computed from the kernel's
  real register usage. Its entry point `amdgcn_as(text, verbose=False) -> text` is a
  pure-Python transform on the assembly string (stdlib only — no LLVM, no libtriton).
- `amdgcnas_plugin.py` — a `knobs.runtime.add_stages_inspection_hook` that wraps
  the `amdgcn` compile stage: run in-tree codegen, then `amdgcn_as` on the result.

## Use
`bench.py` installs the hook when `TRITON_AMDGCNAS_PLUGIN=1` (set it to `2` for
verbose peephole logging). The peephole runs on top of Triton's MFMA scheduler, which the
kernels enable themselves; to reproduce the full `llir+amdgcnas` stack:

```bash
TRITON_AMDGCNAS_PLUGIN=1 python bench.py --version 8 --K 8192 --dtype fp16
```

The kernels it is used on (a16w16 v7 and later, a8w8, a4w4) keep their MFMA accumulators in
AGPRs with `cd_regclass="a"`, and the peephole assumes that, so it is only used on these pinned
kernels.

## What LICM may hoist
LICM moves a loop-invariant instruction to the end of the prologue. When the register it writes is
written again inside the loop, the hoisted copy is renamed to a free register and its users are
renamed with it. Three rules keep that safe whatever registers the compiler assigned:

- A free register is free *inside the loop*. If the value is also read after the loop, the
  epilogue can overwrite such a register before those reads (the accumulator read-backs do), so the
  hoisted copy takes a register that no other block touches.
- An instruction that reads a loop-invariant value whose definition stays in the loop stays in the
  loop too.
- A value that is carried around the back edge to a use at the top of the loop is not hoisted.

`scripts/run_perf_table.py` wires these into the tutorial's configs; see
[gemm/README §2.1](../../kernels/gemm/intra_wave/README.md#21-triton-build-the-mfma-scheduler-and-the-amdgcnas-plugin)
for the component stack.
