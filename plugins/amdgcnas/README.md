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
verbose peephole logging). The peephole runs on top of the LLIR scheduler; to
reproduce the full `llir+force-agpr+amdgcnas` stack:

```bash
LLVM_PASS_PLUGIN_PATH=.../plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
TRITON_FORCE_MFMA_AGPR=1 \
TRITON_AMDGCNAS_PLUGIN=1 \
    python bench.py --version 8 --K 8192 --dtype fp16
```

`scripts/run_perf_table.py` wires these into the tutorial's configs; see
[gemm/README §2.1](../../kernels/gemm/README.md#21-triton-build-and-the-out-of-tree-plugins)
for the component stack.
