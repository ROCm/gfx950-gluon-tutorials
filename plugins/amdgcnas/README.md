# amdgcnas — out-of-tree post-assembly peephole

`amdgcnas` originally did two separable things inside the Triton fork. This
directory moves the algorithmic half out of tree; the pieces are now:

| Piece | What it does | Where it lives now |
|-------|--------------|--------------------|
| `amdgpu-agpr-alloc=256` | reserve 256 AGPRs for MFMA accumulators | **kernel option** `llvm_fn_attrs="amdgpu-agpr-alloc=256"` (kernels read it from `TRITON_LLVM_FN_ATTRS`) |
| `amdgpu-mfma-vgpr-form=0` | keep accumulators in AGPR form | **in-tree** LLVM flag, gated by `TRITON_ENABLE_AMDGPU_RA_HINTS` / `TRITON_ENABLE_AMDGCN_AS` in `llvm.cc` |
| post-assembly **peephole** (LICM + MFMA/scalar interleave) | rewrite the final assembly text | **out-of-tree here**, via the stages hook |

## Files
- `amdgcnas_ext.py` — verbatim copy of Triton's `python/triton/tools/amdgcnas.py`.
  Its entry point `amdgcn_as(text, verbose=False) -> text` is a pure-Python
  transform on the assembly string (stdlib only — no LLVM, no libtriton).
- `amdgcnas_plugin.py` — a `knobs.runtime.add_stages_inspection_hook` that wraps
  the `amdgcn` compile stage: run in-tree codegen, then `amdgcn_as` on the result.

## Why this one is trivial (vs the LLIR scheduler)
The peephole is post-codegen text, so it attaches at the `amdgcn` stage with the
pure-Python stages hook. **No `.so`, no `TRITON_EXT_ENABLED`, no `RTLD_GLOBAL`,
no LLVM ABI lock, no Triton rebuild** — it works on stock Triton.

## Use
`bench.py` installs the hook when `TRITON_AMDGCNAS_PLUGIN=1`. To reproduce the
full `llir+amdgcnas` stack out of tree:

```bash
LLVM_PASS_PLUGIN_PATH=.../plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
TRITON_LLVM_FN_ATTRS="amdgpu-agpr-alloc=256" \
TRITON_ENABLE_AMDGPU_RA_HINTS=1 \
TRITON_AMDGCNAS_PLUGIN=1 \
    python bench.py --version 8 --K 8192 --dtype fp16
```
`scripts/run_perf_table.py` sets these for the `llir+ra` and `llir+amdgcnas` configs.
