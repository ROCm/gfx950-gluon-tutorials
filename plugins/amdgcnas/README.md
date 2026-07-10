# amdgcnas — out-of-tree post-assembly peephole

`amdgcnas` originally bundled register-allocation hints and a post-assembly
peephole inside the Triton fork. Those are now **two separate components**: the RA
hints became the **force-agpr** component (a single env var, `TRITON_FORCE_MFMA_AGPR`),
and `amdgcnas` is *only* the peephole, shipped out-of-tree here.

| Piece | Component | What it does | How to enable |
|-------|-----------|--------------|---------------|
| `amdgpu-agpr-alloc=256` | force-agpr | reserve 256 AGPRs for MFMA accumulators | `TRITON_FORCE_MFMA_AGPR=1` → kernels set `llvm_fn_attrs="amdgpu-agpr-alloc=256"` |
| `amdgpu-mfma-vgpr-form=0` | force-agpr | keep accumulators in AGPR form | `TRITON_FORCE_MFMA_AGPR=1` → `llvm.cc` sets the flag |
| post-assembly **peephole** (LICM + MFMA/scalar interleave) | amdgcnas | rewrite the final assembly text | `TRITON_AMDGCNAS_PLUGIN=1` (out-of-tree hook here) |

## Files
- `amdgcnas_ext.py` — reduced from Triton's `python/triton/tools/amdgcnas.py`
  (the LICM / save-restore / loop-scheduling subset that helps these kernels, with
  the VGPR-count directives recomputed from the kernel's real register usage). Its
  entry point `amdgcn_as(text, verbose=False) -> text` is a pure-Python transform on
  the assembly string (stdlib only — no LLVM, no libtriton).
- `amdgcnas_plugin.py` — a `knobs.runtime.add_stages_inspection_hook` that wraps
  the `amdgcn` compile stage: run in-tree codegen, then `amdgcn_as` on the result.

## Why this one is trivial (vs the LLIR scheduler)
The peephole is post-codegen text, so it attaches at the `amdgcn` stage with the
pure-Python stages hook. **No `.so`, no `TRITON_EXT_ENABLED`, no `RTLD_GLOBAL`,
no LLVM ABI lock, no Triton rebuild** — it works on stock Triton.

## Use
`bench.py` installs the hook when `TRITON_AMDGCNAS_PLUGIN=1` (set it to `2` for
verbose peephole logging). To reproduce the full `llir+force-agpr+amdgcnas` stack out of tree:

```bash
LLVM_PASS_PLUGIN_PATH=.../plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
TRITON_FORCE_MFMA_AGPR=1 \
TRITON_AMDGCNAS_PLUGIN=1 \
    python bench.py --version 8 --K 8192 --dtype fp16
```
`scripts/run_perf_table.py` sets these for the `llir+force-agpr` and `llir+force-agpr+amdgcnas` configs.
