# Regenerating the IR / Assembly Dumps

Two kernel versions in this tutorial bundle compiler dump artifacts so the README narrative can link to specific lines:

- `kernels/gemm/a16w16/v3_lds/ir_dump_K4096_fp16/` (three subdirectories: `no_swizzling/`, `swizzling_8-2-8/`, `padding_512-16/`)
- `kernels/gemm/a16w16/v5_local_prefetch/ir_dump_K4096_fp16/`
- `kernels/gemm/a16w16/v5_local_prefetch/ir_dump_K4096_fp16_llirSched/`

Each directory contains four files: `.ttgir` (Triton GPU IR), `.llir` (LLVM IR), `.amdgcn` (final assembly as Triton emits it, with debug labels), and `.s` (the same assembly with `.loc` directives and `.Ltmp` labels stripped, which is what the READMEs link to so the cited line numbers stay stable). All artifacts in this repository were produced against the [`gfx950-tutorial-v1.0`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v1.0) tag in `triton-lang/triton`. To verify them, or regenerate after a Triton bump, follow the steps below.

## Prerequisites

Triton is built from the `gfx950-tutorial-v1.0` tag (or any commit reachable from it).
As of `v1.0` the LLIR scheduler and the `amdgcnas` peephole ship as out-of-tree plugins
in this repo (`plugins/`), so build Triton with plugin symbols exposed:

```bash
git clone https://github.com/triton-lang/triton -b gfx950-tutorial-v1.0 /tmp/triton
cd /tmp/triton && TRITON_EXT_ENABLED=1 pip install -e .
```

The scheduler `.so` is prebuilt at `plugins/llir_scheduler/libLlirSched.so`; see
`plugins/llir_scheduler/README.md` if you need to rebuild it against the pinned LLVM.

## How Triton emits dump artifacts

When a Triton kernel is compiled, the resulting IR / assembly artifacts land in the Triton cache directory (`$TRITON_CACHE_DIR`, default `~/.triton/cache/<hash>/`). Each compilation produces `<kernel_name>.ttgir` (Triton GPU IR), `<kernel_name>.llir` (LLVM IR), and `<kernel_name>.amdgcn` (final assembly with debug labels) alongside the compiled binary.

The `.s` file checked into this repository is **not** produced directly by Triton — it is a cleaned-up copy of `.amdgcn` with `.loc` directives and `.Ltmp*:` temp labels stripped, so the line numbers cited from the READMEs stay stable.

In addition, **all four artifacts** (`.ttgir`, `.llir`, `.amdgcn`, `.s`) are *path-normalized*: the absolute build paths Triton embeds (in `loc(...)` strings and `.file` directives) are rewritten to repo-relative form, so the dumps reproduce identically across machines and don't leak local checkout paths. The cleanup is a small shell helper:

```bash
# $1 = cache dir holding <kernel>.{ttgir,llir,amdgcn}, $2 = dest dir, $3 = kernel name
emit() {
  local SRC="$1" DEST="$2" K="$3"; mkdir -p "$DEST"
  # .ttgir / .llir / .amdgcn — path-normalize only
  for ext in ttgir llir amdgcn; do
    sed -e 's#/[^" ]*/kernels/#kernels/#g' -e 's#/[^" ]*/python/triton/#python/triton/#g' \
        "$SRC/$K.$ext" > "$DEST/$K.$ext"
  done
  # .s — strip .loc / .Ltmp labels, then path-normalize
  sed -e '/^[[:space:]]*\.loc[[:space:]]/d' -e '/^\.Ltmp[0-9]*:/d' \
      -e 's#/[^" ]*/kernels/#kernels/#g' -e 's#/[^" ]*/python/triton/#python/triton/#g' \
      "$SRC/$K.amdgcn" > "$DEST/$K.s"
}
```

The standard regeneration workflow is therefore:

1. **Point Triton at a fresh cache directory** so the run produces artifacts deterministically (and so multiple variants don't collide):
   ```bash
   export TRITON_CACHE_DIR=/tmp/triton_cache_v3_swizzling
   rm -rf "$TRITON_CACHE_DIR"
   ```
2. **Run the kernel once** so Triton compiles and caches it.
3. **Copy the three Triton-emitted files** (`.ttgir`, `.llir`, `.amdgcn`) from `$TRITON_CACHE_DIR/<hash>/` into the corresponding `ir_dump_*` directory in this repository.
4. **Generate `.s`** from the copied `.amdgcn` using the `sed` command above.

The exact run command depends on which variant you are regenerating.

## v3_lds — three LDS layout variants

The three subdirectories under `v3_lds/ir_dump_K4096_fp16/` correspond to the three LDS layout configurations defined in `v3_lds/matmul_kernel.py`:

| Subdirectory | Source kernel | Selected by |
|---|---|---|
| `no_swizzling/` | `v3_lds_swizzling` with `SwizzledSharedLayout(1, 1, 1, ...)` | edit `matmul()` to call `v3_lds_swizzling[grid](...)` and set `vec=1, perPhase=1, maxPhase=1` |
| `swizzling_8-2-8/` | `v3_lds_swizzling` with `SwizzledSharedLayout(8, 2, 8, ...)` | edit `matmul()` to call `v3_lds_swizzling[grid](...)` (default in repo) |
| `padding_512-16/` | `v3_lds_padding` | edit `matmul()` to call `v3_lds_padding[grid](...)` (default in repo) |

For each variant:

```bash
# Choose the variant by editing v3_lds/matmul_kernel.py's matmul() launcher
# (see the comment near line 244)

cd kernels/gemm/a16w16
export TRITON_CACHE_DIR=/tmp/triton_cache_v3_<variant>
rm -rf "$TRITON_CACHE_DIR"
python bench.py --version 3 --K 4096 --dtype fp16

# Locate the matching cache subdirectory, then copy + clean + path-normalize
# with the emit() helper defined above. The kernel name is v3_lds_swizzling
# for the no_swizzling / swizzling_8-2-8 variants and v3_lds_padding for padding.
SRC=$(dirname "$(ls "$TRITON_CACHE_DIR"/*/v3_lds_*.amdgcn | head -1)")
emit "$SRC" v3_lds/ir_dump_K4096_fp16/<variant> <kernel_name>
```

## v5_local_prefetch — base, llirSched, and llirSched+amdgcnas variants

```bash
cd kernels/gemm/a16w16

# (uses the emit() helper defined above)

# As of gfx950-tutorial-v1.0 the scheduler and amdgcnas peephole are out-of-tree
# plugins: bench.py loads the scheduler .so when LLVM_PASS_PLUGIN_PATH is set and
# installs the amdgcnas hook when TRITON_AMDGCNAS_PLUGIN=1.
LLIR_PLUGIN="$(git rev-parse --show-toplevel)/plugins/llir_scheduler/libLlirSched.so"

# --- base variant ---
export TRITON_CACHE_DIR=/tmp/triton_cache_v5_base
rm -rf "$TRITON_CACHE_DIR"
python bench.py --version 5 --K 4096 --dtype fp16
SRC=$(dirname "$(ls "$TRITON_CACHE_DIR"/*/v5_local_prefetch.amdgcn | head -1)")
emit "$SRC" v5_local_prefetch/ir_dump_K4096_fp16 v5_local_prefetch

# --- llirSched variant ---
export TRITON_CACHE_DIR=/tmp/triton_cache_v5_llir
rm -rf "$TRITON_CACHE_DIR"
LLVM_PASS_PLUGIN_PATH="$LLIR_PLUGIN" LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
    python bench.py --version 5 --K 4096 --dtype fp16
SRC=$(dirname "$(ls "$TRITON_CACHE_DIR"/*/v5_local_prefetch.amdgcn | head -1)")
emit "$SRC" v5_local_prefetch/ir_dump_K4096_fp16_llirSched v5_local_prefetch

# --- llirSched + amdgcnas variant ---
export TRITON_CACHE_DIR=/tmp/triton_cache_v5_amdgcnas
rm -rf "$TRITON_CACHE_DIR"
LLVM_PASS_PLUGIN_PATH="$LLIR_PLUGIN" LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
    TRITON_LLVM_FN_ATTRS=amdgpu-agpr-alloc=256 TRITON_ENABLE_AMDGPU_RA_HINTS=1 \
    TRITON_AMDGCNAS_PLUGIN=1 python bench.py --version 5 --K 4096 --dtype fp16
SRC=$(dirname "$(ls "$TRITON_CACHE_DIR"/*/v5_local_prefetch.amdgcn | head -1)")
emit "$SRC" v5_local_prefetch/ir_dump_K4096_fp16_llirSched_amdgcnas v5_local_prefetch
```

> [!NOTE]
> The `llirSched + amdgcnas` artifacts are not bundled in the repository (no README cites them), but the recipe above lets you reproduce them locally if you want to inspect what `amdgcnas` actually does to the assembly.

## Verification

The files are checked into the repository so that the line-number anchors in the v3 and v5 READMEs (`README.md#L427`, `#L814-L907`, etc.) remain stable for readers. After regeneration, diff your local copies against the checked-in files — the artifacts should match (or the line numbers in the cited README sections need updating).

If a Triton bump shifts artifacts in a way that invalidates the cited line numbers, the affected README sections need to be updated alongside the regenerated dumps.
