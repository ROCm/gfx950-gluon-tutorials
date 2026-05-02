# Regenerating the IR / Assembly Dumps

Two kernel versions in this tutorial bundle compiler dump artifacts so the README narrative can link to specific lines:

- `kernels/gemm/a16w16/v3_lds/ir_dump_K4096_fp16/` (three subdirectories: `no_swizzling/`, `swizzling_8-2-8/`, `padding_512-16/`)
- `kernels/gemm/a16w16/v5_local_prefetch/ir_dump_K4096_fp16/`
- `kernels/gemm/a16w16/v5_local_prefetch/ir_dump_K4096_fp16_llirSched/`

Each directory contains four files: `.ttgir` (Triton GPU IR), `.llir` (LLVM IR), `.amdgcn` (final assembly as Triton emits it, with debug labels), and `.s` (the same assembly with `.loc` directives and `.Ltmp` labels stripped, which is what the READMEs link to so the cited line numbers stay stable). All artifacts in this repository were produced against the [`gfx950-tutorial-v0.1`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v0.1) tag in `triton-lang/triton`. To verify them, or regenerate after a Triton bump, follow the steps below.

## Prerequisites

Triton is built from the `gfx950-tutorial-v0.1` tag (or any commit reachable from it):

```bash
git clone https://github.com/triton-lang/triton -b gfx950-tutorial-v0.1 /tmp/triton
cd /tmp/triton && pip install -e .
```

## How Triton emits dump artifacts

When a Triton kernel is compiled, the resulting IR / assembly artifacts land in the Triton cache directory (`$TRITON_CACHE_DIR`, default `~/.triton/cache/<hash>/`). Each compilation produces `<kernel_name>.ttgir` (Triton GPU IR), `<kernel_name>.llir` (LLVM IR), and `<kernel_name>.amdgcn` (final assembly with debug labels) alongside the compiled binary.

The `.s` file checked into this repository is **not** produced directly by Triton — it is a cleaned-up copy of `.amdgcn` with `.loc` directives and `.Ltmp*:` temp labels stripped, so the line numbers cited from the READMEs stay stable. The cleanup is a two-line `sed`:

```bash
sed -e '/^[[:space:]]*\.loc[[:space:]]/d' -e '/^\.Ltmp[0-9]*:/d' input.amdgcn > output.s
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

# Locate the matching cache subdirectory and copy + clean artifacts:
DEST=v3_lds/ir_dump_K4096_fp16/<variant>
cp "$TRITON_CACHE_DIR"/*/v3_lds_*.{ttgir,llir,amdgcn} "$DEST/"
for f in "$DEST"/*.amdgcn; do
    sed -e '/^[[:space:]]*\.loc[[:space:]]/d' -e '/^\.Ltmp[0-9]*:/d' \
        "$f" > "${f%.amdgcn}.s"
done
```

## v5_local_prefetch — base, llirSched, and llirSched+amdgcnas variants

```bash
cd kernels/gemm/a16w16

# --- base variant ---
export TRITON_CACHE_DIR=/tmp/triton_cache_v5_base
rm -rf "$TRITON_CACHE_DIR"
python bench.py --version 5 --K 4096 --dtype fp16
DEST=v5_local_prefetch/ir_dump_K4096_fp16
cp "$TRITON_CACHE_DIR"/*/v5_local_prefetch.{ttgir,llir,amdgcn} "$DEST/"
sed -e '/^[[:space:]]*\.loc[[:space:]]/d' -e '/^\.Ltmp[0-9]*:/d' \
    "$DEST/v5_local_prefetch.amdgcn" > "$DEST/v5_local_prefetch.s"

# --- llirSched variant ---
export TRITON_CACHE_DIR=/tmp/triton_cache_v5_llir
rm -rf "$TRITON_CACHE_DIR"
TRITON_ENABLE_LLIR_SCHED=1 python bench.py --version 5 --K 4096 --dtype fp16
DEST=v5_local_prefetch/ir_dump_K4096_fp16_llirSched
cp "$TRITON_CACHE_DIR"/*/v5_local_prefetch.{ttgir,llir,amdgcn} "$DEST/"
sed -e '/^[[:space:]]*\.loc[[:space:]]/d' -e '/^\.Ltmp[0-9]*:/d' \
    "$DEST/v5_local_prefetch.amdgcn" > "$DEST/v5_local_prefetch.s"

# --- llirSched + amdgcnas variant ---
export TRITON_CACHE_DIR=/tmp/triton_cache_v5_amdgcnas
rm -rf "$TRITON_CACHE_DIR"
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
    python bench.py --version 5 --K 4096 --dtype fp16
DEST=v5_local_prefetch/ir_dump_K4096_fp16_llirSched_amdgcnas
mkdir -p "$DEST"
cp "$TRITON_CACHE_DIR"/*/v5_local_prefetch.{ttgir,llir,amdgcn} "$DEST/"
sed -e '/^[[:space:]]*\.loc[[:space:]]/d' -e '/^\.Ltmp[0-9]*:/d' \
    "$DEST/v5_local_prefetch.amdgcn" > "$DEST/v5_local_prefetch.s"
```

> [!NOTE]
> The `llirSched + amdgcnas` artifacts are not bundled in the repository (no README cites them), but the recipe above lets you reproduce them locally if you want to inspect what `amdgcnas` actually does to the assembly.

## Verification

The files are checked into the repository so that the line-number anchors in the v3 and v5 READMEs (`README.md#L427`, `#L814-L907`, etc.) remain stable for readers. After regeneration, diff your local copies against the checked-in files — the artifacts should match (or the line numbers in the cited README sections need updating).

If a Triton bump shifts artifacts in a way that invalidates the cited line numbers, the affected README sections need to be updated alongside the regenerated dumps.
