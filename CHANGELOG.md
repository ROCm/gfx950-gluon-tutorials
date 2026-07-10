# Changelog

This file records changes to the tutorial kernels that are driven by upstream
compiler / Triton evolution.

---

## 2026-07-09 — Re-pin to `gfx950-tutorial-v1.0` (out-of-tree scheduler + amdgcnas plugins)

The pinned Triton commit moves from `gfx950-tutorial-v0.3` to
`gfx950-tutorial-v1.0`. Unlike the earlier maintenance re-pins, this one changes
*where* the tutorial's compiler passes live: `v1.0` **removes** the in-tree LLIR
scheduler and the `amdgcnas` post-assembly tool from Triton, and the tutorial now
ships them as out-of-tree plugins in `plugins/`:

The three components each have their own enable knob and their own cumulative
`run_perf_table.py` config (`base` / `llir` / `llir+force-agpr` /
`llir+force-agpr+amdgcnas`):

- **llirSched** → `plugins/llir_scheduler/libLlirSched.so`, an LLVM pass plugin
  loaded via `LLVM_PASS_PLUGIN_PATH` (+ `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1`).
  Replaces the old `TRITON_ENABLE_LLIR_SCHED=1` env var.
- **force-agpr** → the RA hints, now a single env var `TRITON_FORCE_MFMA_AGPR=1`:
  the kernels set `llvm_fn_attrs="amdgpu-agpr-alloc=256"` (reserve AGPRs) and
  `llvm.cc` sets `amdgpu-mfma-vgpr-form=0` (AGPR-form MFMA). Replaces the earlier
  split `TRITON_LLVM_FN_ATTRS` / `TRITON_ENABLE_AMDGPU_RA_HINTS`.
- **amdgcnas** → now *only* the post-assembly peephole, a pure-Python
  `amdgcn`-stage hook (`TRITON_AMDGCNAS_PLUGIN=1`), reduced to the LICM /
  save-restore / loop-scheduling passes. Replaces `TRITON_ENABLE_AMDGCN_AS=1`.

Triton must be built with `TRITON_EXT_ENABLED=1` for the scheduler plugin to
resolve LLVM symbols. `v1.0` shares `v0.3`'s upstream base (`63a5e1f0e`) and LLVM
pin (`62b7cf96`), so the prebuilt `.so` and all numbers carry over: the plugin
path reproduces the in-tree scheduler's assembly, and the TFLOPS / MFMA-efficiency
numbers and IR/assembly dumps are unchanged within run-to-run noise. Both
`run_perf_table.py` and `run_counter_collection.py` now drive the plugins;
`v0.1`–`v0.3` remain immutable and still reproduce their original numbers.

## 2026-07-08 — Re-pin to `gfx950-tutorial-v0.3` (rebase onto upstream main)

The pinned Triton commit moved from `gfx950-tutorial-v0.2` to
`gfx950-tutorial-v0.3`. `v0.3` carries the **identical** tutorial commit — the
LLIR scheduler, `amdgcnas`, and the RA-hints split — and rebases it onto a newer
`triton-lang/triton` `main` (base `63a5e1f0e`, "[AMD][gfx1250] Enable local
prefetch schedule in pipeliner"). The `gfx950-tutorial` development branch was
force-moved to the rebased commit; `v0.1` and `v0.2` remain immutable and still
reproduce their original numbers.

This is a maintenance re-pin to keep the toolchain building against current
upstream — there is **no kernel change and no perf change**. All TFLOPS and MFMA
efficiency numbers, the v0–v9 progression chart, and the IR/assembly dumps are
unchanged from `v0.2` (within run-to-run noise); only the pinned tag referenced
in the docs (`SUPPORT.md`, `docs/regenerating_ir_dumps.md`, the `kernels/gemm`
READMEs, and the collection scripts) advances to `v0.3`.

## 2026-06-15 — MXFP4 (a4w4): add `v1_sliceMN`, version the directory

The `a4w4/` directory is now versioned like `a16w16/`: the original kernel
becomes `v0_sliceN`, and a new `v1_sliceMN` is added (both kept as a
progression; `bench.py --version` and `run_perf_table.py --versions` select
between them).

`v1_sliceMN` makes two changes over `v0_sliceN`:
- **Async scale pipeline.** Scales stream straight into LDS via
  `buffer_load_to_lds` alongside the input tiles, replacing v0's
  `buffer_load → local_store → local_load` round-trip — no `ds_write`, so the
  LLIR scheduler no longer has to reason about it in the hot loop.
- **M+N slicing.** A is sliced along M and B along N into a 2×2 grid of
  128×128 accumulator quadrants (the `a16w16/v8_sliceMN` pattern), for a more
  balanced buffer-load distribution at large K.

Measured on MI355 at 4096×4096×32768 with `llirSched + amdgcnas`, `v1_sliceMN`
reaches **5387 TFLOPS / 94.5% MFMA efficiency**, up from `v0_sliceN`'s
5265 / 91.6%. The MXFP4 headline in `kernels/gemm/README.md` now reports the
v1 number. No Triton pin change — both kernels use the existing
`gfx950-tutorial-v0.2` toolchain.

## 2026-06-01 — Re-pin to `gfx950-tutorial-v0.2` (rebase onto upstream main)

The pinned Triton commit moved from `gfx950-tutorial-v0.1` to
`gfx950-tutorial-v0.2`. `v0.2` rebases the `gfx950-tutorial` branch (the LLIR
scheduler, `amdgcnas`, and RA hints) onto a current `triton-lang/triton` `main`,
and all perf numbers, the v0–v9 chart, and the IR/assembly dumps in this repo
were refreshed against it. `v0.1` remains immutable and still reproduces the old
numbers.

**MXFP4 MFMA efficiency: 92.41% → ~90%.** Between the two pins, upstream
[triton #10383](https://github.com/triton-lang/triton/pull/10383) migrated
workgroup memory barriers from a bare `s.barrier` to the MMRA-annotated triple
`fence release ; s.barrier ; fence acquire` (the backend-recommended form). The
LLIR scheduler is fence-unaware: it relocates the `s.barrier` and the MFMAs but
not the fences, so the release fence drifts up next to the producing LDS store
and the backend drains LDS (`s_waitcnt lgkmcnt(0)`) *before* the MFMA run
instead of at the barrier, exposing LDS latency the old schedule hid. End-to-end
TFLOPS is essentially unchanged (5255 → 5253); MFMA efficiency drops ~7pp.

The LLIR scheduler now re-glues each barrier's `release`/`acquire` fences back
to the `s.barrier` (`release ; barrier ; acquire`), recovering most of the loss
(85% → ~90%). A residual ~1–2pp vs `v0.1` remains (a single-MFMA spacing
decision at two barriers plus register-allocation drift); it is not a
fence/waitcnt-placement issue. FP16 and BF8 are unchanged within noise (BF8
reaches 99.98% MFMA efficiency at `v0.2`).

Separately, `v0.2` decouples the LLVM machine-scheduler disable from the
`TRITON_ENABLE_LLIR_SCHED` env var: misched is now disabled only when the LLIR
scheduler actually scheduled a kernel (and the option is restored per-compile),
so misched stays enabled when the pass is off or bails out.

## 2026-04-09 — Removed `load_shared_relaxed` in favor of `smem.load()`

All call sites of the form
```python
gl.amd.cdna4.async_copy.load_shared_relaxed(smem, layout)
```
have been replaced with the standard
```python
smem.load(layout)
```

`load_shared_relaxed` existed to inject a no-alias annotation between an LDS
load and any in-flight `buffer_load_to_lds` async copies, so the LLVM backend
wouldn't insert overly conservative `vmcnt` waits. As of OAI-triton commit
[`d78665bc2b`](https://github.com/ROCm/triton/commit/d78665bc2b) on the
`gfx950-tutorial` branch (the asyncMarker / `wait_asyncmark` rework), the
no-alias relationship is now derived automatically from the dependency chain
`async_copy` → `commit_group` → `wait_group` → `local_load`. The relaxed
variant is no longer needed — the standard `smem.load()` lowers to identical
assembly.

Affected: all kernels that use `async_copy` (`a16w16/v2`–`v8`, `a8w8`, `a4w4`).
Requires OAI-triton commit `d78665bc2b` or later.
