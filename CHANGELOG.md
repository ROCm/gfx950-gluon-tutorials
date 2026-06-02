# Changelog

This file records changes to the tutorial kernels that are driven by upstream
compiler / Triton evolution.

---

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
