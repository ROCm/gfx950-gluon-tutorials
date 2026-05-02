# Changelog

This file records changes to the tutorial kernels that are driven by upstream
compiler / Triton evolution.

---

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
