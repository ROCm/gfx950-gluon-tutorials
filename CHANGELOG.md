# Changelog

This file records changes to the tutorial kernels that are driven by upstream
compiler / Triton evolution.

---

## 2026-07-29 — New Triton tag `gfx950-tutorial-v2.0` (Gluon `warp_predicate`, warp-pipeline barriers)

[`gfx950-tutorial-v2.0`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v2.0)
is `gfx950-tutorial-v1.1` plus two commits. **It is the pin for the attention kernels only** —
the GEMM kernels stay on `v1.1`, and their recorded numbers stay as measured there until they
are re-measured on `v2.0`.

- **`gl.warp_predicate`** — a per-wave masked-skip region lowering to `s_and_saveexec` +
  `s_cbranch_execz`, with no cross-wave reduction and no barrier. `kernels/attention/fmha_v4.py`
  does not compile without it: it is what lets a wavefront skip the accumulator rescale when
  none of its own rows advanced their running max.
- **warp-pipeline barriers** — always emit LOCAL cluster barriers rather than the minimal set,
  place the loop-carried wrap-around barrier at the top of the loop body, and force a hard LDS
  drain there with `amdgpu.memory_counter_wait(ds = 0)`. Together these keep `s_waitcnt lgkmcnt`
  out of the MFMA stages.

- **LLVM pin unchanged (`850a2b1b`).** Both tags carry the same LLVM, so the out-of-tree
  LLIR-scheduler plugin (`plugins/llir_scheduler/libLlirSched.so`) is **not** rebuilt, and the
  one `.so` works against either tag. This is the first re-pin where that holds.
- **The barrier change fires for every warp-pipeline kernel**, not just attention, so it was
  checked against `v1.1` before tagging: all four `inter_wave` GEMM kernels (a16w16 fp16/bf16,
  a8w8, a4w4) pass their correctness checks on both tags.

## 2026-07-29 — Flash Attention forward kernels (`kernels/attention/`)

Adds the first non-GEMM section of the tutorial: `fmha_v3` (eager softmax rescale) and `fmha_v4`
(lazy rescale), two Gluon flash-attention forward kernels sharing one 8-wave warp-pipeline
architecture. They extend the `inter_wave` design to a kernel with a **third** category of
instruction competing for the SIMD — the softmax's vector math — which the README works
through as an MFMA ↔ VALU **co-execution** budget.

- **llirSched gains a second scheduling model.** Regions mixing `mfma` with memory keep the
  existing throughput interleave; regions mixing `mfma` with VALU are instead *declared* with
  `sched_group_barrier` and built by AMDGPU's IGroupLP, with a separate algorithm for regions
  whose VALU demand exceeds the available MFMA shadow. Region classification picks between
  them, so the GEMM kernels are unaffected — their assembly is unchanged.
- **`amdgcnas` is not used by these kernels.**
- **Three FA-specific scripts**: `scripts/fa_kernel_time.py`, `scripts/fly_kernel_time.py`
  (times ROCm/FlyDSL under our protocol for a like-for-like comparison) and
  `scripts/att_pick_cu.py`.

> [!IMPORTANT]
> These kernels need `gfx950-tutorial-v2.0` — `fmha_v4` does not compile without
> `gl.warp_predicate`. See the re-pin entry above.

## 2026-07-14 — Re-pin to `gfx950-tutorial-v1.1` (upstream merge of the warp-pipeline barrier PR)

The pinned Triton commit moves from `gfx950-tutorial-v1.0` to `gfx950-tutorial-v1.1`.
`v1.1` rebases the tutorial's out-of-tree-plugin support onto current upstream
`triton-lang/triton` `main`, which now carries the merged warp-pipeline barrier work
([#10840](https://github.com/triton-lang/triton/pull/10840)): `ConvertWarpPipeline`
**always** emits a `sched.barrier` (`SchedGroupMask::non_mem_non_sideeffect`) after every
top-level memory op in each warp-pipeline stage — no opt-in flag. The tutorial's local
`fence_loads` / `keep_order` commits are dropped (superseded by the merge).

- **LLVM pin: `62b7cf96` → `850a2b1b`.** The out-of-tree LLIR-scheduler plugin
  (`plugins/llir_scheduler/libLlirSched.so`) is ABI-locked to the LLVM and has been
  **rebuilt** against `850a2b1b`.
- **`inter_wave` a4w4 v1/v2 drop the `fence_loads=True` kwarg** — the barrier is now
  unconditional, so the kwarg is a no-op and is removed.

Perf is preserved by the re-pin (MI355X, rocprof):

- **intra_wave** (`llir+force-agpr+amdgcnas`): a16w16 v9 1418 / 98.6%, a8w8 3231 / 99.5%,
  a4w4 v0 4901 / 81.2%, v1 5180 / 93.9% — matches `v1.0` within noise.
- **inter_wave**: a16w16 1440 / 99.8%, a8w8 2868 / 99.7%, a4w4 v1 4931 / 79.5%,
  v2 4798 / 98.9% — matches. **a4w4 v0 improves 57% → 66% MFMA** (4276 TFLOPS): v0 never
  used the flag, so the now-always-on barrier applies to it too.

## 2026-07-13 — 8-wave warp-pipeline (`inter_wave`) GEMM kernels + `gemm/` reorganization

Adds an **8-wave warp-pipeline** variant of each GEMM — a second scheduling model
alongside the 4-wave `llir + force-agpr + amdgcnas` kernels — and reorganizes
`kernels/gemm/` around the two models.

**The 8-wave kernels** (`inter_wave/a16w16` FP16/BF16, `inter_wave/a8w8` BF8,
`inter_wave/a4w4` MXFP4) launch **8 warps/CTA (2 waves/SIMD)** and schedule the hot
loop at the **wave level** with `warp_pipeline_stage`: the two resident waves per SIMD
are kept out of phase (a ping-pong) so one issues MFMAs while the other issues loads,
then they swap. They run with **no AGPRs** (`amdgpu-agpr-alloc=0,0` via `llvm_fn_attrs`)
and need **no plugins and no env vars** — they build against stock `gfx950-tutorial-v1.0`
(the `a4w4` v1/v2 kernels additionally need a Triton built with warp-pipeline
`fence_loads`, upstream PR #10840). The theory is in
[`docs/warp_pipelining.md`](docs/warp_pipelining.md); the warp-pipeline schedule
and `docs/warp_pipelining.md` are adapted, with permission, from the write-up by
**Jungwook Park**, who designed the Triton AMD `WarpPipeliner` / `ConvertWarpPipeline`
passes.

Performance (MI355X, 4096×4096):

- **FP16** — the 8-wave kernel edges the 4-wave `v9` (~1442 vs 1421 TFLOPS @ K=8192,
  ~99.8% loop MFMA).
- **BF8 / MXFP4** — the tuned 4-wave stack still leads on TFLOPS, but the 8-wave reaches
  comparable MFMA occupancy at roughly **half the VGPRs** (no-AGPR: 242–256 vs ~488).

**Reorganization** (no behavior change to the existing 4-wave kernels):

- `kernels/gemm/{a16w16,a8w8,a4w4}` moved under `kernels/gemm/intra_wave/`; the 8-wave
  kernels live under `kernels/gemm/inter_wave/`.
- `gemm/README` is split into a family overview + per-route READMEs
  (`intra_wave/README`, `inter_wave/README`) and reframed around *where scheduling
  intelligence lives*, with a 4-wave-vs-8-wave bar chart.
- The XCD-aware `get_pids` helper is shared from `kernels/gemm/utils/common.py`
  (used by both routes) instead of being duplicated per kernel.
- 8-wave perf collection is a single `scripts/collect_perf.py --kernel {a16w16,a8w8,a4w4}`
  (the per-dir `collect_perf.py` copies are gone); an `inter_wave/` kernel dir is now just
  `{matmul_kernel, README, bench.py}`, matching `intra_wave/`.

No Triton pin change — the 4-wave kernels keep the `gfx950-tutorial-v1.0` toolchain.

## 2026-07-10 — LLIR scheduler is now the `sched.barrier` variant

The v1.0 out-of-tree LLIR-scheduler plugin is the **`sched.barrier` variant**: it
pins the MFMA↔memory interleave with `llvm.amdgcn.sched.barrier(0)` after each
memory anchor, instead of the earlier in-tree scheduler that disabled LLVM's
`misched` / `post-misched`. It schedules the hot loop differently, so some measured
numbers on the pinned build differ from the earlier in-tree results:

- **Higher in-loop MFMA efficiency** at a slightly lower clock. E.g. a16w16
  `v5 + llirSched`: ~74% → ~80% MFMA efficiency, TFLOPS roughly flat.
- **v6 (loop-unroll) no longer hits the 512-VGPR spill cliff.** The old in-tree
  scheduler spilled ~104 VGPRs on v6 (~19% MFMA efficiency, ~344 TFLOPS); the
  `sched.barrier` variant keeps v6 at 508 VGPRs / ~4 spills / ~88% MFMA efficiency
  / ~1177 TFLOPS. The v6 README §4 is rewritten to this current behavior.

Per-kernel performance tables are being refreshed against the pinned v1.0 build.

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
