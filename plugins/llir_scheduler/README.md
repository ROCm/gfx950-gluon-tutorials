# llirSched — out-of-tree LLVM pass plugin (the attention co-execution model)

Since `gfx950-tutorial-v3.0` this plugin carries **one model**: MFMA ↔ VALU co-execution, for the
warp-pipelined attention kernels ([`kernels/attention/`](../../kernels/attention/README.md) §6).
Every vector op of a dot cluster has to land in a specific MFMA's 24-cycle shadow, in the right
form, or it falls outside and costs cycles. The plugin does not reorder: it *declares* the intended
pipeline with `sched_group_barrier` and lets AMDGPU's IGroupLP construct it. A second algorithm
handles regions whose VALU demand exceeds the available shadow, and the memory stages are paced
with two `s_nop`s at their head. A kernel is scheduled span by span between the
`llvm.amdgcn.sched.barrier`s that `ConvertWarpPipeline` emits at its stage boundaries; a kernel
without such barriers is left untouched.

The **MFMA ↔ memory model** — the throughput interleave of the GEMM hot loops that this plugin
carried from `v1.0` to `v2.3` — is now part of Triton itself:
[`third_party/amd/lib/Target/MFMASchedule/MFMASchedule.cpp`](https://github.com/triton-lang/triton/pull/12209),
opt-in per kernel with `schedule_hint="mfma-schedule"`. The intra_wave GEMM kernels pass that option
in their launch and do not load this plugin; see
[`kernels/gemm/intra_wave/README.md` §2.1](../../kernels/gemm/intra_wave/README.md#21-triton-build-the-mfma-scheduler-and-the-amdgcnas-plugin).

> **Learn the algorithm** → [`llir_scheduler.html`](llir_scheduler.html). The illustrated design
> reference covers both models: instruction classification, dependency-safe region formation, the
> MFMA↔memory interleaving budget and its cost model, how the schedule is pinned with
> `sched.barrier` (§1–§7, the in-tree pass), and the region routing and co-execution declaration
> (§8–§9, this plugin).

## Files
- `LlirSchedPlugin.cpp` — the pass, as a new-PassManager plugin (`llvmGetPassPluginInfo`,
  auto-inserted at the `OptimizerLast` extension point).
- `libLlirSched.so` — prebuilt plugin (see pin below).
- `llir_scheduler.html` — design reference.

## Pinned toolchain (important — ABI lock)
The `.so` is a native LLVM plugin and is **ABI-locked to the exact LLVM that Triton is built
with**. This tutorial pins Triton to
[`gfx950-tutorial-v3.0`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v3.0).
Triton uses two LLVMs: the **core LLVM `b010a18d`** (see `cmake/llvm-info.json`) runs the LLVM-IR
pipeline, including this plugin, and a separately pinned AMD codegen LLVM (`6bc4aaf6`, see
`cmake/amd-llvm-info.json`) turns the result into AMDGCN. The plugin only sees the core LLVM, and
the prebuilt `.so` here is built against it. If the core LLVM pin moves, **rebuild the `.so`**.

## Build
Build against the same LLVM Triton uses (downloaded to `~/.triton/llvm/llvm-b010a18d-*`):

```bash
LLVM=$(dirname $(dirname $(find ~/.triton/llvm -name llvm-config | head -1)))
g++ -shared -fPIC -fvisibility=default \
    $("$LLVM/bin/llvm-config" --cxxflags) \
    -o libLlirSched.so LlirSchedPlugin.cpp
```
The plugin does **not** link LLVM; it resolves LLVM symbols from `libtriton` at load time (see
prerequisites).

## Triton prerequisites
- **Build with default visibility:** `TRITON_EXT_ENABLED=1 pip install -e .` (the default
  `-fvisibility=hidden` build exports no LLVM symbols, and `PassPlugin::Load` fails with
  `undefined symbol`). Only the attention kernels need this; the GEMM kernels run on a plain build.
- The pin keeps the target machine when a pass plugin is loaded
  ([triton-lang/triton#10849](https://github.com/triton-lang/triton/pull/10849)), so the plugin
  runs inside the real O3 pipeline.

`kernels/attention/bench.py` loads `libtriton` with `RTLD_GLOBAL` whenever `LLVM_PASS_PLUGIN_PATH`
is set, so the plugin can resolve symbols.

## Use
From `kernels/attention/`:

```bash
FA_MODULE=fmha_v4 DISABLE_LLVM_OPT=disable-machine-sink \
LLVM_PASS_PLUGIN_PATH=$PWD/../../plugins/llir_scheduler/libLlirSched.so \
python bench.py --batch 32 --hq 8 --hk 8 --seqlen 8192
```
The plugin reads no environment variables of its own; debug output goes through `LLVM_DEBUG`
(`TRITON_LLVM_DEBUG_ONLY=tritonamdgpu-llir-schedule`).

## History
- `v1.0`–`v2.3`: both models in one plugin; the GEMM kernels loaded it through
  `LLVM_PASS_PLUGIN_PATH` (`run_perf_table.py`'s `llir` config).
- `v3.0`: the MFMA ↔ memory model moved upstream as Triton's MFMA scheduler; the plugin keeps the
  co-execution model only, with the review-round fixes of the upstream version (span-by-span
  classification, no environment knobs — the memory-stage pacing is fixed at two `s_nop`s, so the
  `LLIRSCHED_WP_MEMNOP` ablation of earlier revisions is gone).
