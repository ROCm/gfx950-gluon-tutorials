# LLIR Scheduler — out-of-tree LLVM pass plugin

The LLIR scheduler is shipped here as an **out-of-tree LLVM pass plugin**. It classifies
each scheduling region and routes it to one of two models:

- **MFMA ↔ memory — a throughput problem.** The original model, used by the GEMM hot loops
  and described in the
  [a16w16 v5 README](../../kernels/gemm/intra_wave/a16w16/v5_local_prefetch/README.md). It
  reorders the MFMA/`ds_read`/`buffer_load` instructions in the LLVM-IR hot loop and pins
  the order with `llvm.amdgcn.sched.barrier(i32 0)` in front of each memory anchor, so LLVM's
  machine scheduler preserves the interleave (no misched-disable needed).
- **MFMA ↔ VALU — a co-execution problem.** Used by the flash-attention dot clusters
  ([`kernels/attention/`](../../kernels/attention/README.md) §6). Every vector op has to be
  assigned to a specific MFMA's 24-cycle shadow, in the right form, or it falls outside and
  costs cycles. Here the plugin does not reorder: it *declares* the intended pipeline with
  `sched_group_barrier` and lets AMDGPU's IGroupLP construct it. A second algorithm handles
  regions whose VALU demand exceeds the available shadow.

> **Learn the algorithm** → [`llir_scheduler.html`](llir_scheduler.html). The illustrated
> design reference walks through the whole pass: instruction classification, dependency-safe
> region formation, the MFMA↔memory interleaving budget and its cost model, and how the
> schedule is pinned with `sched.barrier` instead of disabling misched.

## Register-class pins
Kernels that pin MFMA accumulators with Triton's `cd_regclass` option
(`gl.amd.cdna3.mfma(..., cd_regclass="a")`, `gl.amd.cdna4.mfma_scaled(..., cd_regclass="a")`) —
in this tutorial a16w16 v7 and later, a8w8 and a4w4 — carry an empty `"=a,0"` / `"=v,0"` inline
asm on each MFMA tile's C and D. The pins only work where they sit: LLVM removes the copies they
imply only when each pin is directly next to its own MFMA and the pinned MFMAs are not reordered.
Pinning a whole `mfma()` call's accumulator at once (16 tiles, then 32 MFMAs) left 480
`v_accvgpr` copies in v9's loop.

So the MFMA ↔ memory model keeps these pins attached: when it moves an MFMA, the C pin goes
directly before it and the D pin directly after it, and it looks through pins when hoisting MFMA
inputs and sinking result extracts. It also puts a `sched.barrier(0)` after each D pin, so LLVM's
machine scheduler cannot reorder the pinned MFMAs between memory anchors; without those fences
v9's loop kept 200 copies. Kernels without pins get no pin fences. In both cases each memory
anchor's fence goes directly *in front of* the anchor, so every window between fences starts with
its memory op; with pins, that keeps a load between two pinned tiles. The original experiment
(v9 on the pre-upstream `cd_regclass` branches) is on the
[`agpr-reg-class-pinning`](https://github.com/ROCm/gfx950-gluon-tutorials/tree/agpr-reg-class-pinning/experiments/v9_agpr_pinning)
branch.

## Files
- `LlirSchedPlugin.cpp` — the pass, as a new-PassManager plugin (`llvmGetPassPluginInfo`,
  auto-inserted at the `OptimizerLast` extension point).
- `libLlirSched.so` — prebuilt plugin (see pin below).
- `llir_scheduler.html` — design reference: how the pass forms regions, sizes the
  MFMA↔memory interleave, and pins the result with `llvm.amdgcn.sched.barrier`.

## Pinned toolchain (important — ABI lock)
The `.so` is a native LLVM plugin and is **ABI-locked to the exact LLVM that
Triton is built with**. This tutorial pins Triton to [`gfx950-tutorial-v3.0`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v3.0) for both the GEMM and the
attention kernels. Triton now uses two LLVMs: the **core LLVM `b010a18d`** (see `cmake/llvm-info.json`)
runs the LLVM-IR pipeline, including this plugin, and a separately pinned AMD codegen LLVM
(`6bc4aaf6`, see `cmake/amd-llvm-info.json`) turns the result into AMDGCN. The plugin only sees
the core LLVM, and the prebuilt `.so` here is built against it. If the core LLVM pin moves,
**rebuild the `.so`** — the v2.0 `.so` (LLVM `850a2b1b`) segfaults against this pin.

## Build
Build against the same LLVM Triton uses (downloaded to `~/.triton/llvm/llvm-b010a18d-*`):

```bash
LLVM=$(dirname $(dirname $(find ~/.triton/llvm -name llvm-config | head -1)))
g++ -shared -fPIC -fvisibility=default \
    $("$LLVM/bin/llvm-config" --cxxflags) \
    -o libLlirSched.so LlirSchedPlugin.cpp
```
The plugin does **not** link LLVM; it resolves LLVM symbols from `libtriton` at
load time (see prerequisites).

## Triton prerequisites
The pin includes the source change this plugin needs,
[triton-lang/triton#10849](https://github.com/triton-lang/triton/pull/10849): Triton
*always sets the TargetMachine when an arch is given*, without which `optimize_module` runs
all of O3 with no target machine and codegen regresses (v9 loses ~11%). The one thing left to
the builder is symbol visibility:

- **Build with default visibility:** `TRITON_EXT_ENABLED=1 pip install -e .`
  (the default `-fvisibility=hidden` build exports no LLVM symbols, and
  `PassPlugin::Load` fails with `undefined symbol`).

`bench.py` handles the runtime requirement automatically: when `LLVM_PASS_PLUGIN_PATH`
is set it loads `libtriton` with `RTLD_GLOBAL` so the plugin can resolve symbols.

## Use
```bash
LLVM_PASS_PLUGIN_PATH=/abs/path/plugins/llir_scheduler/libLlirSched.so \
    python bench.py --version 8 --K 8192 --dtype fp16
```
`scripts/run_perf_table.py` wires this into the `llir` and `llir+amdgcnas` configs
automatically.

## The plugin source
`LlirSchedPlugin.cpp` is the maintained plugin source — a self-contained
new-PassManager LLVM pass plugin: it carries no Triton headers and registers
itself via `llvmGetPassPluginInfo`,
auto-inserted at the `OptimizerLast` extension point. Edit it and rebuild the
`.so` with the `g++` command in **Build** above.
