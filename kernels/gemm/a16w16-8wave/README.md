# a16w16-8wave — 8-wave warp-pipeline FP16/BF16 GEMM (gfx950)

An **8-wave** (8 warps/CTA → **2 waves/SIMD**) FP16/BF16 GEMM for gfx950 / MI350X.
Two versions are selected with `--version`, mirroring the `a16w16/v0_naive …
v9_beyond_hotloop` layout. Both schedule the hot loop at the **wave level** with
`warp_pipeline_stage`, and run with **no AGPRs**.

> [!IMPORTANT]
> The 4-wave `llir+amdgcnas` toolchain (`TRITON_ENABLE_LLIR_SCHED` /
> `TRITON_ENABLE_AMDGCN_AS`) is built around the 4-wave register/schedule model and
> **fails register allocation** at 8 waves. These kernels schedule themselves via
> `warp_pipeline_stage` and run in plain "base" mode with
> `TRITON_HIP_AGPR_ALLOC="0,0"` (no AGPRs) — the f32 accumulators write VGPRs
> directly, which packs tighter than the default AGPR allocation. See
> [parent compiler note](#running).

## 1. Design comparison

| | `v0_BK32_nS3` (8-wave) | `v1_sliceMN_BK64_nS2` (8-wave) | `a16w16/v9` (4-wave ref) |
|---|---|---|---|
| Warps / CTA | 8 (`[2,4]`) | 8 (`[2,4]`) | 4 (`[2,2]`) |
| Waves / SIMD | 2 | 2 | 1 |
| Tile M×N×K | 256×256×**32** | 256×256×**64** | 256×256×64 |
| M/N slicing | none (full 256×256) | **2×2 quadrants** | 2×2 quadrants |
| LDS buffers | **3** (triple ring) | **2** (double) | 2 (double) |
| K-unroll | 3× | 2× | 2× |
| LDS allocation | one `smemA[3]`/`smemB[3]` ring | **4 separate** per-quadrant | 4 separate per-quadrant |
| `local_load` | **relaxed** (dodge membar) | non-relaxed (separate allocs) | non-relaxed (separate allocs) |
| Hot-loop scheduling | `warp_pipeline_stage` | `warp_pipeline_stage` | LLIR scheduler + amdgcnas |
| XCD PID remap | yes (v9-style) | yes (v9-style) | yes |

The two 8-wave kernels share the `warp_pipeline_stage` wave-level scheduling and the
v9 XCD remap; they differ in the loop body. v0 drives a single full 256×256
accumulator from a triple-buffered ring; v1 borrows v9's four-quadrant slicing.

## 2. v0_BK32_nS3 — a tuned port

`v0` is a port of
[`f16_gemm_warp_pipeline_gfx950.py`](https://github.com/AMD-Triton/gluon-kernels/blob/main/kernels/cdna4/gemm/f16_gemm_warp_pipeline_gfx950.py)
into the tutorial layout, so the tutorial's `bench.py` / `collect_perf.py` rocprof +
ATT tooling can drive it. As ported, the loop was MFMA-starved (~36% per-wave /
~72% per-SIMD). The fix progression (rocprof cold-rotating, 4096²×8192 fp16):

> [!NOTE]
> The TFLOPS / MFMA numbers in this section (and the MI350X table in §4) were collected
> on **MI350X**, not MI355X. §4 has a separate MI355X table.

| Step | TFLOPS | MFMA (per-SIMD) | Optimization |
|---|---|---|---|
| as-ported | 760 | ~72% | baseline: 2×-unroll, default backend (f32 accumulator allocated in AGPRs), original PID map |
| + no-AGPR | 820 | 83.0% | `TRITON_HIP_AGPR_ALLOC="0,0"` forbids AGPRs so the gfx950 MFMAs write VGPRs directly; the unified register file packs tighter (256 VGPR + 16 spills → 212 VGPR / 0 spills) |
| + 3× unroll | 839 | 83.7% | unroll = ring size (3) makes the LDS buffer indices compile-time constants `0,1,2`, removing the runtime `tile % 3` and wrap-around address math |
| + v9 XCD remap | 909 | 85.2% | XCD-aware PID remap + `GROUP_SIZE_M` swizzle (from `common.py`) for L2 locality — cut measured VMEM latency substantially |
| + relaxed `local_load` | ~915 | ~85% | each mem region reads `smem.index(k+1)` (LR) then writes `smem.index(k)` (AC) — same allocation, different ring index. The membar can't disambiguate `MemDescIndexOp` sub-buffers, so it inserts a redundant `lgkmcnt(0)`+`s_barrier`; `load_shared_relaxed` carries an async-wait token the AMD `membarFilter` skips |

Result @4096²×8192: **~915 TFLOPS, ~85% loop MFMA, 188 VGPR / 0 spills**. Because the
single triple-buffered ring covers the full 256×256 tile, its `buffer_load`s cluster
in time and hit the TCP/HBM buffer-load stall at large K — MFMA drops to ~78–80% at
K ≥ 16384 (see the [v8_sliceMN TCP analysis](../a16w16/v8_sliceMN/README.md#4-buffer-load-throughput-and-tcp-limitations)).
The near-100% loop MFMA is reached by v1 below, which slices the tile to spread the
loads.

## 3. v1_sliceMN_BK64_nS2 — sliceMN × warp-pipeline

`v1` combines the hot-loop structure of
[`a16w16/v8_sliceMN`](../a16w16/v8_sliceMN/README.md) — the slicing v9 uses — with the
8-wave `warp_pipeline_stage` scheduling. The 256×256 tile is split into a **2×2 grid
of [128×128] quadrants**, each operand half-tile in its **own** double-buffered LDS
allocation (`smemA_top/bot`, `smemB_left/right`). Two consequences:

- **Non-relaxed loads, no membar barrier.** The four separate allocations have
  distinct buffer IDs, so the membar disambiguates LR vs AC by allocation — the
  loads stay plain `smem.index().load()` and carry no extra barrier. v0's relaxed
  trick is unnecessary here.
- **BLOCK_K=64, 2 buffers, loads spread across four regions** — more compute per
  async copy and no clustering, so the K≥16384 stall that hurts v0 disappears.

The loop is unrolled 2× → **8 mfma regions + 8 mem regions**, each wrapped in
`warp_pipeline_stage` with `cdna4_async.wait_group(5)` placed **before** the mfma
region. A store-side pointer-walk + a de-interleaved epilogue eliminate the
epilogue's accumulator spills (see [v1 README](v1_sliceMN_BK64_nS2/README.md#4-epilogue-register-pressure-and-the-spill-fix)).

Result @4096²×8192: **~1039 TFLOPS, ~99.8% loop MFMA, 242 VGPR / 0 spills** — and it
*gains* with K, reaching ~1083 at K=32768.

## 4. Performance

MI350X, gfx950, 4096×4096, fp16, **no-AGPR** (`TRITON_HIP_AGPR_ALLOC="0,0"`), rocprof
cold-rotating tensors:

| K | v0 TFLOPS | v0 MFMA eff | v1 TFLOPS | v1 MFMA eff |
|---|---|---|---|---|
| 8192  | 915.5 | 84.96% | **1038.6** | 99.84% |
| 16384 | 860.3 | 78.18% | **1069.4** | 99.92% |
| 32768 | 889.3 | 79.70% | **1082.6** | 99.90% |

v1 beats v0 by **+13% / +24% / +22%** at the three K values. v0 dips at large K (the
buffer-load stall); v1 climbs as the fixed prologue/epilogue cost amortizes.

### MI355X

MI355X, gfx950, 4096×4096, fp16, rocprof cold-rotating (`--rotating-buffer-size 2048`).
v0/v1 are **no-AGPR** (`TRITON_HIP_AGPR_ALLOC="0,0"`); the 4-wave `a16w16/v9` reference
uses `schedule_hint="gemm-4waves, force-agpr"` + amdgcnas (`TRITON_ENABLE_AMDGCN_AS=1`):

| K | v0 TFLOPS | v0 MFMA eff | v1 TFLOPS | v1 MFMA eff | v9 TFLOPS | v9 MFMA eff |
|---|---|---|---|---|---|---|
| 8192  | 1194.4 | 83.30% | 1442.0 | 99.84% | **1474.3** | 97.37% |
| 16384 | 1109.6 | 64.82% | 1478.8 | 99.26% | **1522.5** | 97.13% |
| 32768 | 1120.5 | 60.38% | 1289.0 | 97.72% | **1303.5** | 80.91% |

On MI355X the 4-wave `a16w16/v9` (LLIR scheduler + force-agpr + amdgcnas) edges 8-wave
v1 on TFLOPS at all three K (**+2% / +3% / +1%**); v1 keeps the highest loop MFMA-eff
(~99%). Both clear v0 by a wide margin — v0's full-tile triple-ring hits the buffer-load
stall, so its loop MFMA-eff falls to ~60–65% at K ≥ 16384 (v0 row is a 5-run median;
its absolute TFLOPS *rises* vs MI350X while eff *drops* — MI355X's faster MFMA outpaces
the unchanged memory latency, idling the units a larger fraction of the loop).

> [!NOTE]
> **MFMA eff is per-SIMD and loop-only.** `process_json.py` reports one wave's
> MFMA-cycle fraction; with 2 waves/SIMD interleaving issue, `collect_perf.py`
> doubles it for the per-SIMD figure (the 4-wave kernels run 1 wave/SIMD, factor 1).
> The number is for the hot loop — v1's prologue/epilogue carry no MFMA, so its
> *whole-kernel* efficiency is lower (~94% at K=8192) and converges toward the loop
> number as K grows.

### Trace (MI355X, K=8192)

![v0 / v1 / v9 ATT trace on MI355X](images/v0_v1_v9_trace_mi355.png)

ATT timelines for v0, v1, and v9 at 4096²×8192 on MI355X. **v1 and v9 both reach very
high MFMA utilization** — near-solid MFMA issue, with the loads hidden behind compute.
**v0 is bound by HBM latency**: its full-tile triple-ring clusters the `buffer_load`s in
time, so the MFMA units stall waiting on memory (the gaps in the v0 lane).

## 5. Running

```bash
# correctness + do_bench TFLOPS
TRITON_HIP_AGPR_ALLOC="0,0" python bench.py --version 1 --K 8192 --dtype fp16

# rocprof cold-rotating TFLOPS + MFMA efficiency (ATT) + VGPR/spill
TRITON_HIP_AGPR_ALLOC="0,0" python collect_perf.py --version 1 --K 8192 --dtype fp16

# large K needs a bigger rotating buffer to stay cold (≥3 copies)
TRITON_HIP_AGPR_ALLOC="0,0" python collect_perf.py --version 1 --K 32768 --dtype fp16 --rotating-buffer-size 2048
```

`--version 0` selects `v0_BK32_nS3`, `--version 1` selects `v1_sliceMN_BK64_nS2`. Drop
`--K` to sweep all sizes, `--dtype` to run fp16 + bf16. Clear `~/.triton/cache` after
editing a kernel.

The `TRITON_HIP_AGPR_ALLOC` env var is read by a small local hook in the AMD backend
(`backends/amd/compiler.py`, `make_llir`); it is inert unless set and does **not**
survive a Triton rebuild, so re-apply it after rebuilding Triton from source.

## 6. Files

- `common.py` — shared `get_pids` (XCD-aware PID remap + `GROUP_SIZE_M` swizzle) and
  `store_result`, used by every version.
- `bench.py` — correctness + do_bench TFLOPS + `--rocprof` rotating-tensor mode.
- `collect_perf.py` — rocprof kernel-trace TFLOPS (cold/rotating) + ATT MFMA
  efficiency + VGPR/spill.
- `collect_counters.py` — VMEM-latency rocprof counters.
- `v0_BK32_nS3/`, `v1_sliceMN_BK64_nS2/` — the kernels, each with its own README;
  every `matmul_kernel.py` exposes `matmul_kernel_only`, `matmul`, `MIN_K`,
  `KERNEL_NAME`.
