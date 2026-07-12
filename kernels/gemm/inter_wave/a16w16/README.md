# inter_wave/a16w16 — 8-wave warp-pipeline FP16/BF16 GEMM (gfx950)

An **8-wave** (8 warps/CTA → **2 waves/SIMD**) FP16/BF16 GEMM for gfx950 / MI350X.
It schedules the hot loop at the **wave level** with `warp_pipeline_stage`, slices the
256×256 output tile into a **2×2 grid of [128×128] quadrants**, and runs with **no
AGPRs**.

> For the theory behind `warp_pipeline_stage` — the phase-shifted, two-group
> schedule, why it raises MFMA utilization, and the barrier/membar rules this
> kernel follows — see [`docs/warp_pipelining.md`](../../../../docs/warp_pipelining.md).

> [!IMPORTANT]
> The 4-wave `llir+amdgcnas` toolchain (`TRITON_ENABLE_LLIR_SCHED` /
> `TRITON_ENABLE_AMDGCN_AS`) is built around the 4-wave register/schedule model and
> **fails register allocation** at 8 waves. This kernel schedules itself via
> `warp_pipeline_stage` and runs with **no AGPRs** — it sets `amdgpu-agpr-alloc=0,0`
> at launch via Triton's built-in `llvm_fn_attrs` option, so the f32 accumulators write
> VGPRs directly, which packs tighter than the default AGPR allocation.

## 1. Design

The kernel (`a16w16_kernel` — sliceMN, `BLOCK_K=64`, 2-buffer) combines the hot-loop
structure of the 4-wave [`a16w16/v8_sliceMN`](../../intra_wave/a16w16/v8_sliceMN/README.md)
— the slicing `v9` uses — with 8-wave `warp_pipeline_stage` wave-level scheduling. The
256×256 tile is split into a **2×2 grid of [128×128] quadrants**, each operand half-tile
in its **own** double-buffered LDS allocation (`smemA_top/bot`, `smemB_left/right`).

| | **this kernel** (8-wave) | `a16w16/v9` (4-wave ref) |
|---|---|---|
| Warps / CTA | 8 (`[2,4]`) | 4 (`[2,2]`) |
| Waves / SIMD | 2 | 1 |
| Tile M×N×K | 256×256×64 | 256×256×64 |
| M/N slicing | 2×2 quadrants | 2×2 quadrants |
| LDS buffers | 2 (double) | 2 (double) |
| LDS allocation | 4 separate per-quadrant | 4 separate per-quadrant |
| K-unroll | 2× | 2× |
| `local_load` | non-relaxed (separate allocs) | non-relaxed (separate allocs) |
| Hot-loop scheduling | `warp_pipeline_stage` | LLIR scheduler + amdgcnas |
| XCD PID remap | yes (v9-style) | yes |

Two consequences of the four separate allocations:

- **Non-relaxed loads, no membar barrier.** Because `smemA_top`, `smemA_bot`,
  `smemB_left`, `smemB_right` are four *separate* allocations with distinct buffer IDs,
  the membar disambiguates the load-read (LR) from the async-copy write (AC) by
  allocation — so the loads stay plain (non-relaxed) `smem.index().load()` and carry no
  extra `s_barrier`.
- **`BLOCK_K=64` with only 2 buffers.** A larger K-step means more compute per async copy
  to hide its latency, and the four-region structure spreads the buffer loads across the
  K-step — so the TCP/HBM buffer-load stall that hits a full-tile ring at K≥16384 (the
  [v8_sliceMN TCP analysis](../../intra_wave/a16w16/v8_sliceMN/README.md#4-buffer-load-throughput-and-tcp-limitations))
  does not appear here.

## 2. Loop structure

The output is split into four quadrants, each its own f32 accumulator:

```
acc_tl += DOT(A_top, B_left)     acc_tr += DOT(A_top, B_right)
acc_bl += DOT(A_bot, B_left)     acc_br += DOT(A_bot, B_right)
```

The loop is unrolled 2× → **8 mfma regions + 8 mem regions**. Each region is wrapped in
`warp_pipeline_stage`, with `cdna4_async.wait_group(5)` placed **before** the mfma region
(so the async copy whose data the upcoming load needs is drained ahead of use, and the
wait sits at an empty stage-cluster boundary so `WarpPipeliner` accepts it):

```
cdna4_async.wait_group(5)
with warp_pipeline_stage("mfma", priority=0):
    acc_X = mfma(operand_a, operand_b, acc_X)
with warp_pipeline_stage("mem", priority=1):
    operand = smem.index(buf).load(dotOp)      # LR for the next region
    cdna4_async.buffer_load_to_shared(...)     # AC refills this buffer
    cdna4_async.commit_group()
```

The 8-wave global-load layouts are the 4-wave `v9` layouts with **one extra warp
dimension** (tiling M for A, N for B, since `warpsPerCTA=[2,4]` = 8 warps); the shared /
dot-operand / MFMA layouts are reused unchanged. B is pre-transposed to `(N, K)` and fed
as a logical `(K, N)` operand via strides, so K is contiguous for the async copy.

The 8 mfma regions and 8 mem regions are interleaved across the two co-resident wave
groups (**ping-pong**) via `warp_pipeline_stage`: one wave group's MFMAs issue while the
other group's loads are in flight, then they swap. Each region's `wait_group(5)` drains
the async copy whose data the upcoming load needs, so the load → MFMA dependency is
satisfied without stalling the issue pipe. The figure below shows the full unrolled
schedule — 8 mfma / 8 mem regions over the four quadrants × 2 buffers.

<p align="center">
  <img src="images/new_8wave_pingpong_design.png" alt="8-wave warp ping-pong loop design" width="640">
</p>

## 3. Epilogue: register pressure and the spill fix

At 8 waves there are **2 waves/SIMD**, so the per-wave budget is 256 VGPR. The four live
f32 `[128×128]` accumulators alone are 128 VGPR; with the dot operands (~96) and the store
machinery, the **epilogue** (not the loop) overflowed and spilled the accumulators to
scratch — 67 spills, a 32,240-cycle epilogue (~11% of the kernel at K=8192). The loop body
itself never spills (it runs at ~99.8% MFMA).

Two kernel-side changes bring spills to **0**:

1. **Store-side pointer-walk.** All four quadrants have identical internal structure, so
   they share **one** within-quadrant offset tensor plus four **scalar** base pointers
   (`c_tl/bl/tr/br_base = c_base + const`), instead of four full `[128×128]` offset
   tensors (~32 VGPR each).
2. **De-interleave the epilogue.** Finish all four final mfmas *before* the convert+store
   phase. This lets the dot operands die first, so only the four accumulators (+ one
   in-flight `convert_layout`) are live during the stores.

> [!NOTE]
> The store downcasts f32→f16 (`v_cvt_pk_f16_f32`) **before** `convert_layout`, so the
> layout shuffle through LDS already moves f16, not f32. The remaining ~16,660-cycle
> epilogue is the inherent `convert_layout` LDS round-trip (`mfmaLayout → BlockedLayout`)
> plus the stores.

## 4. Performance

MI355X, gfx950, 4096×4096, fp16, **no-AGPR** (`amdgpu-agpr-alloc=0,0` via `llvm_fn_attrs`),
current build (Triton 3.8.0), rocprof cold-rotating (`--rotating-buffer-size 2048`). The
8-wave kernel vs the 4-wave `a16w16/v9` reference (LLIR scheduler + force-agpr + amdgcnas,
`TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1`):

| K | 8-wave TFLOPS | 8-wave MFMA eff | v9 TFLOPS | v9 MFMA eff |
|---|---|---|---|---|
| 8192  | 1446 | 99.8% | **1485** | 97.0% |
| 16384 | 1495 | 99.3% | **1532** | 97.4% |
| 32768 | 1287 | 92.3% | **1310** | ~81% |

VGPRs / spills: **242 / 0** (loop-spill-free).

The 4-wave `a16w16/v9` (LLIR scheduler + force-agpr + amdgcnas) edges the 8-wave kernel on
TFLOPS at all three K (**~+3% / +2% / +2%**); the 8-wave keeps the highest loop MFMA-eff
(~99% at K ≤ 16384, dipping to ~92% at K=32768 as the buffer-load stall sets in).
(MFMA-eff is a single-dispatch ATT reading — treat the last digit as noise.)

> [!NOTE]
> **MFMA eff is per-SIMD and loop-only.** `process_json.py` reports one wave's MFMA-cycle
> fraction; with 2 waves/SIMD interleaving issue, `scripts/collect_perf.py` doubles it for the
> per-SIMD figure (the 4-wave kernels run 1 wave/SIMD, factor 1). The number is for the hot
> loop — the prologue/epilogue carry no MFMA, so the *whole-kernel* efficiency is lower
> (~94% at K=8192) and converges toward the loop number as K grows.

### Trace (MI355X, K=8192)

![8-wave sliceMN vs 4-wave v9 ATT trace on MI355X](images/v0_v1_v9_trace_mi355.png)

ATT timelines at 4096²×8192 on MI355X. **Left:** an earlier full-tile triple-buffered ring
prototype — bound by HBM latency, its loads cluster in time so the MFMA units stall
(the gaps in the lane). **Middle:** this 8-wave sliceMN kernel. **Right:** the 4-wave `v9`.
Both the sliceMN kernel and `v9` reach **near-solid MFMA issue**, with the loads hidden
behind compute — which is exactly why slicing the tile (to spread the loads) beats the
full-tile ring.

## 5. Running

```bash
# correctness + do_bench TFLOPS (from this kernel dir)
python bench.py --K 8192 --dtype fp16

# rocprof cold-rotating TFLOPS + MFMA efficiency (ATT) + VGPR/spill (from the repo root)
python scripts/collect_perf.py --kernel a16w16 --K 8192 --dtype fp16

# large K needs a bigger rotating buffer to stay cold (≥3 copies)
python scripts/collect_perf.py --kernel a16w16 --K 32768 --dtype fp16 --rotating-buffer-size 2048
```

Drop `--K` to sweep all sizes, `--dtype` to run fp16 + bf16. Clear `~/.triton/cache` after
editing the kernel.

The **no-AGPR** setting is baked into the kernel's launch via Triton's built-in
`llvm_fn_attrs=(("amdgpu-agpr-alloc","0,0"),)` compile option — no env var or compiler
patch needed, and it survives a Triton rebuild.

## 6. Files

- `matmul_kernel.py` — the kernel; exposes `a16w16_kernel` (the jit kernel),
  `matmul_kernel_only` / `matmul` (launch wrappers), `MIN_K`, `KERNEL_NAME`.
- `get_pids` (XCD-aware PID remap + `GROUP_SIZE_M` swizzle) is imported from the shared
  [`kernels/gemm/utils/common.py`](../../utils/common.py) (`bench.py` puts it on the path).
- `bench.py` — correctness + do_bench TFLOPS + `--rocprof` rotating-tensor mode.
- Perf is collected with the shared [`scripts/collect_perf.py`](../../../../scripts/collect_perf.py)
  (`--kernel a16w16`); VMEM-latency counters with
  [`scripts/collect_counters.py`](../../../../scripts/collect_counters.py).
