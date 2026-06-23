# a16w16-8wave — 8-wave warp-pipeline FP16 GEMM

An **8-wave** FP16/BF16 GEMM for gfx950, ported from
[`AMD-Triton/gluon-kernels`](https://github.com/AMD-Triton/gluon-kernels)
(`kernels/cdna4/gemm/f16_gemm_warp_pipeline_gfx950.py`) into the tutorial layout
so the tutorial's bench / rocprof / ATT tooling can drive it. Versions live in
subdirs and are selected with `--version`, mirroring `a16w16/v0_naive … v9_beyond_hotloop`.

## Versions

| `--version` | dir | summary |
|---|---|---|
| `0` (default) | [`v0_BK32_nS3/`](v0_BK32_nS3/README.md) | Tuned baseline: BLOCK_K=32, **3-buffer** ring (3×-unrolled), relaxed local_load + v9-style pointer-walk base/offset, v9 XCD PID remap, no-AGPR. ~912 TFLOPS. |
| `1` | [`v1_sliceMN_BK64_nS2/`](v1_sliceMN_BK64_nS2/README.md) | M/N quadrant slicing (à la `a16w16/v8_sliceMN`), BLOCK_K=64, **2-buffer**, four separate per-quadrant LDS allocs (non-relaxed loads, no membar barrier), 0 spills. **~1039 TFLOPS** (+14% over v0). |

Each version subdir has its own README with the design rationale and perf.

## How it differs from the 4-wave tutorial kernels (`a16w16/`)

| | a16w16-8wave v0 | a16w16 v9 (4-wave) |
|---|---|---|
| Warps / CTA | **8** (`warpsPerCTA=[2,4]`) | 4 (`[2,2]`) |
| Tile M×N×K | **256×256×32** | 256×256×64 |
| LDS buffers | **3** (triple) | 2 (double) |
| Scheduling | **`warp_pipeline_stage`** (wave-level) | LLIR scheduler + amdgcnas |
| LDS allocation | one `smemA[3]`/`smemB[3]` ring | separate per-quadrant allocations |

> [!IMPORTANT]
> The `llir+amdgcnas` toolchain (`TRITON_ENABLE_LLIR_SCHED` /
> `TRITON_ENABLE_AMDGCN_AS`) is built around the **4-wave** register/schedule
> model and **fails register allocation** on the 8-wave kernels. They schedule
> themselves via `warp_pipeline_stage`, so they run in plain "base" mode and use
> `TRITON_HIP_AGPR_ALLOC="0,0"` (no-AGPR) for best occupancy.

## Files

- `common.py` — shared `get_pids` (XCD-aware PID remap + GROUP_SIZE_M swizzle) and
  `store_result` (masked store epilogue), used by every version.
- `bench.py` — correctness + do_bench TFLOPS + `--rocprof` rotating-tensor mode.
  `--version` selects the kernel; same other args as `a16w16/bench.py`
  (`--K`, `--dtype`, `--rocprof`, `--rotating-buffer-size`).
- `collect_perf.py` — rocprof kernel-trace (TFLOPS, cold/rotating) + ATT
  (`scripts/run_att.py` → `process_json.py`, MFMA efficiency) + VGPR/spill.
- `collect_counters.py` — VMEM-latency rocprof counters.
- `v0_BK32_nS3/matmul_kernel.py`, `v1_sliceMN_BK64_nS2/matmul_kernel.py` — the
  kernels; each exposes `matmul_kernel_only`, `matmul`, `MIN_K`, `KERNEL_NAME`.

## Running

```bash
# correctness + do_bench TFLOPS (median), v0
TRITON_HIP_AGPR_ALLOC="0,0" python bench.py --version 0 --K 8192 --dtype fp16

# rocprof TFLOPS (cold rotating) + MFMA efficiency (ATT) + VGPR/spill
TRITON_HIP_AGPR_ALLOC="0,0" python collect_perf.py --version 0 --K 8192 --dtype fp16
```

## Perf (MI350X, gfx950, 4096×4096×8192, fp16, no-AGPR)

| Version | rocprof TFLOPS (cold, rotating) | MFMA eff (per-SIMD) | VGPRs / spills | Correct |
|---|---|---|---|---|
| `v0_BK32_nS3` | ~912 | ~85% warm / ~83% cold | 188 / 0 | ✅ |
| `v1_sliceMN_BK64_nS2` | **~1039** | ~99.8% loop / ~94% whole-kernel | 242 / 0 | ✅ |

Correctness verified for K 512…16384, fp16 and bf16. See each version's README for
the design rationale and a full breakdown.

> [!NOTE]
> **MFMA efficiency for an 8-wave kernel.** `process_json.py` reports a **single
> wave's** MFMA-cycle fraction. The tutorial metric is *per SIMD*; the 4-wave
> kernels run 1 wave/SIMD so the two coincide (v9 ≈ 98%). These kernels run
> **2 waves/SIMD**, so per-SIMD utilization ≈ 2× the per-wave number reported by
> `process_json` (the two co-resident waves interleave MFMA issue). The reported
> figure is **loop-only**; v1's prologue/epilogue carry no MFMA, so its
> whole-kernel efficiency (~94% at K=8192) is lower than the ~99.8% loop number
> and improves as K grows and the epilogue amortizes.
