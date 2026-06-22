# a16w16-8wave — 8-wave warp-pipeline FP16 GEMM (baseline)

An **8-wave** FP16/BF16 GEMM for gfx950, ported from
[`AMD-Triton/gluon-kernels`](https://github.com/AMD-Triton/gluon-kernels)
(`kernels/cdna4/gemm/f16_gemm_warp_pipeline_gfx950.py`) into the tutorial layout
so the tutorial's bench / rocprof / ATT tooling can drive it. This is the
**starting baseline** for 8-wave optimization work — it is not yet tuned.

## How it differs from the 4-wave tutorial kernels (`a16w16/`)

| | a16w16-8wave (this) | a16w16 v9 (4-wave) |
|---|---|---|
| Warps / CTA | **8** (`warpsPerCTA=[2,4]`) | 4 (`[2,2]`) |
| Tile M×N×K | **256×256×32** | 256×256×64 |
| LDS buffers | **3** (triple) | 2 (double) |
| Scheduling | **`warp_pipeline_stage`** (wave-level) | LLIR scheduler + amdgcnas |
| Async copy layout | `DistributedLinearLayout` | blocked/swizzled |
| B layout | pre-transposed, K contiguous | K contiguous |

> [!IMPORTANT]
> The `llir+amdgcnas` toolchain (`TRITON_ENABLE_LLIR_SCHED` /
> `TRITON_ENABLE_AMDGCN_AS`) is built around the **4-wave** register/schedule
> model and **fails register allocation** on this 8-wave kernel
> (`no registers from class available to allocate`). This kernel schedules
> itself via `warp_pipeline_stage`, so it runs in plain "base" mode.

## Files

- `matmul_kernel.py` — the `gemm_async_warp_pipeline` kernel + `matmul(a,b,c=None)`
  (a16w16-compatible) and `matmul_kernel_only(a, b_t, c)` (fair, kernel-only).
- `common.py` — PID mapping (GROUP_M + XCD remap), store epilogue, async-load helpers.
- `bench.py` — correctness + do_bench TFLOPS + `--rocprof` rotating-tensor mode.
  Same args as `a16w16/bench.py` (`--K`, `--dtype`, `--rocprof`,
  `--rotating-buffer-size`); no `--version` (single kernel).
- `collect_perf.py` — wires the existing rocprof kernel-trace (TFLOPS) and ATT
  (`scripts/run_att.py` → `scripts/process_json.py`, MFMA efficiency) to this
  kernel, plus VGPR/spill from the `.amdgcn`.

## Running

```bash
# correctness + do_bench TFLOPS (median)
python bench.py --K 8192 --dtype fp16

# TFLOPS (rocprof kernel-trace, cold cache) + MFMA efficiency (ATT) + VGPR/spill
python collect_perf.py --K 8192 --dtype fp16
```

## Baseline (MI350X, gfx950, 4096×4096×8192, fp16)

| Metric | Value |
|---|---|
| Correctness vs torch | ✅ PASS |
| do_bench TFLOPS (median) | ~711 |
| rocprof TFLOPS (cold, rotating) | ~760 |
| MFMA efficiency (per-wave, `process_json`) | 35.92% |
| VGPRs / spills | 256 / 16 |

> [!NOTE]
> **Interpreting MFMA efficiency for an 8-wave kernel.** `process_json.py`
> reports a **single wave's** MFMA-cycle fraction. The tutorial's metric is
> defined *per SIMD*; for the 4-wave kernels there is 1 wave/SIMD, so the two
> coincide (v9 ≈ 98%). This kernel runs **2 waves/SIMD**, so the per-wave 35.92%
> **undercounts** the SIMD's MFMA-unit utilization — the two co-resident waves
> interleave MFMA issue, so true unit utilization is higher (≈2× if they overlap
> well). Per loop iteration this wave does 64 `v_mfma_f32_16x16x32_f16`
> (1024 cycles) out of 2850 loop cycles. **Optimization levers:** eliminate the
> 16 VGPR spills, raise per-wave MFMA overlap, and confirm occupancy (waves/CU).
