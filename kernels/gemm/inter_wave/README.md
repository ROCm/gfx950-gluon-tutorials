# inter_wave — 8-wave warp-pipeline GEMM kernels (gfx950)

> See the [GEMM family README](../README.md) for the directory map, the version catalog,
> and the 4-wave-vs-8-wave performance summary.

The repo carries an **8-wave warp-pipeline** version of each GEMM — [`inter_wave/a16w16/`](a16w16/), [`inter_wave/a8w8/`](a8w8/), and [`inter_wave/a4w4/`](a4w4/). These reach high MFMA utilization on the *same* problems by a different route.

Instead of the LLIR scheduler + force-agpr + amdgcnas, they launch **8 warps/CTA (2 waves/SIMD)** and schedule the hot loop at the **wave level** with `warp_pipeline_stage`: the two resident waves per SIMD are kept out of phase so one issues MFMAs while the other issues loads, then they swap (a "ping-pong"). They run with **no AGPRs** (`amdgpu-agpr-alloc=0,0` via `llvm_fn_attrs`), so the f32 accumulators live in VGPRs and **no environment variables are needed**. The theory is in [`docs/warp_pipelining.md`](../../../docs/warp_pipelining.md).

> [!IMPORTANT]
> The 4-wave `llir+force-agpr+amdgcnas` toolchain is built around the 4-wave register/schedule model and **fails register allocation at 8 waves**, so it is not used here.

## 1. The kernels

| | inter_wave/a16w16 | inter_wave/a8w8 | inter_wave/a4w4 |
|---|---|---|---|
| Data type | FP16 / BF16 | BF8 (e5m2) | MXFP4 (e2m1) |
| Versions | *(single kernel)* | *(single kernel)* | `v0_sliceMN`, `v1_combineBsc`, `v2_mfma32x32x64` |
| Tile M×N×K | 256×256×64 | 256×256×128 | 256×256×256 |
| MFMA | `mfma` `[16,16,32]` | `mfma_scaled` e5m2 `[16,16,128]` | `mfma_scaled` e2m1 `[16,16,128]` |
| Scheduling | `warp_pipeline_stage`, no-AGPR | same | same |

## 2. Performance

Measured on MI355X, gfx950, 4096×4096, Triton `gfx950-tutorial-v1.0` — a4w4 rows also need `fence_loads` PR #10840 — rocprof cold-rotating; per-SIMD loop MFMA eff):

| Kernel (final version) | K=8192 | K=16384 | K=32768 | VGPR / spills |
|---|---|---|---|---|
| inter_wave/a16w16 (fp16) | 1442 / 99.8% | 1489 / 98.1% | 1287 / 81.6% | 242 / 0 |
| inter_wave/a8w8 (BF8)    | 2853 / 99.7% | 3094 / 99.9% | 2968 / 96.8% | 256 / 13 (loop 0) |
| inter_wave/a4w4 `v1` (MXFP4)  | 4116 / 79.7% | 4630 / 79.9% | 4938 / 80.0% | 256 / 12 (loop 0) |

## 3. Running

Run them with the shared `scripts/collect_perf.py` (from the repo root, no env vars):

```bash
python scripts/collect_perf.py --kernel a16w16 --K 8192 --dtype fp16
python scripts/collect_perf.py --kernel a8w8   --K 8192
python scripts/collect_perf.py --kernel a4w4   --version 1 --K 8192
```

## 4. Where the 8-wave lands vs the 4-wave

On the v1.0 build, for **FP16**, the 8-wave kernel now edges the 4-wave `v9` by ~1.5% (1442 vs 1421 @ K=8192) — on the v1.0 build the 4-wave FP16 path sits at 1421. For **BF8**, the tuned 4-wave `llir+force-agpr+amdgcnas` leads (3232 vs 3094 @ K=16384). For **MXFP4**, `v1` (combined B-scale, the default) **beats the 4-wave *base*** at large K (4938 vs 4137 @ K=32768): combining the B scale so it transpose-reads instead of byte-shuffling deleted 118 loop `v_perm`, and the intra-stage `fence_loads` (PR #10840) lifts loop MFMA from ~57% (`v0`) to ~80%, TFLOPS +16–22%. The tuned 4-wave `llir+force-agpr+amdgcnas` (~5.2 PFLOP/s) still leads, as the loop remains LDS/scale-throughput bound. See each kernel's README ([`a16w16`](a16w16/README.md), [`a8w8`](a8w8/README.md), [`a4w4`](a4w4/README.md)) for the full breakdown.
