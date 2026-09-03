# inter_wave — 8-wave warp-pipeline GEMM kernels (gfx950)

> See the [GEMM family README](../README.md) for the directory map, the version catalog,
> and the 4-wave-vs-8-wave performance summary.

The repo carries an **8-wave warp-pipeline** version of each GEMM — [`inter_wave/a16w16/`](a16w16/), [`inter_wave/a8w8/`](a8w8/), and [`inter_wave/a4w4/`](a4w4/). These reach high MFMA utilization on the *same* problems by a different route.

Instead of the LLIR scheduler + force-agpr + amdgcnas, they launch **8 warps/CTA (2 waves/SIMD)** and schedule the hot loop at the **wave level** with `warp_pipeline_stage`: the two resident waves per SIMD are kept out of phase so one issues MFMAs while the other issues loads, then they swap (a "ping-pong"). They run with **no AGPRs** (`amdgpu-agpr-alloc=0,0` via `llvm_fn_attrs`), so the f32 accumulators live in VGPRs and **no environment variables are needed**. The theory is in [`docs/warp_pipelining.md`](../../../docs/warp_pipelining.md).

## 1. The kernels

| | inter_wave/a16w16 | inter_wave/a8w8 | inter_wave/a4w4 |
|---|---|---|---|
| Data type | FP16 / BF16 | BF8 (e5m2) | MXFP4 (e2m1) |
| Versions | *(single kernel)* | *(single kernel)* | `v0_sliceMN`, `v1_combineBsc`, `v2_mfma32x32x64` |
| Tile M×N×K | 256×256×64 | 256×256×128 | 256×256×256 |
| MFMA | `mfma` `[16,16,32]` | `mfma_scaled` e5m2 `[16,16,128]` | `mfma_scaled` e2m1 `[16,16,128]` |
| Scheduling | `warp_pipeline_stage`, no-AGPR | same | same |

## 2. Performance

Measured on MI355X `rocm-smi` GPU[7], 4096×4096, Triton `gfx950-tutorial-v2.1`, plain rocprofv3
with rotating tensors (which carries the always-on warp-pipeline barrier, #10840); rocprof cold-rotating (1000 dispatches, last-100 average), per-SIMD loop MFMA efficiency. One headline shape per data type — FP16 K=8192, BF8 K=16384, MXFP4 K=32768:

| Kernel | K | TFLOPS / MFMA eff | VGPR / spills |
|---|---|---|---|
| inter_wave/a16w16 (fp16) | 8192 | 1479 / 99.84% | 248 / 0 |
| inter_wave/a8w8 (BF8) | 16384 | 3151 / 96.22% | 256 / 9 (loop 0) |
| inter_wave/a4w4 `v0` (MXFP4) | 32768 | 4585 / 67.46% | 256 / 30 (loop 0) |
| inter_wave/a4w4 `v1` (MXFP4) | 32768 | 4885 / 75.10% | 256 / 14 (loop 0) |
| inter_wave/a4w4 `v2` (MXFP4) | 32768 | 5159 / 93.80% | 244 / 0 |

## 3. Running

Run them with the shared `scripts/collect_perf.py` (from the repo root, no env vars). It always
drives `rocprofv3` — there is **no `--rocprof` flag**: a `--kernel-trace` pass measures TFLOPS
(around an internal `bench.py --rocprof` run) and a `--att` pass measures per-SIMD loop MFMA
efficiency. Use `--skip-trace` / `--skip-att` to run just one.

```bash
python scripts/collect_perf.py --kernel a16w16 --K 8192  --dtype fp16
python scripts/collect_perf.py --kernel a8w8   --K 16384
python scripts/collect_perf.py --kernel a4w4   --version 1 --K 32768
```
