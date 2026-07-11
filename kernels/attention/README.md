# Attention — FAV3 (rotated 4-cluster) Gluon flash attention (gfx950)

Forward flash-attention kernel for gfx950 (CDNA4 / MI350) written in Triton
Experimental **Gluon**. It matches the pipeline architecture of the "FAV3
Unmatched" series: the hot loop is a rotated 4-cluster software pipeline built
from eight logical sub-clusters (QK / PV MFMAs, softmax numerator/denominator,
LDS local-reads of K/V, and async global→LDS copies of K/V).

Supports `bhsd` / `bshd` layouts, MHA/GQA/MQA, head dims up to 128, fp16/bf16,
causal and non-causal.

## Provenance
Ported from
[`AMD-Triton/gluon-kernels`](https://github.com/AMD-Triton/gluon-kernels)
(`kernels/cdna4/fa/`). `f16_fa_gfx950_common.py` is verbatim;
`f16_fa_gfx950_rotated_4cluster.py` is the upstream kernel with its bundled
standalone benchmark/check harness removed (our `bench.py` replaces it) — only the
Gluon kernel, autotune configs, and the `run_gluon_attention` launcher remain. Both
are excluded from this repo's black/ruff (see `pyproject.toml`) to stay easy to diff
against upstream; `bench.py` is tutorial-native and linted.

## Files
- `f16_fa_gfx950_rotated_4cluster.py` — the Gluon kernel + autotune configs + host launcher (`run_gluon_attention`).
- `f16_fa_gfx950_common.py` — shared helpers (`input_helper`, `sdpa_reference`, `compute_flops`, ...).
- `bench.py` — correctness vs the torch SDPA reference + `do_bench` TFLOPS, with the same
  RTLD_GLOBAL / plugin hooks as the GEMM tutorial `bench.py`.

## Benchmark
```bash
python bench.py                       # focus config: seqlen 8192, non-causal
python bench.py --sweep               # seqlen sweep 1024..16384
python bench.py --causal-mode both    # non-causal AND causal columns
python bench.py --seqlen 16384 --causal-mode causal
python bench.py --rocprof             # cold external timing (wrap with rocprofv3)
```
Defaults: `B=1, HQ=HK=64 (MHA), D=128, fp16, bhsd`.

## Baseline (stock `gfx950-tutorial` triton, do_bench)
`B=1, HQ=HK=64, D=128, fp16, bhsd`:

| seqlen | non-causal TFLOPS | causal TFLOPS |
|---:|---:|---:|
| 1024 | 637 | 372 |
| 2048 | 732 | 462 |
| 4096 | 785 | 589 |
| 8192 | **807** | 641 |
| 16384 | 804 | 644 |

The **8192 non-causal ≈ 807 TFLOPS** headline matches the upstream kernel's own
reference (~789). Causal is inherently ~20% lower — it computes only the lower
triangle (half the useful FLOPs) over the same pipeline/softmax/launch overhead.

> Note: unlike the GEMM tutorial, the out-of-tree **LLIR scheduler plugin does not
> help this kernel** (it is tuned for GEMM MFMA↔memory hot loops and regresses the
> FA rotated-4cluster pipeline). The Gluon kernel already schedules itself via the
> pipeline primitives.
