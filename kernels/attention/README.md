# Attention — FAV3 (rotated 4-cluster) Gluon flash attention (gfx950)

Forward flash-attention kernel for gfx950 (CDNA4 / MI350) written in Triton
Experimental **Gluon**. It matches the pipeline architecture of the "FAV3
Unmatched" series: the hot loop is a rotated 4-cluster software pipeline built
from eight logical sub-clusters (QK / PV MFMAs, softmax numerator/denominator,
LDS local-reads of K/V, and async global→LDS copies of K/V).

This tutorial copy is **simplified down to the single most-performant path**:
non-causal, head dim 128, fp16/bf16, `bhsd` / `bshd` layouts, MHA/GQA/MQA, K
sequence length a multiple of `BLOCK_N` (64). The full upstream kernel also
handles causal masking, ragged tails, other head dims, and a wide autotune space
— all removed here to keep the tutorial focused (see Provenance).

## Provenance
Ported from
[`AMD-Triton/gluon-kernels`](https://github.com/AMD-Triton/gluon-kernels)
(`kernels/cdna4/fa/`). `f16_fa_gfx950_common.py` is verbatim.
`f16_fa_gfx950_rotated_4cluster.py` is the upstream kernel reduced to the single
best config for the focus shape (D=128, non-causal, `BLOCK_M=256`, `BLOCK_N=64`,
8 warps): the per-`(D, BLOCK_N, warps)` layout dispatch, causal / masked-tail
scheduling, preload-all + non-pipelined fallbacks, head-dim padding, and the
multi-config autotune space were all dropped, the pipelined loop was inlined into
one flat `gluon_attn_fwd`, and dead args (`PRE_LOAD_V`, `MMA_TYPE`, `NUM_STAGES`,
`ACTUAL_BLOCK_DMODEL`) were removed. The full version lives in git history and upstream. Both vendored files are excluded from
this repo's black/ruff (see `pyproject.toml`); `bench.py` is tutorial-native and
linted.

## Files
- `f16_fa_gfx950_rotated_4cluster.py` — the Gluon kernel + single autotune config + host launcher (`run_gluon_attention`).
- `f16_fa_gfx950_common.py` — shared helpers (`input_helper`, `sdpa_reference`, `compute_flops`, ...).
- `bench.py` — correctness vs the torch SDPA reference + `do_bench` TFLOPS, with the same
  RTLD_GLOBAL / plugin hooks as the GEMM tutorial `bench.py`.

## Benchmark
```bash
python bench.py                       # focus config: seqlen 8192, non-causal
python bench.py --sweep               # seqlen sweep 1024..16384
python bench.py --seqlen 16384        # a single seqlen
python bench.py --rocprof             # cold external timing (wrap with rocprofv3)
```
Defaults: `B=1, HQ=HK=64 (MHA), D=128, fp16, bhsd`, non-causal.

## Baseline (stock `gfx950-tutorial` triton, do_bench)
`B=1, HQ=HK=64, D=128, fp16, bhsd`, non-causal:

| seqlen | non-causal TFLOPS |
|---:|---:|
| 1024 | 495 |
| 2048 | 739 |
| 4096 | 794 |
| 8192 | **807** |
| 16384 | 801 |

The **8192 non-causal ≈ 807 TFLOPS** headline matches the upstream kernel's own
reference (~789). The single tuned config targets the large-seqlen focus shape, so
the smallest seqlen (1024) is left on the table — upstream's autotune picks a
different config there.

> Note: unlike the GEMM tutorial, the out-of-tree **LLIR scheduler plugin does not
> help this kernel** (it is tuned for GEMM MFMA↔memory hot loops and regresses the
> FA rotated-4cluster pipeline). The Gluon kernel already schedules itself via the
> pipeline primitives.
