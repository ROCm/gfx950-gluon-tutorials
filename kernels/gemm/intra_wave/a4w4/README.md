# MXFP4 GEMM Kernel (a4w4) on AMD GFX9 (Gluon)

This directory implements a high-performance **MXFP4 (e2m1) GEMM** in Gluon,
targeting **AMD MI350/355 GPUs** (gfx950). It builds directly on the techniques
from the [a16w16](../a16w16/) FP16 journey and the [a8w8](../a8w8/) BF8 kernel —
the same double-buffered async-copy tile pipeline, 3-stage pipelining, loop
unrolling, LLIR scheduler, and amdgcnas. What's new and specific to MXFP4 is the
**scale pipeline**: every group of 32 e2m1 elements shares an 8-bit e8m0 scale,
which must be loaded, laid out for the hardware, and fed to `mfma_scaled`.

If you haven't completed the a16w16 journey and reviewed a8w8, start there
first — this kernel assumes familiarity with N/M+N slicing, 3-stage pipelining,
loop unrolling, the LLIR scheduler, and amdgcnas.

## 1. Directory Structure

```
a4w4/
├── bench.py              # Benchmark and correctness test (--version selects kernel)
├── images/               # Layout and pipeline diagrams
├── v0_sliceN/            # N-slicing + LDS round-trip scale pipeline
└── v1_sliceMN/           # M+N slicing + direct-to-LDS async scale pipeline
```

## 2. How to Run

From the `a4w4` directory:

```bash
LLVM_PASS_PLUGIN_PATH=$(git rev-parse --show-toplevel)/plugins/llir_scheduler/libLlirSched.so \
LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1 \
TRITON_FORCE_MFMA_AGPR=1 \
TRITON_AMDGCNAS_PLUGIN=1 \
python bench.py --version 1 --K 32768
```

This runs correctness against a dequantized `torch.matmul` reference and reports
TFLOPS. Use `--version 0|1` to select the kernel and `--rocprof` for accurate
timing + MFMA efficiency. For the full version × config table:

```bash
python ../../../scripts/run_perf_table.py --kernel a4w4 --versions 0 1 \
  --configs base llir llir+force-agpr+amdgcnas --K 32768 --rocprof
```

## 3. The Two Versions

Both versions are kept as a deliberate progression. v0 is the direct MXFP4
adaptation — it exposes *why* the scale pipeline is hard. v1 is the refined
design that removes that difficulty and balances the kernel.

| Version | Name | Scale pipeline | Tiling | Read it for |
|---------|------|----------------|--------|-------------|
| v0 | [sliceN](v0_sliceN/README.md) | `buffer_load → local_store → local_load` (LDS round-trip) | N-only | MXFP4 fundamentals: e8m0 scales, scale layouts, `ds_read_tr`, why scales need register buffering, the `ds_write`-cost problem |
| v1 | [sliceMN](v1_sliceMN/README.md) | `buffer_load_to_lds → local_load` (no `ds_write`) | M+N | The two refinements over v0: a simpler async scale pipeline, and symmetric M+N tiling with balanced buffer loads |

Start with [v0_sliceN](v0_sliceN/README.md) for the MXFP4 concepts, then read
[v1_sliceMN](v1_sliceMN/README.md) for the two changes that make it simpler and
faster.

## 4. Performance

Measured on MI355, 4096×4096×32768, rocprof timing (1000 dispatches, last-100
average), `llir+force-agpr+amdgcnas`:

| Version | TFLOPS | MFMA Eff. |
|---------|--------|-----------|
| v0_sliceN  | 5216 | 82.5% |
| v1_sliceMN | 5820 | 93.8% |

See each version's README for the full per-config table.
