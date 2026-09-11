# Memory-Bound GEMM Kernel for AMD GFX950 (MI350/CDNA4)

This directory contains a Triton Gluon kernel optimized for **memory-bound GEMM**
on AMD GFX950 (MI350). The target workload is skinny matrix multiplication with
small M (e.g., M=32), where the kernel is bottlenecked by HBM bandwidth rather
than compute throughput.

## Problem Setup

```
C[M, N] = A[M, K] x B[K, N]    (fp16, fp32 accumulation)
```

With M=32, the arithmetic intensity is low and the kernel becomes memory-bound.
The goal is to maximize HBM bandwidth utilization on a GPU with 8.1 TB/s peak
HBM bandwidth and 256 CUs.

## Kernel Configuration

| Parameter | Value |
|-----------|-------|
| BLOCK_M   | 32    |
| BLOCK_N   | 256   |
| BLOCK_K   | 64    |
| Warps     | 4 (2x2) |
| Pipeline stages | 2 |
| A tile size | 32 x 64 = 4 KB |
| B tile size | 64 x 256 = 32 KB |
| Total per iteration | 36 KB |

## Key Optimizations

### 1. Async Copy Pipeline

The kernel uses a 2-stage software pipeline with CDNA4 async copy instructions:

```
buffer_load_dwordx4 → LDS → ds_read_b128 → MFMA
```

The prologue fills both pipeline buffers before entering the main loop. Each
iteration overlaps the current MFMA compute with the next iteration's HBM loads.

Ablation testing (removing all `ds_read_b128` and `s_waitcnt lgkmcnt` from the
assembly) confirmed that the kernel is **purely HBM-bandwidth bound** — LDS
reads are completely hidden behind HBM load latency.

### 2. `.cg` Cache Modifier on B Loads

The B tile is unique per workgroup (each workgroup loads a different slice of B
along the N dimension). The `.cg` (cache global) modifier bypasses L2 caching for
B loads, avoiding cache pollution since B data is not reused across workgroups.

The A tile is shared across all workgroups along the N dimension (redundant loads),
so it benefits from caching and uses the default cache policy.

### 3. Large BLOCK_N for TCP Utilization Efficiency

On GFX9, the TCP (Texture Cache per Pipe, 32 KB per CU) limits the total in-flight
bytes per CU. Both A and B tile data compete for TCP capacity, but only B tile
data contributes unique (non-redundant) work. The TCP utilization efficiency is:

```
tcp_efficiency = B_tile_bytes / (A_tile_bytes + B_tile_bytes)
```

With BLOCK_N=256, BLOCK_K=64:
- A tile = 4 KB, B tile = 32 KB
- TCP efficiency = 32 / (4 + 32) = **89%**

This is a significant improvement over smaller BLOCK_N values:

| BLOCK_K | BLOCK_N | A tile | B tile | TCP efficiency |
|---------|---------|--------|--------|----------------|
| 128     | 32      | 8 KB   | 8 KB   | 50%            |
| 128     | 64      | 8 KB   | 16 KB  | 67%            |
| 64      | 128     | 4 KB   | 16 KB  | 80%            |
| 64      | 256     | 4 KB   | 32 KB  | 89%            |

### 4. Per-Thread Load Width Constraint

Each thread must issue at least one full-width `buffer_load_dwordx4` (8 fp16
elements = 16 bytes). This constrains the minimum tile size:

```
BLOCK_M × BLOCK_K / num_threads ≥ 8 elements
```

With BLOCK_M=32, BLOCK_K=64, and 256 threads (4 warps × 64 lanes):
- A tile: 32 × 64 / 256 = 8 elements → 1 buffer_load per thread
- B tile: 64 × 256 / 256 = 64 elements → 4 buffer_loads per thread
- Total: 5 buffer_loads per iteration per thread

## Performance Results

Benchmark configuration: M=32, K=8192, fp16, 256 CUs on GFX950.

### Bandwidth vs. Workgroup Count (BLOCK_N=256)

N is chosen so that `N / BLOCK_N` gives the target number of workgroups:

| Workgroups | N      | Bandwidth (GB/s) |
|------------|--------|-------------------|
| 32         | 8192   | 1529              |
| 64         | 16384  | 2864              |
| 96         | 24576  | 4067              |
| 128        | 32768  | 4580              |
| 160        | 40960  | 4828              |
| 192        | 49152  | 5628              |
| 224        | 57344  | 6198              |
| 256        | 65536  | **6442**          |

Bandwidth scales nearly linearly with workgroup count, reaching **6442 GB/s**
(79% of 8.1 TB/s peak) at 256 workgroups (one per CU).

### Pipeline Stages Comparison (256 WGs, K=8192)

| Stages | Bandwidth (GB/s) |
|--------|-------------------|
| 2      | 6430              |
| 3      | 6442              |

The two configurations perform identically because the TCP (32 KB) is the
binding constraint — not the number of software pipeline buffers.

## Running the Benchmark

```bash
# Run with default sizes (M=32, N=65536, K=1024..16384)
python bench.py

# Run a specific K value
python bench.py --K 8192

# Run with rocprofv3 for external timing (uses rotating buffers to defeat caching)
rocprofv3 --kernel-trace -- python bench.py --K 8192 --rocprof
```

## Files

- `matmul_kernel.py` — Triton Gluon kernel and Python wrapper
- `bench.py` — Benchmark harness with correctness checking and rocprof support

## Further Reading

See [`docs/memory_bandwidth_model.md`](../../../docs/memory_bandwidth_model.md)
for the bandwidth model used to analyze this kernel, including the TCP utilization
efficiency framework and strategies for improvement (A-reuse via inner N-tile
loop, Split-K).
