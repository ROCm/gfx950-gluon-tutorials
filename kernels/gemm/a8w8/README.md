# FP8 GEMM Kernel (a8w8)

This kernel applies the design principles developed in `a16w16/`, adapted for FP8 (e5m2) compute. Only the final optimized version is provided.

## 1. Directory Structure

```
a8w8/
├── matmul_kernel.py      # The kernel implementation
├── bench.py              # Benchmark and correctness test
└── README.md             # This file
```

## 2. Key Differences from FP16

The FP8 kernel uses the same optimization techniques as `a16w16/v8_beyond_hotloop`, with parameters adjusted to match FP8 MFMA instruction characteristics.

| Aspect | FP16 (a16w16) | FP8 (a8w8) |
|--------|---------------|------------|
| Tile size | 256×256×64 | 256×256×128 |
| MFMA instruction | `mfma_f16_16x16x32` | `mfma_f8_16x16x128` |
| MFMA shape | [16, 16, 32] | [16, 16, 128] |
| k_width | 8 | 32 |
| Elements per thread | 8 (16-bit × 8 = 128 bits) | 32 (8-bit × 32 = 256 bits) |
| LDS padding | [[512, 16]] | [[1024, 16], [2048, 32]] |

### 2.1 Why Larger BLOCK_K?

FP8 MFMA processes 128 elements along the K dimension per instruction (vs. 32 for FP16). To maintain the same number of MFMA instructions per iteration, BLOCK_K doubles from 64 to 128.

### 2.2 Scaled MFMA

FP8 uses `mfma_scaled`, which supports per-tensor scaling factors:

```python
acc0 = gl.amd.cdna4.mfma_scaled(a, None, "e5m2", b0, None, "e5m2", acc0)
```

The `None` arguments are placeholders for optional scale tensors. `"e5m2"` specifies the FP8 format (5-bit exponent, 2-bit mantissa).

### 2.3 LDS Layout with Double Padding

The larger K dimension in FP8 requires additional padding to avoid bank conflicts:

```python
sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
    [[1024, 16], [2048, 32]],  # Two padding rules
    ...
)
```

The dual padding `[[1024, 16], [2048, 32]]` ensures bank-conflict-free access for the 256×128 tile.

The layout can be visualized using the layout plotting tool:

![LDS Layout with Double Padding](images/lds_padding_1024-16_2048-32.png)

<details>
<summary>Command to generate this layout</summary>

```bash
python3 scripts/plot_layout.py lds \
  --tensorShape 256 128 \
  --kWidth 32 \
  --nonKDim 16 \
  --banks 64 \
  --layout padding \
  --access read \
  --swizzleVec 16 \
  --sharedLayout "[[1024, 16], [2048, 32]], [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0],[32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0], [128, 0]]" \
  --dtype fp8
```

</details>

## 3. Optimization Techniques

This kernel incorporates all optimizations from the a16w16 tutorial series:

- **XCD-aware PID remapping** — Groups adjacent tiles on the same XCD for L2 cache reuse
- **Workgroup swizzling** — Uses `GROUP_SIZE_M=4` to improve L2 cache hit rate for the A matrix
- **N-slicing** — Separate `smemB0` and `smemB1` buffers to reduce peak register pressure
- **Loop unrolling by 2** — Eliminates register copy overhead at iteration boundaries
- **3-stage pipeline** — Overlaps global loads, LDS loads, and MFMA compute
- **Interleaved epilogue** — Overlaps final MFMA with stores using `extract_slice`

For detailed explanations of these techniques, refer to the corresponding versions in `a16w16/`:
- XCD remapping and workgroup swizzling: [v8_beyond_hotloop](../a16w16/v8_beyond_hotloop/README.md)
- N-slicing: [v7_slice](../a16w16/v7_slice/README.md)
- Loop unrolling: [v6_loop_unroll](../a16w16/v6_loop_unroll/README.md)
- Local prefetch: [v5_local_prefetch](../a16w16/v5_local_prefetch/README.md)
- Global prefetch: [v4_global_prefetch](../a16w16/v4_global_prefetch/README.md)

## 4. Performance

Measured on MI355 with shape 4096×4096×16384, FP8 (e5m2):

| Configuration                   | TFLOPS | VGPRs | Spills | MFMA Eff. |
|---------------------------------|--------|-------|--------|-----------|
| base                            |   2466 |   478 |      0 |       62% |
| llirSched                       |    747 |   512 |    111 |       20% |
| llirSched + amdgcnas            |   3383 |   444 |      0 |       99% |

The [LLIR Scheduler](https://github.com/ROCm/triton/tree/matmul_4waves) alone causes register spills, which severely degrades performance. The [amdgcnas](https://github.com/ROCm/triton/tree/matmul_4waves) post-processor resolves register allocation issues, achieving 99% MFMA efficiency.

## 5. How to Run

From the `a8w8` directory:

```bash
python bench.py --K 8192
```

This runs correctness checks against `torch.matmul` and reports TFLOPS.

For accurate performance measurement with rocprof:

```bash
rocprofv3 --kernel-trace -d out -- python bench.py --K 8192 --rocprof
```

The `--rocprof` flag runs the kernel 1000 times with rotating buffers to defeat GPU caches, producing cold-cache timings representative of real workloads.
