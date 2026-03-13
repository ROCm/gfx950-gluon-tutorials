# MXFP4 GEMM Kernel (a4w4)

This kernel implements a high-performance MXFP4 (e2m1) GEMM targeting AMD MI350/355 GPUs (gfx950). It builds on the design principles from `a16w16/` and `a8w8/`, with additional complexity from per-group scaling and the microscaling (MX) data format.

## 1. Directory Structure

```
a4w4/
├── matmul_kernel.py      # The kernel implementation
├── bench.py              # Benchmark and correctness test
└── README.md             # This file
```

## 2. Key Differences from FP8

MXFP4 introduces per-group scaling factors that must be loaded, stored to LDS, and read back before MFMA can execute. This adds a new class of memory operations not present in FP8.

| Aspect | FP8 (a8w8) | MXFP4 (a4w4) |
|--------|------------|--------------|
| Data format | e5m2 (8-bit) | e2m1 (4-bit) |
| Tile size | 256x256x128 | 256x256x256 |
| MFMA instruction | `mfma_f8_16x16x128` | `mfma_scale_f32_16x16x128` |
| MFMA cycles | 32 | 16 (e2m1) or 32 (f8) |
| Scaling | None (pass `None`) | Per-group e8m0 scales for A and B |
| Scale group size | N/A | 32 elements |
| N-slicing | B split into left/right halves | Same, plus separate left/right scales |
| LDS padding | `[[1024, 16], [2048, 32]]` | `[[1024, 32]]` |

### 2.1 Microscaling (MX) Format

MXFP4 packs two 4-bit values per byte. Each group of 32 elements shares an 8-bit e8m0 scale factor. The `mfma_scaled` instruction takes both the data tile and its scale tensor:

```python
acc_left = gl.amd.cdna4.mfma_scaled(
    a=a, a_scale=a_sc_reg_buf0, a_format="e2m1",
    b=b_left, b_scale=b_sc_left_reg_buf0, b_format="e2m1",
    acc=acc_left,
)
```

### 2.2 Scale Pipeline

Scales follow a separate pipeline from tiles:

1. **GR** (Global Read): `buffer_load` scales into registers (`a_sc_buf1`, `b_sc_left_buf1`, etc.)
2. **LW** (Local Write): `smem_as.store(a_sc_buf1)` writes scales to shared memory
3. **LR** (Local Read): `load_shared_relaxed(smem_as, scale_a_layout)` reads scales back in the layout expected by MFMA

This store-to-LDS + load-from-LDS round-trip is necessary because `buffer_load` delivers scales in a blocked layout, but `mfma_scaled` requires them in a specific scale layout.

### 2.3 LDS Padding

MXFP4 uses single padding `[[1024, 32]]`, which matches the `a16w16` kernel. Both kernels share the same dotoperand layout, so the optimal padding parameters are identical despite different data types.

### 2.4 Why 16-Cycle MFMA?

With e2m1 format (cbsz > 1 or blgp > 1), the `mfma_scale_f32_16x16x128` instruction completes in 16 cycles instead of 32. This means each MFMA occupies less time, requiring more MFMAs to be interleaved with each memory operation to hide latency.

## 3. Pipeline Design

The kernel uses a double-buffered pipeline with loop unrolling by 2. Each unrolled iteration has 4 regions:

```
Region 0: DOT_left(buf0)  | wait | store b_sc, LL B_right | AC A+B_left + GR scales
Region 1: DOT_right(buf0) | wait | store scales -> buf2, LL A+B_left | AC B_right + GR b_sc
           advance ptrs
Region 2: DOT_left(buf2)  | wait | store b_sc, LL B_right | AC A+B_left + GR scales
Region 3: DOT_right(buf2) | wait | store scales -> buf0, LL A+B_left | AC B_right + GR b_sc
           advance ptrs
```

Key design choices:

- **Pre-computed `_next` offsets**: The loop advances base pointers by `2 * (BLOCK_K // 2)` once per unrolled iteration. Odd-iteration loads use `_next` offset variants instead of advancing pointers mid-iteration.
- **`commit_group()` / `wait_group()`**: Each commit group bundles tile async copies and scale buffer loads. The `wait_group(N)` primitive generates correct `vmcnt()` counters. This depends on the `asyncmark`/`wait_asyncmark` intrinsics from commit `2c36e46d84` on the `matmul_4waves` branch.

## 4. LDS Port Contention and ds_write Latency

A critical hardware limitation affects scheduling: **ds_write and buffer_load_to_lds share the same LDS write port**. The texture data unit (TD) has higher priority than the shader sequencer (SQ), so ds_write can stall for up to ~400 cycles when buffer_load_to_lds data is ready.

Furthermore, the LDS has a FIFO queue per SIMD pair with only 8 slots. When ds_write occupies the head of the queue for ~400 cycles, subsequent ds instructions fill the remaining slots. Once full, the next ds instruction stalls until the head drains.

This means **ds_read cannot hide ds_write latency** due to queue slot limitations. Only MFMA instructions (which don't use the ds queue) can effectively hide it.

The LLIR scheduler addresses this by:
1. Placing ds_write (LW) after the ds_read (LR) chunk in each region
2. Allocating leftover MFMA instructions after the first LW to provide ~400+ cycles of coverage

## 5. Interleaved Epilogue with `extract_slice`

The epilogue (last 2 K iterations) uses `extract_slice` to split the [256, K//2] input tile, [256, K//SCALE_GROUP_SIZE] scale tensor, and [256, N//2] accumulator into 4 x [64, ...] slices along the M dimension. Each sliced `mfma_scaled` is interleaved with a `buffer_store` of the previous slice's result, overlapping the final computation with output writes.

## 6. LLIR Scheduler

The kernel requires the LLIR scheduler (`TRITON_ENABLE_LLIR_SCHED=1`) for optimal performance. The scheduler operates at the LLVM IR level (pre-register allocation) and performs two transformations per region:

1. **Anchor movement**: Moves LR instructions (and associated waitcnt/barrier) from between the last LW and first GR to after the 2nd-to-last GR, placing ds_write after the ds_read chunk.
2. **MFMA interleaving**: Distributes MFMA instructions among anchor instructions using a budget-based scheme:
   - 4 MFMAs after each GR (or 2 for 32-cycle MFMA), 1 if GR is followed by LR
   - 1 MFMA after each LR
   - Leftover MFMAs after the first LW (to hide ds_write latency)
   - 2 MFMAs at the end of each region

## 7. amdgcnas Post-Processor

The kernel also benefits from the amdgcnas assembly post-processor (`TRITON_ENABLE_AMDGCN_AS=1`), which performs:

- **LICM (Loop Invariant Code Motion)**: Hoists loop-invariant instructions (LDS address calculations) to the loop prologue, with register renaming when the output register is redefined in the loop
- **`.amdhsa_accum_offset` fix**: Ensures the full VGPR range (v0-v255) is available by setting `accum_offset=256`, preventing renamed VGPRs from aliasing with AGPRs

## 8. Performance

Measured on MI355 with shape 4096x4096x32768, MXFP4 (e2m1):

| Configuration | TFLOPS | MFMA Eff. |
|---------------|--------|-----------|
| base | ~734 | — |
| llirSched + amdgcnas | 5293 | 92% |

## 9. How to Run

From the `a4w4` directory:

```bash
# With LLIR scheduler
TRITON_ENABLE_LLIR_SCHED=1 python bench.py --K 32768

# With both LLIR scheduler and amdgcnas
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 python bench.py --K 32768
```

For accurate performance measurement with rocprof:

```bash
TRITON_ENABLE_LLIR_SCHED=1 TRITON_ENABLE_AMDGCN_AS=1 \
    rocprofv3 --kernel-trace -d out -- python bench.py --K 32768 --rocprof
```
