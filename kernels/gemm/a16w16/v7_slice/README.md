# v7_slice — Reducing Register Pressure via N-Slicing

## 1. Directory Structure

```
v7_slice/
├── matmul_kernel.py                  # The kernel implementation
├── README.md                         # This file
├── ir_dump_K4096_fp16/               # IR dumps for analysis
├── ir_dump_K4096_fp16_llirSched/     # IR dumps with llirSched
└── ir_dump_K4096_fp16_llirSched_amdgcnas/  # IR dumps with llirSched + amdgcnas
```

## 2. Motivation

In previous versions, each iteration computes a full 256×256 output tile. This requires:
- Loading a full 256×64 A tile
- Loading a full 64×256 B tile
- Maintaining a 256×256 accumulator

The B tile and accumulator together consume significant register space, limiting the scheduler's flexibility to interleave instructions effectively.

> [!IMPORTANT]
> By slicing the B matrix along the N dimension into left and right halves (each 64×128), we reduce the live register footprint at any point in time. This gives the scheduler more room to overlap memory operations with compute.

## 3. Slicing Design

### 3.1 Separate LDS Allocations for B

Instead of a single B buffer, we allocate separate buffers for left and right halves:

```python
smemB_left = gl.allocate_shared_memory(
    b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
)
smemB_right = gl.allocate_shared_memory(
    b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
)
```

Similarly, we maintain two separate accumulators:

```python
acc_left = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)
acc_right = gl.zeros((BLOCK_M, BLOCK_N // 2), gl.float32, mfmaLayout)
```

### 3.2 Pipeline Structure

The pipeline now has 4 regions per unrolled iteration (2 sub-iterations × 2 slices):

```
Main Loop (step = 2):
    Region 0: MFMA(A, B_left) → acc_left
              load B_right from LDS
              async_copy A, B_left for next

    Region 1: MFMA(A, B_right) → acc_right
              load A, B_left from LDS (next buffer)
              async_copy B_right for next

    --- Loop unroll separator ---

    Region 2: MFMA(A, B_left) → acc_left
              load B_right from LDS
              async_copy A, B_left for next

    Region 3: MFMA(A, B_right) → acc_right
              load A, B_left from LDS (next buffer)
              async_copy B_right for next
```

### 3.3 Key Insight: Staggered B Loads

The critical optimization is that we don't load both B_left and B_right at the same time:

1. Load A and B_left together (they're needed for the first MFMA)
2. While MFMA computes with B_left, load B_right
3. While MFMA computes with B_right, load next iteration's A and B_left

This staggered loading reduces the peak register usage for B operands by half.

### 3.4 Sliced Epilogue

The epilogue also stores results in two separate operations:

```python
## Store left half
acc_left = gl.amd.cdna3.mfma(a, b_left, acc_left)
c_left = acc_left.to(a_ptr.dtype.element_ty)
gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_left_offsets, stored_value=c_left)

## Store right half
acc_right = gl.amd.cdna3.mfma(a, b_right, acc_right)
c_right = acc_right.to(a_ptr.dtype.element_ty)
gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_right_offsets, stored_value=c_right)
```

This allows the store for `acc_left` to overlap with the final MFMA for `acc_right`.

## 4. Performance Analysis

| Version | TFLOPS | MFMA Eff. |
|---------|--------|-----------|
| v6      |   1025 |       61% |
| v6 + llirSched |   1105 |       84% |
| v7      |   1088 |       64% |
| v7 + llirSched |   1273 |       77% |
| v7 + llirSched + amdgcnas |   1378 |       97% |

With full scheduling optimization (llirSched + amdgcnas), v7 achieves **97% MFMA efficiency**—near the theoretical maximum.

Performance is collected using:
```bash
python bench.py --K 8192 --dtype fp16
```

For an explanation of MFMA efficiency and how to measure it, see [MFMA Efficiency](../../../../docs/mfma_efficiency.md).

## 5. What Comes Next

In `v8_beyond_hotloop`, we explore optimizations beyond the main loop, including prologue/epilogue improvements and additional scheduling techniques.
