# v7_slice — Reducing Register Pressure via N-Slicing

## 1. Directory Structure

```
v7_slice/
├── matmul_kernel.py                        # The kernel implementation
├── README.md                               # This file
├── ir_dump_K8192_fp16/                     # IR dumps for analysis
├── ir_dump_K8192_fp16_llirSched/           # IR dumps with LLIR scheduler
├── ir_dump_K8192_fp16_llirSched_RA/        # IR dumps with LLIR scheduler + RA
└── ir_dump_K8192_fp16_llirSched_amdgcnas/  # IR dumps with LLIR scheduler + amdgcnas
```

## 2. Motivation

In previous versions, each iteration computes a full 256×256 output tile, requiring:
- A tile: 256×64
- B tile: 64×256
- C tile (accumulator): 256×256

As discussed in v5, local prefetch breaks the dependency between `ds_read` and MFMA by issuing `ds_read` for the next iteration while the current MFMA executes. The cost is higher register pressure: since `ds_read` and MFMA have overlapping live ranges, each input tile requires two sets of registers.

This section analyzes the register requirements and motivates the need for slicing.

### 2.1. Register Usage Analysis

**Formula:**

```
registers = (M × N × elemType × sharing_factor) / (num_warps × waveSize)
```

Where:
- `M × N`: tile size in elements
- `elemType`: size in dwords (32-bit). fp16 = 0.5, fp32 = 1.0
- `sharing_factor`: number of warps sharing the same data (from `warpsPerCTA` layout)
- `num_warps`: 4 in our kernel
- `waveSize`: 64 for gfx9

**Understanding sharing_factor:**

The `warpsPerCTA` layout determines which warps share data:
- With `warpsPerCTA = [2, 2]` (our GEMM kernel):
  - A tile: waves 0,1 share; waves 2,3 share → `sharing_factor = 2`
  - B tile: waves 0,2 share; waves 1,3 share → `sharing_factor = 2`
  - C tile: no sharing → `sharing_factor = 1`
- With `warpsPerCTA = [4, 1]` (FlashAttention):
  - A tile: `sharing_factor = 1`
  - B tile: `sharing_factor = 4`
  - C tile: `sharing_factor = 1`

**Calculation for our GEMM kernel:**

| Tile | Size | elemType | sharing_factor | Base | With prefetch |
|------|------|----------|----------------|------|---------------|
| A | 256×64 | 0.5 | 2 | 64 | 128 (×2) |
| B | 64×256 | 0.5 | 2 | 64 | 128 (×2) |
| C | 256×256 | 1.0 | 1 | 256 | 256 |

**Total: 128 + 128 + 256 = 512 registers**

The gfx9 architecture provides exactly 512 registers per SIMD. Additional registers are needed for:
- `ds_read` addresses (1 per tensor)
- `buffer_load` addresses (1 per load)
- Temporaries and scalar values

### 2.2. Block-Level vs. Instruction-Level Analysis

The 512-register count is a block-level upper bound. At the instruction level, the register allocator reuses registers when live ranges do not overlap.

For example, if a `ds_read` is scheduled *after* the MFMA that consumes its previous result, they can share registers. This is why the generated code avoids spills despite the block-level analysis suggesting we exceed the budget.

In [v3_lds](../v3_lds/README.md), we introduced the idea of thinking at the block level rather than the instruction level. The same principle applies to register analysis: we reason about register requirements at the block level, and let the backend handle fine-grained scheduling and register reuse. Instruction-level optimizations can recover a few registers at the margins, but we should not rely on them to fit a tight budget — and we don't have to.

This may seem counter-intuitive, but register allocation is tractable at the block level. We design the kernel in Gluon with sufficient headroom, and the backend simply executes. This separation of concerns — block-level design, instruction-level execution — is a recurring theme in the Gluon approach.

### 2.3. The Need for Slicing

Although register reuse prevents spills, the pressure remains high, leaving little room for:
- Additional operations (e.g., scales, bias)
- Future kernel extensions

Reuse alone is insufficient. We need to reduce register usage **by design**.

> [!IMPORTANT]
> Slicing along M or N halves the register usage for one input tile:
> - Slice along M → halve A tile registers
> - Slice along N → halve B tile registers

In this version, we slice along N. The B tile now requires 64 registers instead of 128:

**New total: 128 + 64 + 256 = 448 registers**

This provides sufficient headroom for the backend and future extensions.

> [!NOTE]
> M and N are output dimensions, not the accumulation dimension (K). Slicing along M or N doubles the number of output tiles without increasing the number of workgroups. This is the same principle underlying **persistent kernels**: each workgroup iterates over multiple output tiles rather than computing a single tile and terminating. The result is reduced register pressure per tile while maintaining the same total computation.

## 3. Slicing Design

### 3.1. Separate LDS Allocations for B

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

### 3.2. Pipeline Structure

The pipeline now has 4 regions per unrolled iteration (2 sub-iterations × 2 slices):

```
Main Loop (step = 2):
    Region 0: MFMA(A, B_left) → acc_left       [registers: a, b_left]
              load B_right from LDS             [registers: b_right]
              async_copy A, B_left for next

    Region 1: MFMA(A, B_right) → acc_right     [registers: a, b_right]
              load A, B_left from LDS           [registers: a_next, b_left]
              async_copy B_right for next

    --- Loop unroll separator ---

    Region 2: MFMA(A, B_left) → acc_left       [registers: a_next, b_left]
              load B_right from LDS             [registers: b_right]
              async_copy A, B_left for next

    Region 3: MFMA(A, B_right) → acc_right     [registers: a_next, b_right]
              load A, B_left from LDS           [registers: a, b_left]
              async_copy B_right for next
```

### 3.3. Key Insight: Staggered B Loads

The critical optimization is that we don't load both B_left and B_right at the same time:

1. Load A and B_left together (they're needed for the first MFMA)
2. While MFMA computes with B_left, load B_right
3. While MFMA computes with B_right, load next iteration's A and B_left

This staggered loading reduces the peak register usage for B operands by half.

### 3.4. Sliced Epilogue

The epilogue stores results in two separate operations:

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

| Version                        | TFLOPS | VGPRs | Copies | MFMA Eff. |
|--------------------------------|--------|-------|--------|-----------|
| v6 + LLIR scheduler            |   1119 |   500 |     51 |       88% |
| v7 + LLIR scheduler            |   1226 |   512 |    116 |       79% |
| v7 + LLIR scheduler + RA       |   1405 |   460 |      0 |       96% |
| v7 + LLIR scheduler + amdgcnas |   1335 |   460 |      0 |       98% |

**Copies** counts `v_accvgpr_*` and `v_mov` instructions inside the main loop — these are AGPR ↔ VGPR copy instructions that move data between accumulator registers and vector registers.

### 4.1. v7 + LLIR Scheduler vs. v6 + LLIR Scheduler

The v7_slice kernel is designed to reduce register pressure through N-slicing. However, comparing the assembly code reveals that v7 uses *more* registers (512 vs 500) and has *more* copy instructions inside the loop (116 vs 51). This indicates that the backend register allocator did not find the optimal register assignment for v7_slice. Despite the kernel design requiring fewer registers, the backend's allocation strategy increases overhead.

The LLIR scheduler addresses instruction scheduling based on the Gluon kernel design. Now we need something similar to address register allocation.

### 4.2. Register Allocation Workaround

The RA row is achieved using the following LLVM flags to force the OpC (input accumulator) and Dst (output accumulator) registers to use the same AGPRs for each MFMA instruction:

```
-amdgpu-mfma-vgpr-form=False
-amdgpu-agpr-alloc=256
```

By forcing all MFMA OpC and Dst operands to use the same AGPRs, the register allocator has more VGPRs available for other variables, simplifying its allocation decisions. However, this is not the optimal solution — using AGPRs for all MFMA operands means the epilogue will have the maximum number of `v_accvgpr` copy instructions before the `v_cvt` instructions, which require MFMA results as input but can only read from VGPRs.

For compute-bound GEMM with large K dimension, approximately 95% of kernel time is spent inside the loop, so the epilogue overhead is acceptable. This workaround allows us to explore other bottlenecks while the backend team works on proper solutions.

Removing all `v_accvgpr` copy instructions inside the loop increases MFMA efficiency to 96%.

![v7 RA-only bottleneck](images/v7_RAonly_bottleneck.png)

The trace shows that removing `v_accvgpr` copies also eliminates the long stalls from VALU instructions due to DIDT protection. The remaining bottleneck is scattered non-MFMA execution regions — typically several SALU instructions issued back-to-back, especially across iteration boundaries.

### 4.3. amdgcnas Assembly Processor

We developed **amdgcnas**, an assembly processor that applies peephole optimizations to eliminate the remaining gaps inside the loop. Enable it with:

```bash
TRITON_ENABLE_AMDGCN_AS=1
```

The amdgcnas pass includes the RA flags described above, plus additional peephole optimizations. With full scheduling optimization (LLIR scheduler + amdgcnas), v7 reaches **98% MFMA efficiency** — near the theoretical maximum.

### 4.4. Performance Collection

Performance is collected using:
```bash
python scripts/run_perf_table.py --kernel a16w16 --versions 6 7 --configs llir llir+amdgcnas --K 8192 --dtype fp16
```
This command can be run from anywhere in the repository. See [run_perf_table.py](../../../../scripts/README.md#run_perf_tablepy) for details.

For an explanation of MFMA efficiency and how to measure it, see [MFMA Efficiency](../../../../docs/mfma_efficiency.md).

## 5. What Comes Next

In `v8_beyond_hotloop`, we explore optimizations beyond the main loop, including prologue/epilogue improvements and additional scheduling techniques.
