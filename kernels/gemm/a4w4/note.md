# vmcnt() Issue with Mixed buffer_load and buffer_load_to_lds

## Problem

When `buffer_load` (regular load to registers) and `buffer_load_to_lds` (async
copy direct-to-LDS) coexist in the same loop, the generated `s_waitcnt vmcnt(N)`
values are incorrect. Both instruction types share the same hardware vmcnt
counter, but Triton and LLVM track them independently.

### Expected loop structure

```
buffer_load scale_a             -- vmcnt +1 (to registers)
buffer_load scale_b             -- vmcnt +1 (to registers)
buffer_load_to_lds x16          -- vmcnt +16 (to LDS)
s_waitcnt vmcnt(18)             -- wait for prev iter's 18 loads
ds_read (local_load tiles)
ds_read (local_load scales)
mfma (dot)
s_waitcnt vmcnt(16)             -- wait for 2 scale loads, keep 16 tile loads in flight
ds_write (store scales to LDS)
```

### Actual generated code

```
s_waitcnt vmcnt(16)             -- WRONG: should be vmcnt(18)
...
s_waitcnt vmcnt(0)              -- WRONG: should be vmcnt(16)
```

## Root Cause

Two independent systems track vmcnt, and neither accounts for the other:

### 1. Triton's `UpdateAsyncWaitCount` pass ignores `buffer_load`

File: `OAI-triton/third_party/amd/lib/TritonAMDGPUTransforms/UpdateAsyncWaitCount.cpp`

`getOpNumberOfAsyncCopyInstructions()` (line 87-120) only counts:
- `AsyncCopyGlobalToLocalOp` (buffer_load_to_lds via tt API)
- `BufferLoadToLocalOp` (buffer_load_to_lds via gluon API)
- `AsyncCopyLocalToGlobalOp`

It returns **0** for `BufferLoadOp` (regular buffer_load to registers). So when
`wait_group(1)` walks backward to compute vmcnt, it only sees 16
buffer_load_to_lds and emits `vmcnt(16)` instead of `vmcnt(18)`.

### 2. LLVM's `SIInsertWaitcnts` inserts conservative vmcnt(0)

File: `llvm-project/llvm/lib/Target/AMDGPU/SIInsertWaitcnts.cpp`

LLVM tracks register def-use chains and should compute `vmcnt(16)` before
`ds_write` (16 buffer_load_to_lds ops between the buffer_load def and its use).
However, it emits `vmcnt(0)` — likely because the Triton-emitted `s_waitcnt`
from `wait_group` resets the score bracket, and across loop back-edges LLVM
conservatively waits for everything.

### Key architectural distinction

- **`buffer_load_to_lds`** is asynchronous — writes to LDS with no register
  output. Requires explicit software sync via `commit_group`/`wait_group`.
- **`buffer_load`** is synchronous — writes to registers with explicit
  consumers. LLVM's register def-use tracking should handle synchronization.
- Both share the **same hardware vmcnt counter**.

## Proposed Solution

### Fix 1: Count `BufferLoadOp` in `UpdateAsyncWaitCount` (Triton)

Add a case for `amdgpu::BufferLoadOp` in `getOpNumberOfAsyncCopyInstructions()`:

```cpp
} else if (auto loadOp = dyn_cast<amdgpu::BufferLoadOp>(op)) {
    // buffer_load shares the vmcnt counter with buffer_load_to_lds.
    // Count instructions for correct vmcnt arithmetic.
    auto ptrType = cast<RankedTensorType>(LLVM::AMD::getPointerTypeWithShape(
        loadOp.getPtr(), loadOp.getOffsets()));
    int contig = LLVM::AMD::getVectorSize(loadOp.getPtr(),
                                          loadOp.getOffsets(), axisInfo);
    contig = std::max(contig, static_cast<int>(loadOp.getContiguity()));
    unsigned numElems = gpu::getTotalElemsPerThread(ptrType);
    return std::max(1, static_cast<int>(numElems) / contig);
}
```

This is purely a counting fix — `buffer_load` is NOT made "async". The
`commit_group`/`wait_group` system still semantically manages only async ops.
The vmcnt value just needs to account for all in-flight vmem operations since
they share a hardware counter.

### Fix 2: Fix LLVM's conservative vmcnt for register dependencies

The `vmcnt(0)` before `ds_write` should be `vmcnt(16)` — LLVM's register
tracking should compute this naturally. The fix belongs in LLVM's
`SIInsertWaitcnts` pass, ensuring it properly tracks `buffer_load_to_lds` ops
(which increment vmcnt but don't define registers) when computing wait counts
for register dependencies, especially across loop back-edges.

**Do NOT use `wait_group()` before `ds_write` to synchronize `buffer_load`
results** — this would break the programming model by conflating async and sync
synchronization semantics.

## Relevant Files

| File | Role |
|------|------|
| `UpdateAsyncWaitCount.cpp` | Triton pass that computes vmcnt for `wait_group` |
| `LoadStoreOpToLLVM.cpp:638-720` | `BufferLoadOp` lowering (no alias metadata) |
| `LoadStoreOpToLLVM.cpp:722-869` | `BufferLoadToLocalOp` lowering (has alias scope) |
| `LoadStoreOpToLLVM.cpp:2287-2342` | `AsyncWaitOp` → `s_waitcnt vmcnt(N)` |
| `AsyncUtility.h` | Alias scope mechanism for async ops |
| `SIInsertWaitcnts.cpp` | LLVM pass that inserts waitcnt instructions |

## Note on `commit_group()`

On CDNA, `commit_group()` emits **no hardware instruction** — it is purely a
software marker in the IR that `UpdateAsyncWaitCount` uses as a boundary when
walking backward to count in-flight operations. `wait_group(N)` walks past N+1
commit groups and sums up async instruction counts, then emits
`s_waitcnt vmcnt(count)`.

---

# LLIR Scheduler Issues for MXFP4 Kernel

## Problem

When running with `TRITON_ENABLE_LLIR_SCHED=1`, the LLIR scheduler detects 2
clusters in the main loop: Region 0 with 1 MFMA and Region 1 with 127 MFMA.
All 128 MFMAs should be in a single cluster.

File: `OAI-triton/third_party/amd/lib/TritonAMDGPUToLLVM/LLIRSchedule.cpp`

### Root cause

`assignRegions()` splits regions when it encounters a memory op
(`BufferLoadLDS` or `LDSLoad`) between two MFMAs. The `SeenMemoryOps` flag is
set by `buffer_load_lds` instructions at the top of the loop (before any MFMA).
When MFMA #1 arrives, `RegionStart` is `nullptr` so the split condition
`(SeenMemoryOps && RegionStart != nullptr)` is false — MFMA #1 starts Region 0.
But `SeenMemoryOps` remains `true`. When MFMA #2 arrives, both conditions are
met and it incorrectly starts a new Region 1.

### What needs to be fixed

1. **Anchor ops are incomplete.** `classifySchedInst()` only recognizes
   `BufferLoadLDS`, `BufferStore`, and `LDSLoad` (addrspace 3 loads). The
   following instruction types should also be included as anchor ops:
   - `buffer_load` (regular load to registers, e.g. scale loads)
   - `ds_write` (LDS store, e.g. scale store to shared memory)
   - `ds_read_b64_tr` (transpose read from LDS for dot operands)

2. **MFMA cycle count for scaled variants.** `mfma_scale_f32_16x16x128_f8f6f4`
   with `cbsz > 1` or `blgp > 1` (i.e., sub-byte formats like E2M1) takes
   **16 cycles** instead of the default 8. The scheduler's interleaving ratio
   must account for this: 4 such MFMAs are needed to cover 1 `buffer_load`
   latency (vs 2 for 8-cycle MFMAs).

3. **Cluster must start with MFMA.** The current requirement that a cluster
   (region) must begin with MFMA instructions is acceptable as a limitation.
   However, this should be explicitly documented and clarified in the code,
   since the `SeenMemoryOps` flag behavior is subtle and the current
   implementation silently produces wrong region splits when memory ops
   precede the first MFMA.

---

# MXFP4 GEMM Kernel Design Notes

## Scale loading strategy

Currently we use `buffer_load_dwordx2` to load scales into registers, then
`ds_write` to store them into LDS, then `ds_read` to load them back in the
layout needed by `mfma_scaled`. This causes the vmcnt issue described above
because `buffer_load` (to registers) and `buffer_load_to_lds` (async tile
copies) share the same hardware vmcnt counter.

After implementing **loop unroll** (doubling BLOCK_K from 256 to 512), the
scale tile per iteration grows from `[M, 8]` to `[M, 16]` uint8 — enough to
use `buffer_load_dwordx4` (16 bytes per thread). At that size, we can switch
to `buffer_load_to_lds` for scales as well, routing them directly to LDS
instead of going through registers. This eliminates the mixed
`buffer_load` / `buffer_load_to_lds` problem entirely, since all global loads
would use the same async copy path.
