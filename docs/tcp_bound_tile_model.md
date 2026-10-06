# Compute-Bound or TCP-Bound? A Per-Tile Test

The [Memory Bandwidth Model](memory_bandwidth_model.md) explains achieved bandwidth with Little's law: bytes in flight per CU divided by the memory round-trip time, capped by the 32 KB TCP. The [v8 README, §4](../kernels/gemm/intra_wave/a16w16/v8_sliceMN/README.md#4-buffer-load-throughput-and-tcp-limitations) traces what that cap does to a GEMM hot loop: once the TCP and the VMEM request queue are full, the next `buffer_load` cannot issue until the oldest one has retired, and whether that stalls depends on the HBM round trip.

This page turns the two into a one-line test you can run on a tile size before writing the kernel: **given (BM, BN, BK) and the element type, can the hot loop ever be compute-bound, or is it bound by how much the CU can have in flight?** The test needs nothing but the tile shape and two hardware numbers.

## 1. The per-CU budget

Three facts from the v8 README, §4.1–4.4:

- Every `buffer_load` and `buffer_load_to_lds` passes through the CU's **TCP (L1), 32 KB**. A `dwordx4` instruction moves 64 lanes x 16 bytes = **1 KB**, so 32 instructions can be in flight per CU.
- Behind the TCP sits a **VMEM request queue of about 12 entries** per CU. Requests parked there have been issued by the wave but have not started their round trip.
- The CU runs **4 waves**, one per SIMD, and they share both.

When the TCP and the queue are full, each wave's next `buffer_load` waits until the oldest in-flight load has come back and retired. So over one round trip of length `L` cycles, the CU cannot push more than

```
C = 32 KB (TCP) + 12 KB (queue) = 44 KB
```

of loads through. (Counting 12 queue entries per wave gives 48 KB; the table below shows both, and the verdicts do not depend on the choice.) By Little's law the CU's bandwidth is capped at `C / L` bytes per cycle, whatever the kernel does.

`L` is not a constant. The v8 README's trace at K = 8192 puts the round trip at **about 1000 cycles**, with the working set largely in L2; at large K the L2 miss rate rises and the round trip grows past that (v8 README, §4.5). Use 1000 cycles for a first pass and remember it is the optimistic end.

## 2. The test

Per K tile, one workgroup loads the A and B tiles:

```
B = (BM + BN) x BK x bytes_per_element        (no padding)
```

and each wave issues

```
n_mfma = BM x BN x BK / (16 x 16 x K_mfma x 4)
```

`16x16xK_mfma` MFMAs of 16 cycles each (4 waves in a 2 x 2 layout). The matrix-core time per K tile is therefore

```
T = 16 x n_mfma   cycles
```

per wave, and since the 4 SIMDs run side by side, `T` is also the CU's timeline for that K tile.

A well-scheduled loop (the LLIR scheduler, or a hand interleave) spreads the K tile's loads evenly over its `T` cycles. The TCP then never fills as long as the kernel asks for at most `C` bytes per `L` cycles:

```
T x C / B  >=  L     -->  compute-bound: the loads never stall the wave
T x C / B  <   L     -->  TCP-bound: issue stalls; the K tile takes B x L / C cycles,
                          not T, and the loop's MFMA efficiency is at most T x C / (B x L)
```

The left-hand side reads as "**MFMA cycles per C bytes of loads**". If the CU spends more than one round trip of matrix-core time per 44 KB it loads, the loads retire faster than new ones are issued and the loop is compute-bound. If it spends less, the wave is waiting on issue slots and the loop is paced by `C / L`, the per-CU bandwidth cap: TCP-bound and memory-bandwidth-bound are the same statement here.

For 16-bit elements and the `16x16x32` MFMA, `BK` cancels and the test is a function of the tile shape alone:

```
T x C / B  =  C x BM x BN / (4096 x (BM + BN))  =  11 x BM x BN / (BM + BN)     (C = 44 KB)
                                                   12 x BM x BN / (BM + BN)     (C = 48 KB)
```

An 8-bit element type with the `16x16x32` fp8 MFMA halves `B` at the same `T`, so it doubles the budget of a given tile shape.

> [!NOTE]
> The budget `C` is per CU. The numbers below assume one workgroup per CU (the intra_wave kernels at 256 workgroups on 256 CUs). Two co-resident workgroups share `C`, which halves each one's budget.

## 3. The eight a16w16 tiles

BK = 64, fp16, 4 waves, `L` = 1000 cycles. "Measured" is the in-loop MFMA efficiency of a tile-parametric build of the v9 kernel (same loop structure, M = 16 x BM, N = 16 x BN, K = 8192, 256 workgroups) under the LLIR scheduler, from thread traces as described in [MFMA Efficiency](mfma_efficiency.md); that build is a test variant and is not in the tree.

| Tile BM x BN | MFMA per K tile (workgroup) | per wave | T = MFMA cycles per wave | A + B bytes | T x C / B, C = 44 KB | C = 48 KB | Verdict at L = 1000 | Predicted MFMA eff (44 KB) | Measured |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 256 x 256 | 512 | 128 | 2048 | 64 KB | 1408 | 1536 | compute-bound | 100% | 96% |
| 256 x 128 | 256 | 64 | 1024 | 48 KB | 939 | 1024 | at the boundary | 94% | 83% |
| 128 x 256 | 256 | 64 | 1024 | 48 KB | 939 | 1024 | at the boundary | 94% | 84% |
| 128 x 128 | 128 | 32 | 512 | 32 KB | 704 | 768 | TCP-bound | 70% | 69% |
| 256 x 64 | 128 | 32 | 512 | 40 KB | 563 | 614 | TCP-bound | 56% | 55% |
| 64 x 256 | 128 | 32 | 512 | 40 KB | 563 | 614 | TCP-bound | 56% | 57% |
| 128 x 64 | 64 | 16 | 256 | 24 KB | 469 | 512 | TCP-bound | 47% | 47% |
| 64 x 128 | 64 | 16 | 256 | 24 KB | 469 | 512 | TCP-bound | 47% | 47% |

Three things to read off the table:

- **Below 256 x 128 the loop is TCP-bound at 256 workgroups**, and the predicted ceiling matches the measured in-loop MFMA efficiency to within a few points on every such tile. No schedule reaches higher there: a scheduler can spread the loads, it cannot shrink `B / T`.
- **256 x 128 and 128 x 256 sit on the boundary.** With `C` = 44 KB they are just short, with 48 KB just over, and at this tile the loop schedule decides: the measured 83–84% is the sum-model LLIR scheduler, and schedules that pack the loads more tightly move it by several points.
- **256 x 256 is compute-bound with margin** (1408 cycles against 1000), which is why it tolerates the round-trip growth at large K better than the smaller tiles do, and why the v7 → v8 M-slicing in the [v8 README](../kernels/gemm/intra_wave/a16w16/v8_sliceMN/README.md) matters only once `L` has grown past the margin.

## 4. What the verdict changes

**A TCP-bound tile cannot be rescued inside the loop.** The cap is bytes per round trip per CU. The levers that move `T x C / B` are the tile shape (the term `BM x BN / (BM + BN)`, so growing the smaller side helps most), the element type (fewer bytes per MFMA cycle), and the round trip `L` itself (L2 locality: the XCD-aware workgroup remap of v9 acts on `L` through the L2 hit rate). Register budget and LDS capacity decide which of those shapes you can afford; see [v7_sliceN](../kernels/gemm/intra_wave/a16w16/v7_sliceN/README.md) for the register side.

**More LDS buffers do not help this condition.** They address a different one, below.

## 5. The second condition: prefetch depth

Independent of the issue cap, a load has to *arrive* before its consumer waits on it. With `num_stages` LDS buffers, a K tile's loads are issued `num_stages - 1` tiles before they are read, so the wave does not stall on `s_waitcnt vmcnt` only if

```
(num_stages - 1) x T  >=  L
```

This is the `effective_pipeline_depth` term of the [Memory Bandwidth Model, §3.1](memory_bandwidth_model.md#31-total-execution-cycles). It is what LDS capacity buys. With 160 KB of LDS and unpadded tiles:

| Tile | A + B | Buffers that fit in 160 KB | `(num_stages - 1) x T` with 2 buffers | with all that fit |
|---|---:|---:|---:|---:|
| 256 x 256 | 64 KB | 2 | 2048 | 2048 |
| 256 x 128, 128 x 256 | 48 KB | 3 | 1024 | 2048 |
| 128 x 128 | 32 KB | 5 | 512 | 2048 |
| 256 x 64, 64 x 256 | 40 KB | 4 | 512 | 1536 |
| 128 x 64, 64 x 128 | 24 KB | 6 | 256 | 1280 |

(Padding for bank-conflict-free LDS layouts takes a few KB per buffer; the v9 kernel's two padded buffers use 135040 bytes at 256 x 256.)

Both conditions have to hold. A 128 x 128 tile with 5 buffers passes the prefetch test (2048 >= 1000) and still fails the TCP test (704 < 1000): deeper buffering alone leaves it at the same ceiling. A 256 x 128 tile with 2 buffers passes both only just (1024 >= 1000), and a third buffer is the cheap way to take the prefetch side out of the picture while the schedule works on the TCP side.

## 6. Checklist

For a new tile or data type:

1. `B` = bytes of A + B per K tile per workgroup; `T` = 16 x MFMAs per wave per K tile.
2. `T x 44 KB / B` against the round trip you expect (1000 cycles at L2-friendly K, more beyond). Under: TCP-bound, and the ratio over `L` is the in-loop MFMA efficiency to expect.
3. `(num_stages - 1) x T` against the same `L` for the buffer count the LDS allows.
4. Only when both pass is the loop's remaining gap a scheduling problem, which is where the LLIR scheduler and the v7 → v9 steps apply.
