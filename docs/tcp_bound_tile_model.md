# Compute-Bound, Bandwidth-Bound or Latency-Bound? A Per-Tile Test

The [Memory Bandwidth Model](memory_bandwidth_model.md) explains achieved bandwidth with Little's law: bytes in flight per CU divided by the memory round-trip time, capped by the 32 KB TCP. The [v8 README, §4](../kernels/gemm/intra_wave/a16w16/v8_sliceMN/README.md#4-buffer-load-throughput-and-tcp-limitations) traces what that cap does to a GEMM hot loop: once the TCP and the VMEM request queue are full, the next `buffer_load` cannot issue until the oldest one has retired, and whether that stalls depends on the HBM round trip.

This page turns the two into a one-line test you can run on a tile size before writing the kernel: **given (BM, BN, BK) and the element type, can the hot loop ever be compute-bound, or is it bound by how much the CU can have in flight?** It then separates the two ways a loop waits on memory, which look alike in a trace and need opposite fixes.

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

**A full TCP is a saturated memory system seen from the CU.** Summed over the 256 CUs of an MI355X, 44 KB per 1000 cycles is 11.5 KB per cycle, about 27 TB/s at 2.4 GHz, more than three times the 8 TB/s HBM delivers. The TCPs cannot all be full at a 1000-cycle round trip unless L2 is serving most of the requests; when it is not, the round trip stretches until the bytes in flight divided by it equal what the memory system can deliver. So **TCP-bound and memory-bandwidth-bound are the same condition**: every CU is issuing as fast as the hardware allows, and the memory system cannot absorb more. The cap on the CU side is `C`; the latency `L` it sees is set by the load on the other side.

`L` is therefore not a constant. The v8 README's trace at K = 8192 puts the round trip at **about 1000 cycles**, with the working set largely in L2; at large K the L2 miss rate rises and the round trip grows past that (v8 README, §4.5). Use 1000 cycles for a first pass and remember it is the optimistic end.

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
T x C / B  <   L     -->  bandwidth-bound: issue stalls; the K tile takes B x L / C cycles,
                          not T, and the loop's MFMA efficiency is at most T x C / (B x L)
```

The left-hand side reads as "**MFMA cycles per C bytes of loads**". If the CU spends more than one round trip of matrix-core time per 44 KB it loads, the loads retire faster than new ones are issued and the loop is compute-bound. If it spends less, the wave is waiting on issue slots and the loop is paced by `C / L`.

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
| 128 x 128 | 128 | 32 | 512 | 32 KB | 704 | 768 | bandwidth-bound | 70% | 69% |
| 256 x 64 | 128 | 32 | 512 | 40 KB | 563 | 614 | bandwidth-bound | 56% | 55% |
| 64 x 256 | 128 | 32 | 512 | 40 KB | 563 | 614 | bandwidth-bound | 56% | 57% |
| 128 x 64 | 64 | 16 | 256 | 24 KB | 469 | 512 | bandwidth-bound | 47% | 47% |
| 64 x 128 | 64 | 16 | 256 | 24 KB | 469 | 512 | bandwidth-bound | 47% | 47% |

Three things to read off the table:

- **Below 256 x 128 the loop is bandwidth-bound at 256 workgroups**, and the predicted ceiling matches the measured in-loop MFMA efficiency to within a few points on every such tile. No schedule reaches higher there: a scheduler can spread the loads, it cannot shrink `B / T`.
- **256 x 128 and 128 x 256 sit on the boundary.** With `C` = 44 KB they are just short, with 48 KB just over, and at this tile the loop schedule decides: the measured 83–84% is the sum-model LLIR scheduler, and schedules that pack the loads more tightly move it by several points.
- **256 x 256 is compute-bound with margin** (1408 cycles against 1000), which is why it tolerates the round-trip growth at large K better than the smaller tiles do, and why the v7 → v8 M-slicing in the [v8 README](../kernels/gemm/intra_wave/a16w16/v8_sliceMN/README.md) matters only once `L` has grown past the margin.

## 4. Who is waiting on vmcnt: bandwidth-bound or latency-bound

When a loop is not compute-bound, some instruction is waiting for a load's round trip. Two different instructions can be the one waiting, and they mean different things:

| The instruction that waits | Why it waits | The loop is | Trace signature |
|---|---|---|---|
| the next `buffer_load` itself | the TCP and the queue are full; it needs the oldest load to retire | **bandwidth-bound**: the CU is at its in-flight cap `C`, the memory system cannot absorb more | the `buffer_load` instruction stretches (the yellow rectangles in the v8 README, §4.5); the `s_waitcnt` in front of the LDS reads is short |
| the consumer: `s_waitcnt vmcnt(N)` in front of the `ds_read`s (or the barrier before them) | the `buffer_load_to_lds` it depends on has not landed | **latency-bound**: the loads are issued at a rate the memory system could serve, but each one is issued too late, so the round trip is exposed | long `s_waitcnt` before the reads; the `buffer_load`s issue promptly |

Both are "waiting on vmcnt", and a stall-count summary does not separate them; which instruction carries the wait does. The fixes are opposite, and applying one to the other's symptom does nothing.

## 5. Bandwidth-bound: spread the loads, then accept the cap

Two checks, in order:

1. **Is it a burst?** The TCP and the queue absorb 44 KB. A kernel whose average rate passes the test can still stall if it issues more than that within one round trip and then sits on the MFMAs: v7 at large K issues 16 loads per wave across two adjacent regions, 64 KB per CU inside ~1000 cycles, and stalls; v8 spreads the same 16 loads over four regions and does not (v8 README, §4.5–4.6). Spreading evenly, with the LLIR scheduler or the kernel's slicing, is the first and only in-loop remedy.
2. **Spread evenly and still stalling?** Then the average rate itself exceeds `C / L`: the test of §2 fails and the loop is bandwidth-bound. Nothing in the loop helps: a scheduler cannot shrink bytes per MFMA cycle. The levers are the tile shape (the term `BM x BN / (BM + BN)`, so growing the smaller side helps most), the element type (fewer bytes per MFMA cycle), and the round trip `L` itself (L2 locality: the XCD-aware workgroup remap of v9 acts on `L` through the L2 hit rate). Register budget and LDS capacity decide which of those shapes you can afford; see [v7_sliceN](../kernels/gemm/intra_wave/a16w16/v7_sliceN/README.md) for the register side.

More LDS buffers do not help this case. They address the other one.

## 6. Latency-bound: deepen the pipeline

A load has to *arrive* before its consumer waits on it. With `num_stages` LDS buffers, a K tile's loads are issued `num_stages - 1` tiles before they are read, so the consumer does not stall only if

```
(num_stages - 1) x T  >=  L
```

This is the `effective_pipeline_depth` term of the [Memory Bandwidth Model, §3.1](memory_bandwidth_model.md#31-total-execution-cycles), and it is what LDS capacity buys. With 160 KB of LDS and unpadded tiles:

| Tile | A + B | Buffers that fit in 160 KB | `(num_stages - 1) x T` with 2 buffers | with all that fit |
|---|---:|---:|---:|---:|
| 256 x 256 | 64 KB | 2 | 2048 | 2048 |
| 256 x 128, 128 x 256 | 48 KB | 3 | 1024 | 2048 |
| 128 x 128 | 32 KB | 5 | 512 | 2048 |
| 256 x 64, 64 x 256 | 40 KB | 4 | 512 | 1536 |
| 128 x 64, 64 x 128 | 24 KB | 6 | 256 | 1280 |

(Padding for bank-conflict-free LDS layouts takes a few KB per buffer; the v9 kernel's two padded buffers use 135040 bytes at 256 x 256.)

Deepening the pipeline does not make a loop compute-bound by itself; it **moves the bound**. With more loads in flight per wave the consumer stops waiting, the TCP fills instead, and the loop lands wherever the §2 test puts it: compute-bound if the test passes, bandwidth-bound if it does not. A 128 x 128 tile with 5 buffers passes the prefetch test (2048 >= 1000) and then meets the cap (704 < 1000) at the same 70% ceiling. A 256 x 128 tile with 2 buffers passes both only just (1024 against 1000 on each side), and a third buffer is the cheap way to take the latency side out of the picture while the schedule works on the bandwidth side.

## 7. Checklist

For a new tile or data type:

1. `B` = bytes of A + B per K tile per workgroup; `T` = 16 x MFMAs per wave per K tile.
2. `T x 44 KB / B` against the round trip you expect (1000 cycles at L2-friendly K, more beyond). Under: bandwidth-bound once the loads are spread, and the ratio over `L` is the in-loop MFMA efficiency to expect.
3. `(num_stages - 1) x T` against the same `L` for the buffer count the LDS allows. Under: latency-bound; deepen the pipeline, which moves the loop to step 2's verdict.
4. If the loop still waits, read the trace for *which* instruction waits: a stretched `buffer_load` is the cap of step 2, a long `s_waitcnt` before the reads is the depth of step 3.
5. Only when both pass is the loop's remaining gap a scheduling problem, which is where the LLIR scheduler and the v7 → v9 steps apply.
