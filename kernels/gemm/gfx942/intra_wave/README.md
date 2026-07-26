# intra_wave — 4-wave a16w16 GEMM for gfx942 / MI300X

Port of the gfx950 tutorial's [`intra_wave/a16w16/v9_beyond_hotloop`](../../intra_wave/a16w16/v9_beyond_hotloop):
4 warps/CTA = **1 wave/SIMD**, a 256×256 output tile sliced into a 2×2 grid of
128×128 quadrants, and XCD-aware PID remapping with `GROUP_SIZE_M=4`.

```bash
cd ..                                            # kernels/gemm/gfx942
python bench.py -k intra_wave                     # correctness + TFLOPS
python bench.py -k intra_wave --K 4160 --dtype fp16 --show-clock
rocprofv3 --kernel-trace -f csv -d out -- python bench.py -k intra_wave --K 4160 --rocprof
```

`bench.py` sets `TRITON_FORCE_MFMA_AGPR=1`, which is what supplies the
`amdgpu-agpr-alloc=256` hint. Without it the 256 f32 accumulators compete with
the 192 VGPRs of operands and staging and the kernel spills.

## Configuration

| | |
|---|---|
| tile M×N×K | 256 × 256 × 64 |
| warps / CTA | 4, `warps_per_cta = [2, 2]` |
| MFMA | `AMDMFMALayout(version=3, instr_shape=[16,16,16], transposed=True)`, `k_width=8` |
| LDS | 4 × 16 KB half-tile slots, **single-buffered**, 65536 B total |
| shared layout | `SwizzledSharedLayout(8, 2, 8)` — no padding, LDS is full |
| accumulators | AGPR (`amdgpu-agpr-alloc=256`) |
| codegen | 256 VGPR + 256 AGPR, 0 spills, 1 wave/SIMD |

## The pipeline

CDNA3 has no `buffer_load_to_shared`, so global data takes the register
round-trip `buffer_load → local_store → local_load`, and 64 KB of LDS is exactly
one 256×256×64 stage — double buffering is impossible at this tile size. The
kernel instead recycles LDS at **half-tile granularity**: each of `A_top`,
`A_bot`, `B_left`, `B_right` owns one 16 KB slot, refilled one region after its
last read.

```
region 0:  DOT C_tl | LR A_bot(k)    | LW B_left(k+1)  | GR B_left(k+2)
region 1:  DOT C_bl | LR B_right(k)  | LW A_top(k+1)   | GR A_top(k+2)
region 2:  DOT C_tr | LR B_left(k+1) | LW A_bot(k+1)   | GR A_bot(k+2)
region 3:  DOT C_br | LR A_top(k+1)  | LW B_right(k+1) | GR B_right(k+2)
```

Every slot's write lands strictly between its previous and next read, one region
apart on both sides, so one barrier per region is sufficient. The pipeline depth
matches gfx950's double buffer: `GR(k+2) → LW(k+1)` is a full K-step (~4096
cycles, 256 MFMAs) of HBM latency hiding; `LR(k+1) → DOT(k+1)` is one region
(~1024 cycles) of LDS latency hiding.

Per K-step the generated loop is exactly 256 `v_mfma`, 32 `ds_read_b128`, 16
`ds_write_b128`, 16 `buffer_load_dwordx4`, 3 `s_barrier`, 0 spills, 0
`v_accvgpr` copies.

## Register budget

```
C accumulators   4 × [128×128] f32 / (4 × 64)   = 256   -> AGPR
dot operands     4 × half-tile fp16              = 128   -> VGPR
global staging   4 × half-tile fp16              =  64   -> VGPR
                                                   ----
                                                   448
```

At 1 wave/SIMD the whole 512-register file (256 VGPR + 256 AGPR) belongs to the
wave, which is what makes the 256-register accumulator affordable — and why this
kernel wants AGPRs while the 8-wave one forbids them.

## Performance

MI300X, `do_bench` median of 3. Shapes are the gfx942-native ones: N=4864 gives
a 16×19 = 304-workgroup grid (one per CU, no tail), and K is an odd multiple of
64 so A's row stride is not a power of two. See the [gfx942 README §3.1](../README.md)
for why that is worth ~20-30%, and §3.3 on power throttling.

| M | N | K | fp16 | bf16 | torch fp16 |
|---|---|---|---|---|---|
| 4096 | 4864 | 2112 | 534 | 565 | 579 |
| 4096 | 4864 | 4160 | 567 | 586 | 620 |
| 4096 | 4864 | 8256 | 561 | 567 | 572 |
| 4096 | 4864 | 16448 | 575 | 600 | 586 |

TFLOPS. The 8-wave [`inter_wave`](../inter_wave/) kernel is 7-10% ahead of this
one on the same shapes; the reason is below.

`GROUP_SIZE_M` was swept over 1-16 with `bench.py --sweep-gm`:
1 and 2 cost 2%, 16 costs ~0, and 4 through 8 are indistinguishable. Left at 4.

## Known limit: the barriers

Ablating the hot loop at 4096²×8192 shows neither memory op costs anything alone
— `local_load` and `buffer_load` are free — but the pair `local_load` +
`local_store` costs **+41%**. That is entirely the barriers membar must insert
for the write-after-read/read-after-write on each slot, plus the
`s_waitcnt lgkmcnt(0)` each drags along; the backend clusters the MFMAs and lands
two of the three barriers ~6 MFMAs apart, so an 8-deep `ds_read` burst drains
with almost no compute to hide it.

Regrouping the stores to cut barriers from 3 to 2 made it worse (428 TFLOPS —
the resulting 16-store/16-read clustering costs more than the barrier saved), and
so did splitting into 8 shorter regions (438). The structural fix is the
[`inter_wave`](../inter_wave/) kernel, where a second resident wave group has
MFMA work queued whenever the first is at a barrier.
