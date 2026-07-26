# inter_wave — 8-wave warp-pipeline a16w16 GEMM for gfx942 / MI300X

Port of the gfx950 tutorial's [`inter_wave/a16w16`](../../inter_wave/a16w16):
8 warps/CTA = **2 waves/SIMD**, the hot loop split into `warp_pipeline_stage`
clusters so the two wave groups run a stage out of phase — while one group is on
the matrix unit, the other issues its loads and absorbs the waits.

```bash
cd ..                                            # kernels/gemm/gfx942
python bench.py -k inter_wave                     # correctness + TFLOPS
python bench.py -k inter_wave --K 4160 --dtype fp16 --show-clock
rocprofv3 --kernel-trace -f csv -d out -- python bench.py -k inter_wave --K 4160 --rocprof
```

No environment variables needed; the no-AGPR setting is baked into the launch.

## Configuration

| | |
|---|---|
| tile M×N×K | 256 × 256 × 64 |
| warps / CTA | 8, `warps_per_cta = [2, 4]` |
| MFMA | `AMDMFMALayout(version=3, instr_shape=[16,16,16], transposed=True)`, `k_width=8` |
| LDS | 4 × 16 KB half-tile slots, single-buffered, 65536 B total |
| dot operands | read as **K=32 slices** of the K=64 LDS tile |
| regions / K-step | 8 (4 quadrants × 2 K-slices) |
| accumulators | VGPR (`amdgpu-agpr-alloc=0,0`) |
| codegen | 224 VGPR, 0 AGPR, 0 spills, **2 waves/SIMD** |

The memory pipeline is inherited unchanged from [`intra_wave`](../intra_wave/):
`buffer_load → local_store → local_load` (no `buffer_load_to_shared` on CDNA3)
and per-half-tile LDS slot recycling (64 KB is exactly one stage).

## What is specific to the 8-wave kernel

### The register wall

Two waves per SIMD split the unified 512-register file, so each wave gets **256
registers, VGPR and AGPR together**. AGPRs buy no capacity at this occupancy —
hence `amdgpu-agpr-alloc=0,0`, the opposite of the 4-wave kernel. Budgeting a
256×256×64 stage against 256:

```
accumulators   4 × [128×128] f32 / (8 × 64)   = 128
dot operands   4 live half-tiles (K=64)        =  96
global staging 4 half-tiles                    =  32
                                                 ---
                                                 256   before addressing
```

Built that way it compiles to 256 VGPRs **with 28 spill slots** and runs at 279
TFLOPS. The dot operands are the only compressible term, so they are read as
**K=32 slices** of the same K=64 LDS tile — `smem.slice(0, 32, k_dim)` and
`slice(32, 32, k_dim)` — halving that term to 48 and landing at 224 VGPR, 0
spills, 2 waves/SIMD, 510 TFLOPS.

Slicing on the *read* side only is deliberate. Dropping `BLOCK_K` to 32 would cut
the same registers, but it also halves the contiguous run of every global load
from 128 B to 64 B, doubling TCP cache-line pressure: that variant measured 386
TFLOPS, and ablating its `buffer_load`s moved it to 503, i.e. the loads were
costing 30%. Keeping `BLOCK_K=64` for the global side and slicing only the LDS
read gets both.

### The region schedule

Slicing K doubles the region count to 8, which is also what the warp pipeline
wants: a K-step is 256 MFMA/SIMD ≈ 4096 cycles, so 8 regions put a cluster
boundary every ~512 cycles — the same barrier cadence as the gfx950 8-wave
kernel, where 4 regions of a 2×-faster MFMA give the same interval. A 4-region
version of this kernel (one boundary per ~1024 cycles) and a `BLOCK_K=32`
version (one per ~256 cycles) both measured slower.

```
region   DOT                      LR (next region's operand)   LW / GR
--------------------------------------------------------------------
R0   C_tl += A_t[k0] B_l[k0]      A_b[k0]                      -
R1   C_bl += A_b[k0] B_l[k0]      B_r[k0]                      -
R2   C_tr += A_t[k0] B_r[k0]      A_t[k1]                      -
R3   C_br += A_b[k0] B_r[k0]      B_l[k1]                      A_top
R4   C_tl += A_t[k1] B_l[k1]      A_b[k1]                      B_left
R5   C_bl += A_b[k1] B_l[k1]      B_r[k1]                      A_bot
R6   C_tr += A_t[k1] B_r[k1]      B_l[k0] of tile k+1          B_right
R7   C_br += A_b[k1] B_r[k1]      A_t[k0] of tile k+1          -
```

Each operand register is overwritten exactly one region after its last use, so
four live slices suffice — no doubling. Each LDS slot is refilled one region
after its last read (`A_top` read last in R2, refilled in R3; `B_left` R3/R4;
`A_bot` R4/R5; `B_right` R5/R6), so one barrier per region boundary is both
sufficient and correct, and the producer→consumer window spans well over the two
stages that [`docs/warp_pipelining.md` §7](../../../../docs/warp_pipelining.md)
requires when the groups are a stage apart.

The `"mem"` cluster carries the higher `s_setprio` (1 vs 0), as the tutorial
prescribes, so its address VALU keeps issuing while the other group is on the
matrix unit. The generated assembly confirms the schedule: the pre-loop
`s_and_saveexec_b64` / `s_cbranch_execz` / `s_barrier` is the `cond_barrier`
phase shift, and every cluster boundary carries an `s_setprio` + `s_barrier`.

## Performance

MI300X, `do_bench` median of 3. Shapes are the gfx942-native ones: N=4864 gives
a 16×19 = 304-workgroup grid (one per CU, no tail), and K is an odd multiple of
64 so A's row stride is not a power of two. See the [gfx942 README §3.1](../README.md)
for why that is worth ~20-30%, and §3.3 on power throttling.

**fp16**

| M | N | K | inter_wave | 4-wave | torch | vs torch |
|---|---|---|---|---|---|---|
| 4096 | 4864 | 2112 | 572 | 534 | 579 | 0.99 |
| 4096 | 4864 | 4160 | **622** | 567 | 620 | 1.00 |
| 4096 | 4864 | 8256 | **575** | 561 | 572 | 1.01 |
| 4096 | 4864 | 16448 | **611** | 575 | 586 | 1.04 |

**bf16**

| M | N | K | inter_wave | 4-wave | torch | vs torch |
|---|---|---|---|---|---|---|
| 4096 | 4864 | 2112 | **609** | 565 | 595 | 1.02 |
| 4096 | 4864 | 4160 | **651** | 586 | 647 | 1.01 |
| 4096 | 4864 | 8256 | **609** | 567 | 602 | 1.01 |
| 4096 | 4864 | 16448 | **659** | 600 | 623 | 1.06 |

TFLOPS. Consistently 7-10% ahead of the 4-wave kernel and at or above hipBLASLt
on every shape, which is what warp-pipelining is supposed to buy on gfx942: the
4-wave kernel loses 41% to LDS barriers it cannot hide, and here the co-resident
wave group hides them.

**These wall-clock numbers understate the gap.** ATT traces (see the
[gfx942 README §3.4-3.5](../README.md)) put per-SIMD MFMA
efficiency at **86.5% here vs 69.7%** for the 4-wave kernel, and the traced
dispatch at 715k cycles vs 856k -- 16.5% fewer cycles, 17.8% less wall time at
matched clock. Most of that advantage is eaten back in a benchmark loop because
the denser MFMA stream draws more power and the 750 W cap clocks the part down;
at K=8256 the two kernels end up only ~2% apart in `do_bench`.

`GROUP_SIZE_M` was swept over 1-16 with `bench.py --sweep-gm`:
GM=1 and GM=2 cost 1-2%, and 4 through 8 are indistinguishable (spreads overlap).
Left at 4.
