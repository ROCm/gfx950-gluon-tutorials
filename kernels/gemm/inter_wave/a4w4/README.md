# inter_wave/a4w4 — 8-wave warp-pipeline MXFP4 (a4w4) GEMM (gfx950)

An **8-wave** (8 warps/CTA → **2 waves/SIMD**) MXFP4/e2m1 GEMM for gfx950 / MI350X /
MI355X. This is the 4-bit sibling of [`inter_wave/a16w16`](../a16w16/README.md): it takes
that kernel's `v1_sliceMN` warp-pipeline solution and swaps the 16-bit numerics for
packed FP4 **with the MXFP4 scale pipeline** from the 4-wave [`a4w4`](../a4w4/README.md)
kernel. Read the [`inter_wave/a16w16`](../a16w16/README.md) and
[`a4w4`](../a4w4/README.md) READMEs first — this document covers the 4-bit deltas, the two
kernel versions, and the measured performance.

## 1. Versions

| ver | dir | B-scale handling | K=8192 | K=32768 | loop MFMA eff |
|---|---|---|---|---|---|
| **v0** | [`v0_sliceMN`](v0_sliceMN/README.md) | N-sliced `[128,8]` halves → `ds_read_u8` + `v_perm` | 3659 | 4383 | ~67% |
| **v1** | [`v1_combineBsc`](v1_combineBsc/README.md) | combined `[256,8]` → `ds_read_b64_tr_b8` | 4209 | 5051 | ~81% |
| **v2** | [`v2_mfma32x32x64`](v2_mfma32x32x64/README.md) | v1's combined `[256,8]`; **32×32×64 MFMA** | **4436** | **5136** | **~99%** |

> [!IMPORTANT]
> **v2 is the one to use.** On `gfx950-tutorial-v2.3` it leads v1 at every K: +5.4% at K=8192,
> +3.8% at K=16384 (4923 vs 4741) and +1.7% at K=32768. It runs its loop at ~99% MFMA efficiency
> against v1's ~81%, but its denser MFMA stream draws more power, so its margin depends on the clock
> the die sustains: on a slower MI355X (measured on the v2.2 pin), v1 was ahead at K ≥ 16384.
> `bench.py` defaults to v2 (`--version 2`).

**v2** ([`v2_mfma32x32x64`](v2_mfma32x32x64/README.md)) widens the MFMA **16×16×128 → 32×32×64**
with a width-matched, bank-conflict-free LDS layout. That lifts loop MFMA efficiency to **~99%**
and keeps it **spill-free at 244 VGPRs**, where v0 and v1 spill 26 and 12.

**v1** eliminates v0's B-scale `v_perm` and is worth **+15%** over v0 at every K. It is the step v2 builds on.
**v0** is the pedagogical baseline that exposes the problem.

## 2. What changes from 16-bit to 4-bit

The 8-wave skeleton — 8 warps `[2,4]`, the 2×2 `[128×128]` quadrant slicing with four
separate double-buffered LDS allocations, the `warp_pipeline_stage` ping-pong schedule,
and no-AGPR — is the same as `inter_wave/a16w16`. The MXFP4 numerics come from the 4-wave
`a4w4/v1_sliceMN`:

| | `inter_wave/a16w16` (16-bit) | **`inter_wave/a4w4` (4-bit)** |
|---|---|---|
| Operand dtype | fp16 / bf16 (2 B) | **packed FP4 / e2m1 (uint8, 2 nibbles/byte)** |
| `BLOCK_K` | 64 | **256** (K//2 = 128 B / row into LDS) |
| Per-block scale | — | **e8m0, one per 32 elements (`SCALE_GROUP_SIZE=32`)** |
| MFMA | `mfma` `[16,16,32]` | **`mfma_scaled(…,"e2m1",…,"e2m1")` `[16,16,128]`**, `k_width=16` |
| Output C | fp16 / bf16 | **bf16** |

On gfx950 the MFMA lowers to `v_mfma_scale_f32_16x16x128_f8f6f4 … cbsz:4 blgp:4` (the FP4
format select), which runs at **~2× the BF8 rate** — so the a4w4 peak is roughly double
a8w8's.

## 3. Running

```bash
# v2 (recommended, bench.py's default): correctness + do_bench TFLOPS (from this kernel dir)
python bench.py --version 2 --K 8192

# rocprof cold-rotating TFLOPS + ATT MFMA efficiency + VGPR/spill (from the repo root)
python scripts/collect_perf.py --kernel a4w4 --version 2 --K 8192
python scripts/collect_perf.py --kernel a4w4 --version 2 --K 32768 --rotating-buffer-size 2048

# v0 baseline (the byte-shuffle B scale) for comparison
python bench.py --version 0 --K 8192
python scripts/collect_perf.py --kernel a4w4 --version 0 --K 8192

# v1 (combined B scale, the step v2 builds on)
python bench.py --version 1 --K 8192
python scripts/collect_perf.py --kernel a4w4 --version 1 --K 32768 --rotating-buffer-size 2048
```

Inputs are packed MXFP4 (uint8) with e8m0 scales; the output is bf16. Clear
`~/.triton/cache` (or use `TRITON_ALWAYS_COMPILE=1`) after editing a kernel.

## 4. Files

- `get_pids` is imported from the shared [`kernels/gemm/utils/common.py`](../../utils/common.py).
- `bench.py` — correctness (vs dequantized `torch.mm`) + do_bench TFLOPS + `--rocprof`
  rotating-tensor mode, with the MXFP4 input generation from the 4-wave `a4w4/bench.py`.
- Perf is collected with the shared [`scripts/collect_perf.py`](../../../../scripts/collect_perf.py)
  (`--kernel a4w4 --version N`).
- `v0_sliceMN/` — baseline kernel (N-sliced B scale) + its README.
- `v1_combineBsc/` — combined-B-scale kernel + its README.
- `v2_mfma32x32x64/` — 32×32×64 MFMA + conflict-free LDS layout (recommended, `bench.py`'s default) + its README.
  All three expose `matmul_kernel_only`, `matmul`, `MIN_K`, `KERNEL_NAME`.
