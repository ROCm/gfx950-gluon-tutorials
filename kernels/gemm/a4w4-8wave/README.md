# a4w4-8wave — 8-wave warp-pipeline MXFP4 (a4w4) GEMM (gfx950)

An **8-wave** (8 warps/CTA → **2 waves/SIMD**) MXFP4/e2m1 GEMM for gfx950 / MI350X /
MI355X. This is the 4-bit sibling of [`a16w16-8wave`](../a16w16-8wave/README.md): it takes
that kernel's `v1_sliceMN` warp-pipeline solution and swaps the 16-bit numerics for
packed FP4 **with the MXFP4 scale pipeline** from the 4-wave [`a4w4`](../a4w4/README.md)
kernel. Read the [`a16w16-8wave`](../a16w16-8wave/README.md) and
[`a4w4`](../a4w4/README.md) READMEs first — this document covers the 4-bit deltas and the
measured performance.

> [!IMPORTANT]
> Like the fp16 8-wave kernels, this schedules the hot loop at the **wave level** with
> `warp_pipeline_stage` and runs with **no AGPRs** (`amdgpu-agpr-alloc=0,0` via
> `llvm_fn_attrs`). The 4-wave `llir+amdgcnas` toolchain is **not** used here.

## 1. What changes from 16-bit to 4-bit

The 8-wave skeleton — 8 warps `[2,4]`, the 2×2 `[128×128]` quadrant slicing with four
separate double-buffered LDS allocations, the `warp_pipeline_stage` ping-pong schedule,
and no-AGPR — is the same as `a16w16-8wave/v1`. The MXFP4 numerics come from the 4-wave
`a4w4/v1_sliceMN`:

| | `a16w16-8wave/v1` (16-bit) | **`a4w4-8wave/v1` (4-bit)** |
|---|---|---|
| Operand dtype | fp16 / bf16 (2 B) | **packed FP4 / e2m1 (uint8, 2 nibbles/byte)** |
| `BLOCK_K` | 64 | **256** (K//2 = 128 B / row into LDS) |
| Per-block scale | — | **e8m0, one per 32 elements (`SCALE_GROUP_SIZE=32`)** |
| MFMA | `mfma` `[16,16,32]` | **`mfma_scaled(…,"e2m1",…,"e2m1")` `[16,16,128]`**, `k_width=16` |
| B LDS load | plain | **`.permute([1,0])`** (B stored `(N,K//2)`) |
| Output C | fp16 / bf16 | **bf16** |

On gfx950 the MFMA lowers to `v_mfma_scale_f32_16x16x128_f8f6f4 … cbsz:4 blgp:4` (the FP4
format select), which runs at **~2× the BF8 rate** — so the a4w4 peak is roughly double
a8w8's.

### The scale pipeline (from a4w4/v1)

Every tile carries a `[128, 8]` uint8 e8m0 scale half-tile. Following `a4w4/v1_sliceMN`,
each scale streams **straight into LDS** (`buffer_load_to_shared`, no `ds_write`) in the
**same commit group** as its data tile, and is read back with the MFMA scale layout just
before its DOT. Two 8-wave-specific wrinkles:

- **Scale global-load layout must stay dword-granular.** The `buffer_load_to_shared` LDS
  DMA needs 4 bytes/thread. A `[128,8]`=1024-byte half-tile is only 256 dword-threads, but
  an 8-warp CTA spans 512 threads — so `warpsPerCTA=[2,4]` **over-covers M by 2×** (256
  threads issue the dword loads, 256 are masked). A layout that splits it to 2 bytes/thread
  fails to lower (`buffer_load_dword … lds` requires dword). Only `warpsPerCTA` changes
  from the 4-wave `[1,4]` → `[2,4]`.
- **Load-side pointer-walk (see the [v1 README](v1_sliceMN_BK256_nS2/README.md)) is
  required here**, not just nice-to-have — it is what keeps the hot loop spill-free under
  the halved 8-wave VGPR budget.

## 2. Performance

MI355X, gfx950, 4096×4096, MXFP4, **no-AGPR**, rocprof cold-rotating tensors (last-100
average of 1000 dispatches; `--rotating-buffer-size 2048` for K ≥ 16384). The 4-wave
reference is `kernels/gemm/a4w4` measured on the same machine:

Current build (Triton 3.8.0):

| K | **8-wave v1 TFLOPS** | 8-wave MFMA eff (per-SIMD, loop) | 4-wave base | 4-wave llir+amdgcnas |
|---|---|---|---|---|
| 8192  | 3525 | ~57% | 4101 | — |
| 16384 | 4031 | ~57% | — | — |
| 32768 | **4064** | ~57% | — | 5556¹ |

¹ The 4-wave `a4w4` reaches ~5556 TFLOPS with `llir+amdgcnas` at K=32768; that toolchain
targets the 4-wave register model and cannot be applied at 8 waves.

**The 8-wave a4w4 matches the 4-wave *base* at large K** (4064 vs 4101) — but, unlike
a16w16/a8w8 where the 8-wave at least beats the 4-wave *base*, MXFP4 does **not** beat even
that here, and comes nowhere near the tuned 4-wave. The reason is structural: the MXFP4 hot
loop is **throughput-bound on LDS/scale traffic, not latency-bound**, so the 8-wave's
advantage (2 waves/SIMD hiding memory latency by ping-pong) buys little, while its cost — a
**256-VGPR per-wave budget vs the 4-wave's 512** — is real. The load-side pointer-walk keeps
the loop spill-free at **~57% loop MFMA** (the 4-wave base measures ~61% at similar TFLOPS;
the 8-wave's per-SIMD figure runs a touch lower), but there is no latency headroom left to
convert into a win.

> [!NOTE]
> On the current build (ROCm 7.2.4) the ATT decoder **does** decode the FP4 scaled-MFMA
> disassembly, so `collect_perf.py` now reports a real per-SIMD loop MFMA eff (~57%) — earlier
> decoder versions errored on it and the column read `N/A`. As a cross-check, TFLOPS ÷ FP4
> peak (≈6.75 PFLOP/s, from the 4-wave base's ~61% at 4101 TFLOPS) puts the loop near ~60%.

## 3. Running

```bash
# correctness (vs dequantized torch reference) + do_bench TFLOPS
python bench.py --version 1 --K 8192

# rocprof cold-rotating TFLOPS + VGPR/spill  (MFMA-eff via ATT is N/A, see note)
python collect_perf.py --version 1 --K 8192
python collect_perf.py --version 1 --K 32768 --rotating-buffer-size 2048
```

Inputs are packed MXFP4 (uint8) with e8m0 scales; the output is bf16. Clear
`~/.triton/cache` (or use `TRITON_ALWAYS_COMPILE=1`) after editing a kernel.

## 4. Files

- `common.py` — shared `get_pids`, copied from `a16w16-8wave`.
- `bench.py` — correctness (vs dequantized `torch.mm`) + do_bench TFLOPS + `--rocprof`
  rotating-tensor mode, with the MXFP4 input generation from the 4-wave `a4w4/bench.py`.
- `collect_perf.py` — rocprof kernel-trace TFLOPS (cold/rotating) + VGPR/spill.
- `v1_sliceMN_BK256_nS2/` — the kernel + its README. Exposes `matmul_kernel_only`,
  `matmul`, `MIN_K`, `KERNEL_NAME`.
