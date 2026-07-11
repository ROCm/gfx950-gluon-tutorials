# a4w4-8wave — 8-wave warp-pipeline MXFP4 (a4w4) GEMM (gfx950)

An **8-wave** (8 warps/CTA → **2 waves/SIMD**) MXFP4/e2m1 GEMM for gfx950 / MI350X /
MI355X. This is the 4-bit sibling of [`a16w16-8wave`](../a16w16-8wave/README.md): it takes
that kernel's `v1_sliceMN` warp-pipeline solution and swaps the 16-bit numerics for
packed FP4 **with the MXFP4 scale pipeline** from the 4-wave [`a4w4`](../a4w4/README.md)
kernel. Read the [`a16w16-8wave`](../a16w16-8wave/README.md) and
[`a4w4`](../a4w4/README.md) READMEs first — this document covers the 4-bit deltas, the two
kernel versions, and the measured performance.

> [!IMPORTANT]
> Like the fp16 8-wave kernels, this schedules the hot loop at the **wave level** with
> `warp_pipeline_stage` and runs with **no AGPRs** (`amdgpu-agpr-alloc=0,0` via
> `llvm_fn_attrs`). The 4-wave `llir+force-agpr+amdgcnas` toolchain is **not** used here.

## Versions

| ver | dir | B-scale handling | K=8192 | K=32768 | loop MFMA eff |
|---|---|---|---|---|---|
| **v0** | [`v0_sliceMN_BK256_nS2`](v0_sliceMN_BK256_nS2/README.md) | N-sliced `[128,8]` halves → `ds_read_u8` + `v_perm` | 3525 | 4064 | ~57% |
| **v1** | [`v1_combineBsc_BK256_nS2`](v1_combineBsc_BK256_nS2/README.md) | **combined `[256,8]` → `ds_read_b64_tr_b8`** | **4116** | **4938** | **~80%** |

**v1 is the recommended version** (`--version 1`, the default): +16–22% TFLOPS over v0 at
the same shapes, from eliminating the B-scale `v_perm`. v0 is kept as the pedagogical
baseline that exposes the problem.

## 1. What changes from 16-bit to 4-bit (shared by v0 and v1)

The 8-wave skeleton — 8 warps `[2,4]`, the 2×2 `[128×128]` quadrant slicing with four
separate double-buffered LDS allocations, the `warp_pipeline_stage` ping-pong schedule,
and no-AGPR — is the same as `a16w16-8wave/v1`. The MXFP4 numerics come from the 4-wave
`a4w4/v1_sliceMN`:

| | `a16w16-8wave/v1` (16-bit) | **`a4w4-8wave` (4-bit)** |
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

## 2. The B-scale bottleneck, and how v1 fixes it

Every tile carries an e8m0 scale that streams **straight into LDS** (`buffer_load_to_shared`,
no `ds_write`) and is read back with the MFMA scale layout just before its DOT.

**The A scale is fine; the B scale is the problem.** The MFMA scale operand is delivered by
a `ds_read_b64_tr_b8` hardware-transpose read, which needs **8 bytes/thread**. The scale
operand layout is derived from the dot-operand layout, so it inherits the warp tiling:
the A scale is tiled by `WARPS_M=2` (unchanged from 4-wave), but the B scale is tiled by
`WARPS_N=4`. A `[128,8]` B-scale half therefore gives each thread only **4 bytes** — below
the 64-bit transpose-read width — so it degrades to **per-byte `ds_read_u8` + `v_perm`**
byte-shuffle reassembly. In the real kernel that is **118 `v_perm` + 32 `ds_read_u8`**
purely for the B scale, VALU work that stalls between MFMAs and inflates register pressure.

**v0** ([`v0_sliceMN_BK256_nS2`](v0_sliceMN_BK256_nS2/README.md)) N-slices the B scale into
`b_sc_left` / `b_sc_right` `[128,8]` halves like everything else, and pays this cost.

**v1** ([`v1_combineBsc_BK256_nS2`](v1_combineBsc_BK256_nS2/README.md)) keeps the *tiles*
M/N-sliced but loads the **full `[BLOCK_N, NG] = [256, 8]` B scale as ONE combined buffer**
(both the async fill and the LDS read). At `[2,4]` the un-sliced `[256,8]` gives each thread
**8 bytes = 64 bits**, so the read lowers to `ds_read_b64_tr_b8` with **no `v_perm`**.
Because `get_mfma_scale_layout([256,8])` is exactly the per-quadrant
`get_mfma_scale_layout([128,8])` plus one register base `[128,0]`, a zero-cost `split` +
`convert_layout` recovers the left/right `[128,8]` halves that feed the left/right MFMA
columns — the same operands v0 uses.

### 2.1 Why the combined `[256,8]` fill needs a special blocked layout

`b_scales` is **N-contiguous** in HBM (it is `(K/32, N).T`, strides `(1, N)`). gfx950
direct-to-LDS (`buffer_load … lds`) needs a **32-bit dword** per thread *and* cannot scatter
— each warp's dword writes must land in **one contiguous LDS run**. With v0's
`[4,1],[32,2],[2,4]` blocked layout a warp spans **128 N × 2 K**, and the two K groups are
`tileN` apart in LDS: for `[128,8]` that is `tileN=128` = the warp's N-span, so the two K
blocks pack with **no gap** (coalesced ✅); for the combined `[256,8]`, `tileN=256` > the
128-N span, leaving a **128-byte gap** → not coalesced → the load will not lower
(`canCoalesceWriteIntoSharedMemory` fails).

The fix is the blocked layout `[4,1],[64,1],[1,8]`: each warp = **64 N-lanes × 1 K-lane**
covers **256 N × 1 K** = one contiguous 256-byte K-column, and the 8 warps cover the 8 K
groups. This coalesces for the whole `[256,8]`. (Loading the scale *K*-contiguous instead
would seem natural but is worse: K is the strided dim in HBM, giving `vectorSize=1`, which
gfx950 rejects — the load must stay N-major to get a real dword.)

## 3. Performance

MI355X, gfx950, 4096×4096, MXFP4, **no-AGPR**, rocprof cold-rotating tensors (last-100
average of 1000 dispatches; `--rotating-buffer-size 2048` for K ≥ 16384). Triton
`gfx950-tutorial-v1.0` + the `fence_loads` PR (#10840):

| K | v0 TFLOPS | v0 MFMA eff | **v1 TFLOPS** | **v1 MFMA eff** | v1 speedup |
|---|---|---|---|---|---|
| 8192  | 3525 | ~57% | **4116** | **79.7%** | +16.8% |
| 16384 | 3986 | 57.2% | **4630** | **79.9%** | +16.2% |
| 32768 | 4064 | ~57% | **4938** | **80.0%** | +21.5% |

Codegen (K=8192): B-scale `v_perm` **118 → 0**, `ds_read_u8` **32 → 0**, `ds_read_b64_tr_b8`
**8 → 12**. VGPRs/spills **256 / 23 → 256 / 12** — deleting the `v_perm` temporaries also
halves the (epilogue) spills.

The win comes entirely from the loop: removing 118 register-shuffle `v_perm` lifts per-SIMD
loop MFMA efficiency from **~57% to ~73%**, and the intra-stage `fence_loads` (a `sched.barrier`
after the stage's LDS reads, PR #10840) takes it to **~80%** — together **+16–22%** TFLOPS,
growing with K as the loop dominates. The a4w4 loop is still LDS/scale-throughput-bound (it does not reach the
~90%+ of a16w16/a8w8), but v1 recovers most of the headroom the B-scale byte-shuffle was
wasting. The 4-wave `a4w4` reaches ~5189 TFLOPS with `llir+force-agpr+amdgcnas` (a toolchain that
targets the 4-wave register model and cannot be applied at 8 waves).

## 4. Running

```bash
# v1 (recommended, default): correctness + do_bench TFLOPS
python bench.py --version 1 --K 8192

# rocprof cold-rotating TFLOPS + ATT MFMA efficiency + VGPR/spill
python collect_perf.py --version 1 --K 8192
python collect_perf.py --version 1 --K 32768 --rotating-buffer-size 2048

# v0 baseline (the byte-shuffle B scale) for comparison
python bench.py --version 0 --K 8192
python collect_perf.py --version 0 --K 8192
```

Inputs are packed MXFP4 (uint8) with e8m0 scales; the output is bf16. Clear
`~/.triton/cache` (or use `TRITON_ALWAYS_COMPILE=1`) after editing a kernel.

## 5. Files

- `common.py` — shared `get_pids`, copied from `a16w16-8wave`.
- `bench.py` — correctness (vs dequantized `torch.mm`) + do_bench TFLOPS + `--rocprof`
  rotating-tensor mode, with the MXFP4 input generation from the 4-wave `a4w4/bench.py`.
- `collect_perf.py` — rocprof kernel-trace TFLOPS (cold/rotating) + ATT MFMA efficiency +
  VGPR/spill.
- `v0_sliceMN_BK256_nS2/` — baseline kernel (N-sliced B scale) + its README.
- `v1_combineBsc_BK256_nS2/` — combined-B-scale kernel (recommended) + its README. Both
  expose `matmul_kernel_only`, `matmul`, `MIN_K`, `KERNEL_NAME`.
