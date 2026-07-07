# [gfx950] v6 loop-unrolled GEMM: non-deterministic wrong results at narrow tiles

## Summary

The `a16w16/v6_loop_unroll` gluon GEMM (manually ×2-unrolled K-loop, double-buffered
LDS with `async_copy.buffer_load_to_shared`) produces **non-deterministic wrong results**
on narrow tiles, in the plain **base** config (no scheduler, no amdgcnas). With identical
inputs the wrong-element fraction varies run-to-run and the corruption is structurally
confined to **warp-column 1** (the second N-half of the B operand). The logically
equivalent single-stepped (rolled) loop — `v5_local_prefetch` — is correct on every tile.

The failure is a **cross-wave data race on the shared LDS double-buffer**. It is **not**
fixed by the recent LLVM bump (PR #10739), and it is **not** a barrier desync — see
"Investigation summary" below.

## Environment

| | |
|---|---|
| GPU | **AMD Instinct MI355X** |
| Arch | **gfx950** (CDNA4), `gfx950:sramecc+:xnack-` |
| ROCm | **7.0.0** |
| PyTorch | 2.8.0+rocm7.0.0 |
| Python | 3.10.12 |
| Triton | `triton-lang/triton` @ `a45bf3e1` (PR #10739 `lei/bump-llvm-0626`, LLVM `850a2b1b`) |
| | also reproduces on `triton-lang/triton` `main` @ `932ceaa2b1` (LLVM `62b7cf96`) |
| Tutorial | `ROCm/gfx950-gluon-tutorials` @ branch **`llirSched_prod`** (`a7b34c57`) |

Reproduces on **both** LLVM `62b7cf96` and `850a2b1b`, so it is **LLVM-version-independent**
(the hot-loop assembly is byte-identical across the two).

## Steps to reproduce

```bash
# 1. Triton (any of the commits above). If building the LLVM-bump PR:
git clone https://github.com/triton-lang/triton.git && cd triton
git fetch origin pull/10739/head:pr-10739 && git checkout pr-10739
pip install "nanobind==2.10.2"          # build dep for this triton version
MAX_JOBS=128 pip install -e . --no-build-isolation

# 2. Tutorial
git clone -b llirSched_prod https://github.com/ROCm/gfx950-gluon-tutorials.git
cd gfx950-gluon-tutorials/kernels/gemm/a16w16

# 3. Reproduce (M=N=4096, K=8192, fp16, base config). Tile is set via env.
#    Non-deterministic — run a few times.
TILE_M=64 TILE_N=64 TILE_K=64 python bench.py --K 8192 --config base --dtype fp16 --version 6
```

### Actual output (fails)

```
[v6_loop_unroll] M=4096 N=4096 K=8192 dtype='fp16': ❌ Triton and Torch differ   # 64x64  (~90% wrong)
[v6_loop_unroll] M=4096 N=4096 K=8192 dtype='fp16': ❌ Triton and Torch differ
[v6_loop_unroll] M=4096 N=4096 K=8192 dtype='fp16': ❌ Triton and Torch differ
```

### Expected

```
[v6_loop_unroll] ... ✅ Triton and Torch match
```

Control (passes): `TILE_M=256 TILE_N=256 TILE_K=64 python bench.py --K 8192 --config base --dtype fp16 --version 6` → ✅.

### Affected shapes (BLOCK_M × BLOCK_N × 64, fp16, K=8192)

| | N=256 | N=128 | N=64 |
|---|---|---|---|
| **M=256** | ✅ | ✅ | ❌ |
| **M=128** | ✅ | ✅ | ❌ |
| **M=64**  | ✅ | ❌ | ❌ |

`64x64` is the most reliable reproducer (~90% wrong nearly every run); `64x128` ≈ 68%.

## Error magnitude

Not fp rounding noise — corrupted operand data. Reference outputs average |ref| ≈ 72.

| metric (64×64 example) | value |
|---|---|
| wrong elements (>0.1 abs) | ~90% at 64×64 (~16% at the 64×256 read-then-fill variant) |
| \|error\| on wrong elems | median ~7, max ~58 |
| relative error on wrong elems | median ~12%, mean ~93% (bimodal: partial + garbage) |
| localization | ~31% of N[128:256) wrong vs ~2% of N[0:128) — i.e. warp-column 1 / B second half |

## Investigation summary (what it is / isn't)

- **Not an LLVM codegen bug.** Rebuilt on PR #10739 (LLVM `850a2b1b`); the hot-loop `.amdgcn`
  is byte-identical to LLVM `62b7cf96` and the failure rate is unchanged.
- **Not a barrier desync.** ATT trace (32 waves) shows every wave executes an identical
  257 `s_barrier`; the `s_waitcnt vmcnt(0)`/`lgkmcnt(0)` waits genuinely stall (~1700 / ~90
  cyc), i.e. the fences are present, matched, and load-bearing.
- **Not the local_load↔async_copy order.** `local_load` and the async copy in one K-step
  target *different* buffers; swapping their order only shifts the tile-size threshold
  (fill-then-read fails at 64×64; read-then-fill also fails at 64×256). Both orders race.
- **Scales with async-copy concurrency.** The only thing that deterministically fixes it is
  reducing the number of concurrently in-flight `buffer_load…lds`: inserting
  `s_waitcnt vmcnt(0)` after **every** fill → 0% wrong; after every **other** fill (5 in
  flight) → ~7%; baseline (10 in flight) → ~20%. Adding delay (`s_nop`) or extra `s_barrier`
  does **nothing**.

**Working hypothesis:** a single `s_waitcnt vmcnt(0)` over a *batch* of concurrent
`buffer_load…lds` (global→LDS DMA) does not deterministically order all of the batch's LDS
writes with respect to a cross-wave `ds_read` after the following `s_barrier` — a property
of the gfx950 async-copy path, not the Triton membar logic. It usually lands in time (hence
mostly correct) but loses under tight cross-wave timing at narrow tiles.

## Workaround

Roll the K-loop back to single-step (one K-step/iteration, dynamic buffer index
`g_idx = k % 2`), as in `v5_local_prefetch` — verified correct on all 9 tiles.

## Attachments / repro artifacts

- ATT trace of the failing 64×256 case: `/data/att_v6_loop_unroll_64x256_K8192_base_RACE/`
  (32 fully-stitched per-wave timelines + `code.json` + `SUMMARY.txt`).
- Full asm-surgery matrix and details: `kernels/gemm/a16w16/issues.md`.
