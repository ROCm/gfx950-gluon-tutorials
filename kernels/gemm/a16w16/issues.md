# Known issues — a16w16 kernels

## v6 (loop_unroll): non-deterministic data race at narrow tiles

**Status:** open. **Severity:** correctness (wrong results). **Scope:** `v6_loop_unroll` only.

### Summary
At narrow tiles, `v6_loop_unroll` produces **wrong** GEMM results. The error is a
**non-deterministic data race**: with identical inputs it varies run-to-run and
occasionally even passes by luck. `v5_local_prefetch` (same layouts, same buffers)
is correct on every tile, so this is specific to v6's manually **unrolled** K-loop.

### Affected shapes (BLOCK_M × BLOCK_N × 64, fp16, K = 8192)
Fails on the 4 tiles with a narrow N (or small M+narrow N):

| | N=256 | N=128 | N=64 |
|---|---|---|---|
| **M=256** | ok | ok | ❌ |
| **M=128** | ok | ok | ❌ |
| **M=64**  | ok | ❌ | ❌ |

Correct on: 256×256, 256×128, 128×256, 128×128, 64×256. Fails on: 256×64,
128×64, 64×128, 64×64.

### Symptoms
- **Non-deterministic.** Single-workgroup test (`manual_seed(0)`, fresh process), wrong-element %:
  - 64×128: `0% / 33.6% / 31.5% / 0%`  (sometimes correct)
  - 128×64: `15.5% / 0% / 46.5% / 35.2%`
  - 256×64: `7% / 7% / 4.1% / 45.7%`
  - 64×64:  `~92.6%` every run (timing window almost always lost)
- **Structured.** Corruption is confined to **warp-column 1** (N ≥ BLOCK_N/2, i.e. the
  second N-half = the B operand for the second warp column). Warp-column 0 is always correct.
- Wide tiles (e.g. 256×256) are rock-stable (max|diff| ≈ 0.016, fp noise) — the race
  exists but its window never loses there.

### Root cause
The race lives in v6's **×2 unrolled** double-buffered LDS pipeline. The unrolled and the
single-step (rolled) loops are *logically identical* (same buffer parity, same MFMA
pairing/order), but stock Triton's automatic `MembarAnalysis` emits the **same static
`s_barrier` count (12)** for both forms — so the unrolled body, which covers **2 K-steps
per iteration**, is under-synchronized per K-step on the LDS buffer reuse (WAR: an async
copy overwrites a buffer before all waves finish reading it / RAW across waves). At narrow
tiles the overwrite-vs-read window is tight enough to lose, corrupting warp-column 1's B data.

### What it is NOT (ruled out during debug)
- **Not** `load_contig` / global-load layout coverage (v5 uses the identical
  `compute_gload_layout(..., 8, 4)` and is correct on all 9 tiles).
- **Not** the hardcoded `gStoreLayoutC = BlockedLayout([1,8],[2,32],[4,1],[1,0])` —
  swapping it to `mfmaLayout` does not help; the data is already wrong before the store.
- **Not *caused* by** the LLIR scheduler — `base` (`schedule_hint` off) fails on the same
  4 narrow tiles. But the scheduler is *not* neutral to the race — see "Scheduler
  interaction" below.
- **Not** a single missing barrier — adding one explicit `gl.barrier()` only reduces
  the hit-rate, it does not eliminate it.

### Scheduler interaction (2026-07-06)
The root cause is stock-Triton membar under-sync (above), but the LLIR scheduler is not
neutral to it: interleaving MFMA with the LDS loads/stores — and pinning that order with
`sched.barrier` intrinsics — **tightens the WAR/RAW window**, so the race loses more often
and on more shapes. Measured under `schedule_hint="mfma-mem-interleave"` (4096²×8192,
fp16, 3 runs/cell):

| tile | base | llir | +agpr |
|---|---|---|---|
| 256×256 / 256×128 / 128×256 | ok | ok | ok |
| **128×128** | ok | flaky (~1/3 fail) | ✗ |
| **64×256** | ok | ✗ | ok |
| 256×64 / 128×64 / 64×128 / 64×64 | ✗ | ✗ (larger %) | ✗ |

So the scheduler exposes the race on **128×128 and 64×256** — correct under `base` — and
enlarges the corrupted fraction on the already-failing narrow tiles. It is config-dependent
(llir vs `+agpr` flip on both 128×128 and 64×256): `force-agpr` shifts register allocation
and thus the instruction timing, which is further proof this is a race, not miscodegen. The
fix is unchanged (below); the scheduler only changes how easily the bug is hit.

### Reproduce
```bash
cd kernels/gemm/a16w16
TILE_M=64 TILE_N=128 TILE_K=64 python - <<'PY'
import os, torch, importlib
m = importlib.import_module("v6_loop_unroll.matmul_kernel")
torch.manual_seed(0)
BM,BN = int(os.environ["TILE_M"]), int(os.environ["TILE_N"]); K=8192
a = torch.rand((BM,K), device="cuda", dtype=torch.float16)-0.5
b = (torch.rand((BN,K), device="cuda", dtype=torch.float16).T)-0.5
for _ in range(4):
    d = (m.matmul(a,b).float()-torch.matmul(a,b).float()).abs()
    print("wrong%%:", 100*(d>0.1).float().mean().item())   # varies run-to-run
PY
```

### Fix / workaround
Roll the K-loop back to single-step (one K-step per iteration, dynamic buffer index
`g_idx = k % 2`, like `v5_local_prefetch`). Verified **0% wrong across 4 runs on all 4
failing tiles** while leaving the passing tiles unchanged. Alternatively: find the full
WAR+RAW barrier set the unrolled form needs, or file the missing-barrier behavior as a
Triton `MembarAnalysis` bug for the unrolled gluon double-buffer pattern.
