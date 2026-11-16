# Benchmarking the tutorial's kernels

The GEMM and attention kernels of this tutorial are **power-bound** on MI355X: under sustained
load the socket sits at its power limit and the power-management firmware sets the GFX clock.
Their speed therefore depends on what the GPU did in the last milliseconds to minutes, and a
timing has to say which state it measures. This page describes the protocol every `bench.py`
uses ([`scripts/graph_timer.py`](../scripts/graph_timer.py)), why, and what the earlier timing
paths measured instead.

## 1. The protocol

1. **Back to back, by graph replay.** `d` dispatches are captured in one CUDA graph and replayed
   with at most two replays queued, so the GPU never idles and no host launch sits between
   dispatches. `d` is chosen so a replay lasts about 20 ms (16 to 4096 dispatches); the CUDA
   event pair around each replay then costs a few microseconds per `d` dispatches.
2. **Cold inputs.** Dispatch `i` reads rotating tensor set `i % sets`, with enough sets that the
   inputs in rotation exceed 512 MiB, twice the 256 MiB last-level cache (MALL), and at least two.
   Every dispatch reads its inputs from HBM, as a kernel in a model would.
3. **Warm-up, then a window.** The replays run for a warm-up time and then for a measurement
   window. The result is the **median over the window's replays of replay time / `d`**, reported
   with the 10th and 90th percentiles, the drift between the window's two halves, and (where the
   driver exposes the firmware metrics table, MI350) the window's average socket power and
   effective GFX clock.

Two protocols are predefined:

| `--protocol` | Warm-up | Window | Use |
|---|---|---|---|
| `quick` (default) | 1 s | 4 s | A/B comparisons; alternate the two sides |
| `sustained` | 60 s | 30 s | Published numbers: the level a kernel holds indefinitely |

`--warmup-s` and `--window-s` override either. The kernel's input data are normally distributed
(`torch.randn`), as in a model; see §3.6 for why that matters.

```bash
cd kernels/gemm/intra_wave/a16w16
python bench.py --version 9 --K 8192 --dtype fp16                      # quick
python bench.py --version 9 --K 8192 --dtype fp16 --protocol sustained # published
```

```
[v9_beyond_hotloop 4096x4096x8192 fp16] 185.93 us/dispatch (1478 TFLOPS), p10-p90 185.17-186.85 us,
drift -0.22%: 247 replays x 88 dispatches over 4 s after 1 s warm-up (quick), 4 rotating sets,
GPU 0000:xx:00.0; 1400 W, 1667 MHz
RESULT {...}
```

The `RESULT` line is the same result as JSON, for scripts; `--result-json FILE` appends it to a
file.

## 2. The states a power-bound kernel goes through

Measured on MI355X (1400 W limit) with a16w16 v9, 4096x4096x8192 fp16:

| After the load starts | What happens | Kernel time |
|---|---|---|
| 0 to ~2 ms | the GPU is still at its idle point (2.4 GHz, 0.88 V) | 1-3% slower than settled |
| ~2 to ~20 ms | the power limiter cuts the clock target within 2 ms, the clock dips to ~1.3 GHz | up to 35-45% slower |
| 20 to 50 ms | recovery | within 1-2% |
| 0.05 to ~1 s | settled: power at the limit | the `quick` level |
| 1 to 60 s | the board warms; the clock gives up ~1% | 1.2-1.4% slower at this shape (none at K=4096) |
| from ~60 s | sustained | the `sustained` level |

Idle time also matters: gaps of a few milliseconds between dispatches return the GPU to its idle
point, and every burst then replays the start of this table.

## 3. What the earlier timing paths measured

Measured on two MI355X GPUs of one node (A and B), a16w16 v9 4096x4096x8192 fp16, `llir` build,
against the back-to-back graph replay settled level (TFLOPS, higher is faster):

| Path (where it was used) | GPU A | GPU B |
|---|---|---|
| Graph replay, back to back, settled (reference) | 1478 | 1491 |
| `bench.py` default: `triton.testing.do_bench` | 1394 | 1359 |
| `do_bench` called a second time in the same process | 1405-1412 | 1444-1449 |
| `triton.testing.do_bench_cudagraph` | 1466-1470 | 1509-1519 |
| `bench.py --rocprof`, `AMD_SERIALIZE_KERNEL=3`, last 100 of 1000 (`run_perf_table.py --rocprof`) | 1451 | 1468 |
| the same through the prepared launcher (`--prepared`) | 1454 | 1501 |
| `bench.py --rocprof`, not serialized, last 100 of 1000 (`collect_perf.py`) | 1480 | 1489 |

### 3.1 `triton.testing.do_bench`

`do_bench(fn, warmup=25, rep=100)` times 5 runs to estimate the kernel time, then converts the
25 ms warm-up and 100 ms measurement into run counts. Three things go wrong for these kernels:

* **The estimate.** On the first call in a process, the 5-run estimate includes a one-time ~8 ms
  host stall (the first launch of its cache-flush kernel happens after the start event), so it is
  6-8x too high: the "25 ms" warm-up becomes ~13 runs (~3 ms) and the measurement ~55-75 runs,
  about 6 to 20 ms after the load starts, inside the start-up dip. A second call in the same process
  gets the estimate right.
* **The flush.** Before every timed run it zero-fills a 256 MiB buffer (~40 us). The kernel that
  follows runs ~3% slower (191.7 vs 185.7 us) although its clock is higher (1849 vs 1661 MHz):
  the fill's memory traffic, not power.
* **The events.** Each timed run is bracketed by its own event pair, which adds ~5 us: 2.7% at
  190 us, **25% at a 20 us kernel** (K=512: 698-714 TFLOPS against 753 back to back with cold
  inputs).

### 3.2 `triton.testing.do_bench_cudagraph`

It captures `rep / estimate` calls (20 ms) in one graph, replays it 10 times and returns the mean
per call. There is no warm-up beyond the 5-call estimate, so the first replay contains the
start-up dip (196-203 us vs 181-188 us), and there is no cache flush or rotation, so inputs that
fit in the MALL stay warm (here 2% fast on GPU B). Close to the right answer for a large kernel,
but neither cold nor controlled.

### 3.3 `AMD_SERIALIZE_KERNEL=3`

It makes the runtime wait for each dispatch before launching the next: a 17-26 us host round trip
before every dispatch. At 190 us the GPU is busy 88-91% of the time and its clock rises to
1.79-1.84 GHz (1.66 GHz back to back), yet the kernel read 1.4-2.1% slower than back to back
through the JIT launcher, and -0.8% to +1.8% through the prepared launcher, depending on the GPU.
A 20 us kernel runs on a GPU that is idle half the time (2.2 GHz, 0.6-0.7 kW instead of
1.2-1.3 kW). A serialized per-dispatch time is the time in a different power state, not the
kernel's back-to-back time.

### 3.4 "The last 100 of 1000 dispatches"

1000 dispatches of 190 us end 0.2 s after the start: past the start-up transient, but before the
board warms (§2). At K=8192 the sustained level is another 1.3-1.4% slower. The window is also
only ~20 ms long.

### 3.5 Caches

With a single input set, inputs that fit in the 256 MiB MALL are read from the cache after the
first dispatch (1.5-2% faster at K=8192 on one GPU, no difference on the other). Rotation fixes it;
the rule above sizes it by the inputs. One 256 MiB set (K=16384 fp16) read the same as seven sets
within 0.5% on both GPUs.

### 3.6 Input data

A power-bound kernel's speed depends on how many bits toggle in its datapath. Back to back, the
same kernel took 186.0 / 184.4 us with `randn` inputs, 175.3 / 178.2 us with `rand` (uniform in
[0, 1): 3-5% faster) and 151.5 / 151.2 us with zeros (23% faster). Benchmark with the data
distribution the kernel will see; never with zeros, ones or ranges.

### 3.7 MFMA efficiency from ATT

`rocprofv3 --att` traces the dispatch named by `kernel_iteration_range` and serializes that one
dispatch (all of them only with `--att-serialize-all`). The perf-table scripts used to trace
dispatch 15: with `bench.py`'s default `do_bench` that is a warm-up run 2.8 ms after the load
starts, in the dip (210-223 us against ~190 us later, at 1.55-1.80 GHz). They also took the
TFLOPS column from the `do_bench` run inside the ATT-profiled process, which read 1283 TFLOPS
against 1394 without the profiler. Now the TFLOPS come from a separate timing run, and the traced
dispatch follows 1 s of back-to-back dispatches (`--att-warmup-s`).

### 3.8 The GPU

Two MI355X GPUs of one node differ by 1% back to back and by up to 3% under `do_bench`; the FA
work found the 8 dies of one board spanning 6.5%. A number means little without its GPU, power
limit, and the power and clock it ran at, which every result now carries.

## 4. Scripts

* `bench.py` (every kernel directory): correctness check, then the protocol above. `--rocprof`
  is the profiler mode: no timing, `--warmup-dispatches` + `--iters` dispatches back to back for
  `rocprofv3 --pmc` or `--att`.
* [`scripts/run_perf_table.py`](../scripts/run_perf_table.py) and
  [`scripts/collect_perf.py`](../scripts/collect_perf.py): TFLOPS from a timing run, MFMA
  efficiency from ATT after a warm-up, VGPRs and spills, per config; `--protocol`, and
  `--rocprof` for a kernel-trace cross-check over the same window (no serialization). Each run
  compiles into a private Triton cache; the user's cache is never deleted.
* [`scripts/fa_kernel_time.py`](../scripts/fa_kernel_time.py): the attention `bench.py` timing at
  the FA shape. [`scripts/fly_kernel_time.py`](../scripts/fly_kernel_time.py) times FlyDSL's
  kernel with the same warm-up and window from the kernel trace (its launcher is not known to be
  graph-capturable).

## 5. The published tables

The TFLOPS tables in the kernel READMEs were measured before this protocol: serialized
`rocprofv3` timing, the last 100 of 1000 dispatches, Triton `gfx950-tutorial-v3.0`, on one
well-performing MI355X. They compare versions within a table correctly, but not with numbers
taken under this protocol or on another GPU: the same serialized protocol on the two GPUs above
gives 1451-1469 TFLOPS for v9 `llir+amdgcnas` at 4096x4096x8192 fp16, where the README quotes 1600.

## 6. ATT on drivers older than KFD ABI 1.20

rocprofiler-sdk 1.4 and later create the thread-trace queue directly through KFD and need the
driver to report the queue's context-save sizes, which KFD does from ABI 1.20 on. On older amdgpu
drivers the queue is rejected:

```
F... resource.cpp:141] AMDKFD_IOC_CREATE_QUEUE failed: Invalid argument
```

The decoder library is not the problem. `export ROCPROFILER_SQTT_FORCE_HSA=1` selects the SDK's
HSA path, which ROCm 7.2 uses; [`scripts/run_att.py`](../scripts/run_att.py) retries with it
automatically.
