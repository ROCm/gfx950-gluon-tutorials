# Scripts

Helper scripts for profiling and benchmarking the tutorial's Triton/Gluon kernels.

The first group drives the GEMM kernels. The rest belong to
[`kernels/attention/`](../kernels/attention/README.md), and split into three kinds:

| script | what it does |
|---|---|
| **Measurement** | |
| [`fa_kernel_time.py`](#fa_kernel_timepy) | the reported FA metric — `rocprofv3` kernel time with a prepared launch |
| [`fly_kernel_time.py`](#fly_kernel_timepy) | times ROCm/FlyDSL under *our* protocol, so the two can share a table |
| **Figures** | |
| [`plot_fmha_summary.py`](#plot_fmha_summarypy) | regenerate the attention README's summary chart |

## install_att_decoder.sh

One-shot installer for the [`rocprof-trace-decoder`](https://github.com/ROCm/rocprof-trace-decoder)
library that `rocprofv3 --att` (and therefore `run_att.py`, `run_perf_table.py
--rocprof`) needs at decode time. The decoder is not bundled with ROCm; it must
be fetched separately from the public release page.

### Prerequisites

- `curl` and `tar` in `PATH`
- write access to `/opt/rocm/lib/` (or the `--prefix` you choose)

### Usage

```bash
# default: installs to /opt/rocm/lib/ (no further setup needed)
sudo scripts/install_att_decoder.sh

# alternative prefix; remember to export ROCPROF_ATT_LIBRARY_PATH afterwards
scripts/install_att_decoder.sh --prefix $HOME/.local
export ROCPROF_ATT_LIBRARY_PATH=$HOME/.local/lib/

# pin a different release version
VERSION=0.1.6 scripts/install_att_decoder.sh
```

The default `0.1.6` release is the version this tutorial is tested against.
`run_att.py` looks for `librocprof-trace-decoder.so` under the path in the
`ROCPROF_ATT_LIBRARY_PATH` environment variable; if unset, it falls back to
`/opt/rocm/lib/`, which is where this installer's default prefix lands.

## run_perf_table.py

Automates running benchmarks across kernel versions and scheduler configs, collecting TFLOPS, VGPR count, spills, and MFMA efficiency, then printing a markdown performance table.

### Prerequisites

- Requires `rocprofv3` and the ATT decoder library

### Usage

```bash
# a16w16 kernels (run from anywhere):
python scripts/run_perf_table.py --kernel a16w16 --versions 5 6 7 8 --configs base llir llir+force-agpr+amdgcnas --K 4096 --dtype fp16

# a8w8 kernel (run from anywhere):
python scripts/run_perf_table.py --kernel a8w8 --configs llir+force-agpr+amdgcnas --K 8192

# Final-100 kernel timing with a prepared launcher and three rotating tensor sets:
python scripts/run_perf_table.py --kernel a16w16 --versions 9 --configs llir+force-agpr+amdgcnas --K 8192 --dtype bf16 --rocprof --prepared
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel` | `a16w16` | Kernel type to benchmark (`a16w16`, `a8w8`, or `a4w4`) |
| `--versions` | `5 6 7 8` | Kernel versions to benchmark (ignored for a8w8) |
| `--configs` | `base llir llir+force-agpr+amdgcnas` | Scheduler configs to test |
| `--K` | `4096` | K dimension for the GEMM problem |
| `--dtype` | `fp16` | Data type (`fp16` or `bf16`, ignored for a8w8) |
| `--rocprof` | off | Derive TFLOP/s from `rocprofv3 --kernel-trace` instead of `do_bench` |
| `--prepared` | off | With `--rocprof`, compile and bind once and enter the cached launcher directly |
| `--warmup` | `10` | Warmup dispatches in prepared mode |
| `--iters` | `1000` | Measured dispatches in prepared mode |
| `--rotating-sets` | `3` | Complete input/output tensor sets in prepared mode |
| `--last-n` | `100` | Final matching dispatches averaged from the kernel trace |

### Configs

Each config sets different environment variables before running the benchmark (see [gemm/README.md §2.1](../kernels/gemm/intra_wave/README.md#21-triton-build-and-the-out-of-tree-plugins) for the out-of-tree plugin mechanism):

- **base** — no extra env vars (default Triton scheduling)
- **llir** — `LLVM_PASS_PLUGIN_PATH=…/plugins/llir_scheduler/libLlirSched.so` + `LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE=1`
- **llir+force-agpr** — `llir` + `TRITON_FORCE_MFMA_AGPR=1` (force MFMA accumulators into AGPRs; no peephole)
- **llir+force-agpr+amdgcnas** — `llir+force-agpr` + `TRITON_AMDGCNAS_PLUGIN=1` (adds the post-assembly peephole)

### Examples

Run all a16w16 versions with all configs:

```bash
python scripts/run_perf_table.py
```

Compare v7 and v8 under base and llir configs:

```bash
python scripts/run_perf_table.py --versions 7 8 --configs base llir
```

Run a single version with a specific K and dtype:

```bash
python scripts/run_perf_table.py --versions 8 --configs base --K 8192 --dtype bf16
```

Run a8w8 kernel benchmark:

```bash
python scripts/run_perf_table.py --kernel a8w8 --configs llir+force-agpr+amdgcnas --K 8192
```

### Output

The script prints a markdown table grouped by config:

```
Config: base
| Version              | TFLOPS | VGPRs | Spills | MFMA Eff. |
|----------------------|--------|-------|--------|-----------|
| v5_local_prefetch    |   1001 |   452 |      0 |    57.98% |
| v6_loop_unroll       |   1022 |   444 |      0 |    59.50% |
...
```

If a run fails (e.g. an assertion in the scheduler), the row shows `FAIL` for the affected columns.

### Prepared kernel timing

`--prepared` removes repeated Python argument binding and specialization lookup from the dispatch loop. The driver calls Triton/Gluon's public `warmup(..., grid=...)` interface to compile without launching, binds the cached `CompiledKernel` to each rotating tensor set, and constructs launch metadata once. It then issues exactly `--warmup + --iters` kernel dispatches through the compiled launch stub. This requires no Triton compiler changes.

The reported value remains **kernel time**, not host-to-host sustained throughput: `run_perf_table.py` sorts matching CSV records by numeric dispatch ID and computes `2*M*N*K/time` from the final `--last-n` durations. Preparing the launcher is still useful because it removes artificial host gaps that can alter device duty cycle and frequency during a long profiler run. `AMD_SERIALIZE_KERNEL=3` is set for the rocprof subprocess so each dispatch has an unambiguous serialized timing interval.

The standalone driver also covers the final inter-wave kernels, which are not part of `run_perf_table.py`:

```bash
# Intra-wave BF16 v9; enable the compiler stack in the environment first.
AMD_SERIALIZE_KERNEL=3 rocprofv3 --kernel-trace -f csv \
  --kernel-include-regex v9_beyond_hotloop -d trace_intra_bf16 -- \
  python scripts/benchmark_prepared.py --route intra --kernel a16w16 \
  --version 9 --dtype bf16 --K 8192 --sets 3 --warmup 10 --iters 1000

# Inter-wave BF16; all optional compiler-plugin variables must be unset.
env -u LLVM_PASS_PLUGIN_PATH -u LLVM_PASS_PLUGIN_KEEP_TARGET_MACHINE \
  -u TRITON_FORCE_MFMA_AGPR -u TRITON_AMDGCNAS_PLUGIN \
  AMD_SERIALIZE_KERNEL=3 rocprofv3 --kernel-trace -f csv \
  --kernel-include-regex a16w16_kernel -d trace_inter_bf16 -- \
  python scripts/benchmark_prepared.py --route inter --kernel a16w16 \
  --dtype bf16 --K 8192 --sets 3 --warmup 10 --iters 1000
```

Use `--kernel a8w8 --K 16384` for BF8 and `--kernel a4w4 --version 1 --K 32768` for MXFP4. The a16w16 driver accepts `--dtype fp16` or `--dtype bf16`; the lower-precision kernels have their fixed tutorial contracts. Run the corresponding directory's `bench.py` once outside the profiler for the PyTorch correctness check before collecting a final timing trace.

## run_counter_collection.py

Automates hardware performance counter collection using `rocprofv3` across kernel versions and scheduler configs, then prints a summary table of averaged counter values.

### Prerequisites

- Requires `rocprofv3`

### Usage

```bash
python scripts/run_counter_collection.py --counters TCC_EA0_RDREQ_DRAM_sum,TCP_TCC_READ_REQ_sum
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel` | `a16w16` | Kernel type (`a16w16` or `a8w8`) |
| `--versions` | `5 6 7 8` | Kernel versions to benchmark (ignored for a8w8) |
| `--configs` | `base llir llir+force-agpr+amdgcnas` | Scheduler configs to test |
| `--K` | `4096` | K dimension for the GEMM problem |
| `--dtype` | `fp16` | Data type (`fp16` or `bf16`, ignored for a8w8) |
| `--counters` | (required) | Comma-separated list of hardware counters to collect |

### Examples

Collect L2 cache and TCP read counters for v7 and v8:

```bash
python scripts/run_counter_collection.py --versions 7 9 --configs base \
    --counters TCC_EA0_RDREQ_DRAM_sum,TCP_TCC_READ_REQ_sum --K 4096 --dtype fp16
```

### Output

```
Config: base
| Version              | TCC_EA0_RDREQ_DRAM_sum       | TCP_TCC_READ_REQ_sum         | Dispatches |
|----------------------|------------------------------|------------------------------|------------|
| v7_sliceN            |                    2,361,080 |                    8,388,608 |      1,002 |
| v9_beyond_hotloop    |                    1,574,656 |                    8,388,608 |      1,002 |
```

## run_att.py

Runs `rocprofv3` with Advanced Thread Trace (ATT) on a Python command, then automatically post-processes the trace with `process_json.py` to extract MFMA efficiency and timing breakdowns.

### Prerequisites

- Requires `rocprofv3` and the ATT decoder library
- Requires an `att_matmul.json` config file in the current directory

> [!IMPORTANT]
> The script sets `ROCPROF_ATT_LIBRARY_PATH` to an example path that may not match your system.
> Update the `ROCPROF_ATT_LIBRARY_PATH` default in `run_att.py` (in `run_rocprof_att()`) to point to your ATT decoder library location.

### Usage

```bash
python scripts/run_att.py --att-output <output_dir> <python_command>
```

### Options

| Flag | Required | Description |
|------|----------|-------------|
| `--att-output` | Yes | Output directory for rocprofv3 ATT traces |

### Examples

Profile a16w16 kernel v8:

```bash
cd kernels/gemm/intra_wave/a16w16
python ../../../../scripts/run_att.py --att-output tmp python bench.py --K 4096 --dtype fp16 --version 8
```

### What It Does

1. Runs `rocprofv3 --att` with the specified Python command
2. Locates the generated `ui_*` output directory
3. Calls `process_json.py` to analyze the traces and print results

## process_json.py

Analyzes ATT trace output (`code.json` and wave timing files) to compute loop timing breakdowns and MFMA efficiency. Called automatically by `run_att.py`.

```bash
python scripts/process_json.py path/to/ui_output_dir
```

## calc_kernel_time.py

Calculates average kernel execution time from a rocprofv3 kernel trace CSV file.

### Usage

```bash
python scripts/calc_kernel_time.py <csv_file> <kernel_name> [--unit ns|us|ms]
```

### Options

| Argument | Description |
|----------|-------------|
| `csv_file` | Path to the kernel trace CSV file generated by `rocprofv3 --kernel-trace` |
| `kernel_name` | Kernel name to filter (substring match) |
| `--unit` | Time unit for output: `ns`, `us` (default), or `ms` |

### Example

```bash
# Collect kernel trace (-f csv is required: rocprofv3 on ROCm 7.0+
# defaults to a binary format that calc_kernel_time.py cannot read).
rocprofv3 --kernel-trace -f csv -d out -- python bench.py --version 8 --K 8192 --dtype fp16 --rocprof

# Calculate average kernel time
python scripts/calc_kernel_time.py out/kernel_trace.csv matmul_kernel
```

### Output

```
Kernel : matmul_kernel
File   : out/kernel_trace.csv
Matches: 1000
Avg    : 245.67 us
Min    : 243.12 us
Max    : 248.91 us
```

To convert to TFLOPS: `TFLOPS = 2 × M × N × K / (time_in_seconds × 10^12)`

---

# Attention scripts

Everything below belongs to [`kernels/attention/`](../kernels/attention/README.md). All of
them want that section's environment in front of them — the llirSched plugin and
`disable-machine-sink` — and `FA_MODULE=fmha_v3` (default) or `fmha_v4` to pick the kernel.

## fa_kernel_time.py

The reported FA throughput metric, and the counterpart of `run_perf_table.py --rocprof` on
the GEMM side: `rocprofv3 --kernel-trace` with `AMD_SERIALIZE_KERNEL=3`, averaging the final
N of 1000 dispatches and dividing the FLOP count by that. Kernel timestamps rather than
`do_bench` wall time, because wall time charges the kernel for host-side launch gaps — and
those gaps idle the GPU, which lets the clock drift between dispatches.

```bash
FA_MODULE=fmha_v4 python scripts/fa_kernel_time.py \
  --batch 32 --seqlen 8192 --hq 8 --hk 8 --d 128 --iters 1000 --last-n 100
```

| Flag | Default | Description |
|------|---------|-------------|
| `--batch` / `--seqlen` / `--hq` / `--hk` / `--d` | `1 / 16320 / 64 / 64 / 128` | problem shape; `--hq != --hk` is GQA/MQA |
| `--dtype` | `bf16` | `bf16` or `fp16` |
| `--layout` | `bhsd` | `bhsd` or `bshd` |
| `--iters` / `--last-n` | `1000` / `100` | measured dispatches, and how many of the final ones to average |
| `--launch` | `prepared` | `prepared` binds arguments once and re-enters the launch stub; `jit` / `both` attribute a delta to the launch path itself |
| `--scale-on-q` | `1` | `0` applies `qk_scale` per element inside VEC1 instead of pre-scaling Q |
| `--rotating-buffer-size` | `512` MB | working set spread across rotating tensor sets, so dispatches read cold data rather than MALL-resident data |
| `--no-serialize` | off | leave `AMD_SERIALIZE_KERNEL` unset and let dispatches queue back to back |

## fly_kernel_time.py

Times ROCm/FlyDSL's `flash_attn_dualwave_swp` under *this* repo's protocol, which is the only
reason its numbers can share a table with ours. FlyDSL's own benchmark averages a shallower
window, which leaves the kernel in its thermal transient: six consecutive runs of one config
measured 1236.9, 1242.8, 1165.9, 1167.7, 1159.3 and 1157.5 TFLOPS. Same `rocprofv3`
invocation, same serialization, same rotating-buffer rule and same averaging window as
`fa_kernel_time.py`. Checks the output against `scaled_dot_product_attention` before timing.

```bash
FLYDSL_ROOT=/path/to/FlyDSL python scripts/fly_kernel_time.py \
  --batch 32 --seqlen 8192 --hq 8 --d 128 --iters 1000 --last-n 100
```

Needs a [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) checkout (`pip install -e .` per its own
README). `--eager-rescale` selects the `fmha_v3`-equivalent path; `--setprio`, `--stagger`,
`--waves-per-eu` and `--causal` expose its builder's knobs, whose shipped defaults are its
optimum. Note the builder defaults to `causal=True` while this script passes `causal=False`
to match our kernels.

## plot_fmha_summary.py

Regenerates `kernels/attention/images/results.png`, the summary chart at the top of the
attention README. Same shape as `plot_perf_summary.py` for GEMM: grouped TFLOPS bars with the
per-SIMD in-loop MFMA efficiency in red inside each bar. The numbers live in a `groups` table
at the top of the file — edit it when they change and re-run.

```bash
python scripts/plot_fmha_summary.py
```
