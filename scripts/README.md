# Scripts

Helper scripts for profiling and benchmarking Triton GEMM kernels.

## run_perf_table.py

Automates running benchmarks across kernel versions and scheduler configs, collecting TFLOPS, VGPR count, spills, and MFMA efficiency, then printing a markdown performance table.

### Prerequisites

- Requires `rocprofv3` and the ATT decoder library

### Usage

```bash
# a16w16 kernels (run from anywhere):
python scripts/run_perf_table.py --kernel a16w16 --versions 5 6 7 8 --configs base llir llir+amdgcnas --K 4096 --dtype fp16

# a8w8 kernel (run from anywhere):
python scripts/run_perf_table.py --kernel a8w8 --configs llir+amdgcnas --K 8192
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--kernel` | `a16w16` | Kernel type to benchmark (`a16w16` or `a8w8`) |
| `--versions` | `5 6 7 8` | Kernel versions to benchmark (ignored for a8w8) |
| `--configs` | `base llir llir+amdgcnas` | Scheduler configs to test |
| `--K` | `4096` | K dimension for the GEMM problem |
| `--dtype` | `fp16` | Data type (`fp16` or `bf16`, ignored for a8w8) |

### Configs

Each config sets different environment variables before running the benchmark:

- **base** — no extra env vars (default Triton scheduling)
- **llir** — `TRITON_ENABLE_LLIR_SCHED=1`
- **llir+amdgcnas** — `TRITON_ENABLE_LLIR_SCHED=1` + `TRITON_ENABLE_AMDGCN_AS=1`

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
python scripts/run_perf_table.py --kernel a8w8 --configs llir+amdgcnas --K 8192
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
| `--configs` | `base llir llir+amdgcnas` | Scheduler configs to test |
| `--K` | `4096` | K dimension for the GEMM problem |
| `--dtype` | `fp16` | Data type (`fp16` or `bf16`, ignored for a8w8) |
| `--counters` | (required) | Comma-separated list of hardware counters to collect |

### Examples

Collect L2 cache and TCP read counters for v7 and v8:

```bash
python scripts/run_counter_collection.py --versions 7 8 --configs base \
    --counters TCC_EA0_RDREQ_DRAM_sum,TCP_TCC_READ_REQ_sum --K 4096 --dtype fp16
```

### Output

```
Config: base
| Version              | TCC_EA0_RDREQ_DRAM_sum       | TCP_TCC_READ_REQ_sum         | Dispatches |
|----------------------|------------------------------|------------------------------|------------|
| v7_slice             |                    2,361,080 |                    8,388,608 |      1,002 |
| v8_beyond_hotloop    |                    1,574,656 |                    8,388,608 |      1,002 |
```

## run_att.py

Runs `rocprofv3` with Advanced Thread Trace (ATT) on a Python command, then automatically post-processes the trace with `process_json.py` to extract MFMA efficiency and timing breakdowns.

### Prerequisites

- Requires `rocprofv3` and the ATT decoder library
- Requires an `att_matmul.json` config file in the current directory

> [!IMPORTANT]
> The script sets `ROCPROF_ATT_LIBRARY_PATH` to an example path that may not match your system.
> Update the path in `run_att.py` at line 31 (in `run_rocprof_att()`) to point to your ATT decoder library location.

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
cd kernels/gemm/a16w16
python ../../../scripts/run_att.py --att-output tmp python bench.py --K 4096 --dtype fp16 --version 8
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
# Collect kernel trace
rocprofv3 --kernel-trace -d out -- python bench.py --version 8 --K 8192 --dtype fp16 --rocprof

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
