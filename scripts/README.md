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

## run_att.py

Runs `rocprofv3` with Advanced Thread Trace (ATT) on a Python command, then post-processes the trace with `process_json.py`.

```bash
cd kernels/gemm/a16w16
python ../../../scripts/run_att.py --att-output tmp python bench.py --K 4096 --dtype fp16 --version 8
```

## process_json.py

Analyzes ATT trace output (`code.json` and wave timing files) to compute loop timing breakdowns and MFMA efficiency. Called automatically by `run_att.py`.

```bash
python scripts/process_json.py path/to/ui_output_dir
```
