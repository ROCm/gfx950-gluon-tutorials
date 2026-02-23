# Scripts

Helper scripts for profiling and benchmarking Triton GEMM kernels.

## run_perf_table.py

Automates running benchmarks across kernel versions and scheduler configs, collecting TFLOPS, VGPR count, spills, and MFMA efficiency, then printing a markdown performance table.

### Prerequisites

- Must be run from the `kernels/gemm/a16w16/` directory
- Requires `rocprofv3` and the ATT decoder library
- Requires `bench.py` in the current directory

### Usage

```bash
cd kernels/gemm/a16w16
python ../../../scripts/run_perf_table.py [OPTIONS]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--versions` | `5 6 7 8` | Kernel versions to benchmark |
| `--configs` | `base llir llir+amdgcnas` | Scheduler configs to test |
| `--K` | `4096` | K dimension for the GEMM problem |
| `--dtype` | `fp16` | Data type (`fp16` or `bf16`) |

### Configs

Each config sets different environment variables before running the benchmark:

- **base** — no extra env vars (default Triton scheduling)
- **llir** — `TRITON_ENABLE_LLIR_SCHED=1`
- **llir+amdgcnas** — `TRITON_ENABLE_LLIR_SCHED=1` + `TRITON_ENABLE_AMDGCN_AS=1`

### Examples

Run all versions with all configs:

```bash
python ../../../scripts/run_perf_table.py
```

Compare v7 and v8 under base and llir configs:

```bash
python ../../../scripts/run_perf_table.py --versions 7 8 --configs base llir
```

Run a single version with a specific K and dtype:

```bash
python ../../../scripts/run_perf_table.py --versions 8 --configs base --K 8192 --dtype bf16
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
