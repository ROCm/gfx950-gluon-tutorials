# MFMA Efficiency

## What is MFMA Efficiency?

MFMA efficiency measures how well a kernel utilizes the matrix core. It is defined as the ratio of cycles spent executing MFMA instructions to the total cycles of the main loop, measured per SIMD.

A higher MFMA efficiency means better matrix core utilization. For a compute-bound kernel, the ideal target is close to 100%.

## Why Measure MFMA Efficiency?

TFLOPS is the standard end-to-end performance metric, but it can vary due to factors outside the kernel developer's control—temperature, cooling, system load, and frequency scaling all affect the result.

MFMA efficiency is cycle-based and independent of clock frequency, making it more stable and reproducible across runs. Because it directly reflects instruction scheduling and co-execution behavior, it provides clearer guidance for kernel optimization and compiler tuning.

## How to Measure It

1. **Collect thread traces** using `rocprofv3`. Detailed steps are available at: [Triton Profiling with ATT](https://amd.atlassian.net/wiki/spaces/MLSE/pages/744188574/Triton+Profiling+with+ATT)

2. **Process the traces** using the [`process_json.py`](../scripts/process_json.py) script on the generated `ui_` directory:

```bash
python scripts/process_json.py /path/to/ui_<kernel_name>
```

The script analyzes `code.json` and wave trace files (`se0_sm0_sl0_wv*.json`) to:
- Identify the loop and epilogue boundaries based on instruction hit counts
- Compute cycle durations for prologue, loop, and epilogue phases
- Count MFMA instructions in the loop and calculate total MFMA cycles
- Derive MFMA efficiency as `total_mfma_cycles / average_iteration_duration`

Example output:

```json
{
  "loop_first_index": 773,
  "epilogue_first_index": 1038,
  "mfma_count_in_loop": 128,
  "total_mfma_cycles_in_loop": 2048,
  "loop_hitcount": 1016,
  "epilogue_hitcount": 8,
  "num_iterations": 127.0,
  "wave_durations": {
    "se0_sm0_sl0_wv0.json": 456316
  },
  "average_loop_duration": 456316.0,
  "average_prologue_duration": 2708.0,
  "average_epilogue_duration": 18552.0,
  "pro_ratio": "0.57%",
  "loop_ratio": "95.55%",
  "epi_ratio": "3.88%",
  "average_iteration_duration": 3593.0393700787404,
  "mfma efficiency": "57.00%"
}
```

The output includes:
- **Cycle ratios** for prologue, loop, and epilogue phases
- **MFMA efficiency** within the main loop
- **Iteration statistics** for performance analysis

For deeper investigation, ATT Viewer can visualize thread traces and allow you to zoom into specific code regions. For routine performance tracking, the script output is sufficient.
