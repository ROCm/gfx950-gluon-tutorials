# MFMA Efficiency

## What is MFMA Efficiency?

In the main loop, MFMA efficiency is the ratio of total cycles spent executing MFMA instructions over the total cycles of the loop per SIMD. The higher the MFMA efficiency, the better the utilization of the matrix core. An ideal compute-bound kernel should have MFMA efficiency very close to 100%.

## Why Measure MFMA Efficiency?

TFLOPS is the typical end-to-end performance metric, but it can be affected by factors outside the kernel developer's control: temperature, cooling, system state, etc. MFMA efficiency is cycle-based and does not factor in frequency, making it more stable across runs. Since this metric is directly related to instruction scheduling and co-execution, it serves as a better guide for kernel developers and compilers.

## How to Measure It

Collect thread traces via `rocprofv3`. Steps can be found at: [Triton Profiling with ATT](https://amd.atlassian.net/wiki/spaces/MLSE/pages/744188574/Triton+Profiling+with+ATT)

Then use the `process_json.py` script to process the generated `ui_` directory. Example output:

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

This shows the cycle ratio of prologue, loop, and epilogue, along with MFMA efficiency in the loop. For deeper analysis, ATT Viewer can visualize thread traces to zoom into specific code regions. For MFMA efficiency as a metric, the script suffices.
