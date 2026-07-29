# Project Roadmap

This document outlines the development roadmap for the GFX9 Gluon Tutorials project.

## Project Goals

Build a comprehensive collection of optimized GPU kernels using Gluon, with:
- Step-by-step optimization tutorials
- Performance analysis and profiling
- Scheduler tooling (llirSched + amdgcnas)
- Coverage of compute-bound and memory-bound workloads

## Status Legend

| Icon | Meaning |
|:----:|---------|
| :white_check_mark: | Done |
| :construction: | In Progress |
| :calendar: | Planned |
| :hourglass: | Blocked |
| :grey_question: | Low Priority / TBD |

## Task List

| Workstream | Task | Status |
|------------|------|:------:|
| **Compute-Bound GEMM** | | |
| Tile Sizes | Generalize v9_beyond_hotloop with tile 128×256×64, 256×128×64, and 128×128×128 | :calendar: |
| Tile Sizes | Generalize llirSched and amdgcnas to work with more tile sizes | :calendar: |
| Tile Sizes | Design heuristic to pick tile size based on problem size | :calendar: |
| 4-bit + Scales | Implement baseline 4-bit MoE kernel in Gluon | :calendar: |
| 4-bit + Scales | Optimize with llirSched + amdgcnas | :calendar: |
| 4-bit + Scales | Document preshuffling and related optimizations | :calendar: |
| 8-wave Warp-Pipeline | Ship warp-pipeline (pingpong) GEMM — a16w16, a8w8, a4w4 (`kernels/gemm/*-8wave/`) | :white_check_mark: |
| **FlashAttention** | | |
| FMHA v3 8-Wave | Port the existing Triton FAv3 kernel to Gluon (`kernels/attention/fmha_v3.py`) | :white_check_mark: |
| FMHA v3 8-Wave | Adapt llirSched for the 8-wave dot clusters (MFMA ↔ VALU co-execution model) | :white_check_mark: |
| FMHA v4 8-Wave | Lazy softmax rescale, per-wave skip via `gl.warp_predicate` (`fmha_v4.py`) | :white_check_mark: |
| FA 8-Wave | Adapt amdgcnas for the FA kernels | :grey_question: |
| FMHA 8-Wave | Causal masking and ragged tails (removed from the tutorial cut) | :calendar: |
| Other FA kernels | MLA, and decode-shaped MQA / GQA — separate pipelines beside `fmha_*` | :calendar: |
| **Memory-Bound Kernels** | | |
| GEMM Experiments | Implement experiments to verify the bandwidth model | :calendar: |
| Decode FA | Implement MLA or PA kernel and optimize based on the bandwidth model | :calendar: |
| **LLVM Path** | | |
| LLVM Path | Investigate LLVM scheduling infrastructure | :construction: |
| LLVM Path | Prototype LLVM-based scheduler | :construction: |
| **Backlog** | | |
| Future | 4-wave solution for FA | :grey_question: |

---

*Last updated: 2026-07-29*
