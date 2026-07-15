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
| FAv3 8-Wave | Port existing Triton FAv3 kernel to Gluon | :calendar: |
| FAv3 8-Wave | Adapt llirSched and amdgcnas for 8-wave solution | :calendar: |
| **Memory-Bound Kernels** | | |
| GEMM Experiments | Implement experiments to verify the bandwidth model | :calendar: |
| Decode FA | Implement MLA or PA kernel and optimize based on the bandwidth model | :calendar: |
| **LLVM Path** | | |
| LLVM Path | Investigate LLVM scheduling infrastructure | :construction: |
| LLVM Path | Prototype LLVM-based scheduler | :construction: |
| **Backlog** | | |
| Future | 4-wave solution for FA | :grey_question: |

---

*Last updated: 2026-07-07*
