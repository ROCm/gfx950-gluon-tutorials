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
| Tile Sizes | Generalize v8_beyond_hotloop with tile 128×256×64, 256×128×64, and 128×128×128 | :calendar: |
| Tile Sizes | Generalize llirSched and amdgcnas to work with more tile sizes | :calendar: |
| Tile Sizes | Design heuristic to pick tile size based on problem size | :calendar: |
| Documentation | v0_naive README | :white_check_mark: |
| Documentation | v1_buffer_load README | :white_check_mark: |
| Documentation | v2_async_copy README | :white_check_mark: |
| Documentation | v3_lds README | :white_check_mark: |
| Documentation | v4_global_prefetch README | :white_check_mark: |
| Documentation | v5_local_prefetch README | :white_check_mark: |
| Documentation | v6_loop_unroll README | :white_check_mark: |
| Documentation | v7_slice README | :white_check_mark: |
| Documentation | v8_beyond_hotloop README | :white_check_mark: |
| 4-bit + Scales | Implement baseline 4-bit MoE kernel in Gluon | :calendar: |
| 4-bit + Scales | Optimize with llirSched + amdgcnas | :calendar: |
| 4-bit + Scales | Document preshuffling and related optimizations | :calendar: |
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
| Future | 8-wave pingpong for GEMM kernels | :grey_question: |
| Future | 4-wave solution for FA | :grey_question: |

## Weekly Updates

### Week of YYYY-MM-DD

**Completed:**
- (list items)

**In Progress:**
- (list items)

**Blockers:**
- (list items)

**Next Week:**
- (list items)

---

*Last updated: 2026-02-20*
