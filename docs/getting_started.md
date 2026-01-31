# Getting Started

This document explains how to set up your environment, build, and run your first
gluon kernel on AMD GFX9 GPUs using this repository.

It also explains how the repository is organized and how to follow the tutorial
path from naive kernels to optimized implementations.

---

## 1. Overview

This repository is a hands-on tutorial for writing **high-performance gluon kernels**
on **AMD GFX9 GPUs**.

The focus is on:
- Incremental kernel optimization
- Understanding hardware-driven design choices
- Visualizing layouts and measuring performance

This is **not** a general introduction to GPU programming or gluon. Readers are
expected to have basic familiarity with GPU concepts.

---

## 2. Prerequisites

### 2.1 Hardware Requirements

- Supported GPU architectures:
  - `gfx950`

- To be supported
  - `gfx942`

Notes:
- Performance characteristics may differ slightly across gfx9 variants.
- Some kernels or optimizations may target a specific variant.

---

### 2.2 Software Requirements

- ROCm version: 6.5 +

---

### 2.3 Expected Background

Readers are expected to be familiar with:
- Basic GPU programming concepts (threads, blocks, memory hierarchy)
- General performance concepts (memory-bound vs compute-bound)

Helpful but not required:
- Experience with AMD GPUs
- Familiarity with ROCm tooling

---

## 3. Repository Layout

At a high level, the repository is organized as follows:

- `kernels/`  
  Step-by-step gluon kernel implementations, from naive to optimized versions.

- `layout_plot/`  
  A visualization tool for understanding thread and data layouts used by kernels.

- `profiling/`  
  Documentation and workflows for collecting and analyzing performance traces.

- `docs/`  
  Conceptual background and tutorial documentation.

Each kernel directory contains both code and documentation explaining the design
choices and optimizations.

---

## 4. Environment Setup

### 4.1 ROCm Environment

Set up the required ROCm environment variables:

```bash
# Example (to be filled)
export ROCM_PATH=...
export HIP_VISIBLE_DEVICES=...
```
