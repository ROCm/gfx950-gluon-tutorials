# Support and Compatibility

## Positioning

This repository is **educational reference material**, not a supported product. The kernels demonstrate how to design near-peak-performance GEMM on AMD MI350/MI355 (gfx950) using Gluon. They are intended to teach techniques and provide a starting point for kernel authors, not to be deployed unmodified in production systems.

## Reproducibility

All performance numbers and IR/assembly dumps in this repository are reproduced against a single immutable Triton commit, identified by the [`gfx950-tutorial-v0.2`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v0.2) annotated tag in `triton-lang/triton`. The tag will not be moved or deleted. Building Triton from that tag (or any commit reachable from it) will reproduce the measurements within run-to-run noise.

Later commits on the [`gfx950-tutorial`](https://github.com/triton-lang/triton/tree/gfx950-tutorial) development branch may shift absolute numbers as the compiler evolves; the relative structure (`base` vs `llirSched` vs `llirSched + amdgcnas`) is expected to remain stable.

## Upstream trajectory

The two compiler features the tutorial depends on are on a planned upstreaming path:

- **LLIR scheduler** (`TRITON_ENABLE_LLIR_SCHED`) — targeted for upstream Triton (`triton-lang/triton`) around June 2026, as an opt-in pass.
- **`amdgcnas`** (`TRITON_ENABLE_AMDGCN_AS`) — the LLVM register-hint portion is targeted for upstream LLVM around June 2026; the post-assembly peephole is a longer-term target for an LLVM MachineInstr-level pass.

Once these land upstream, the tutorial's environment-variable gates will continue to work for backwards compatibility, and a future revision of this repository will track the corresponding stable Triton/LLVM releases.

## Issues and pull requests

Issues and pull requests are triaged on a **best-effort basis** by the AMD ML Software Engineering team. There is no service-level commitment.

- **Bug reports** that affect correctness or reproducibility are highest priority.
- **Documentation improvements** (typos, broken links, clearer wording) are welcome.
- **New kernel versions** beyond the existing v0–v9 progression should be discussed in an issue first; this repository is structured as a teaching narrative, and unrelated kernels likely belong in their own repository.

## Hardware

The kernels target **AMD MI350 / MI355** (gfx950). Other MI300-class parts may run the kernels but are not validated and may produce different performance. Earlier `gfx9` parts (Vega, MI50, MI100, MI200) are **not supported** by these kernels — the design relies on gfx950-specific features (e.g. `ds_read_tr`).

## Security

For security-sensitive issues, follow the process in [`SECURITY.md`](SECURITY.md).
