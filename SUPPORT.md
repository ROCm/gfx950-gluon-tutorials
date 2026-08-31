# Support and Compatibility

## Positioning

This repository is **educational reference material**, not a supported product. The kernels demonstrate how to design near-peak-performance GEMM on AMD MI350/MI355 (gfx950) using Gluon. They are intended to teach techniques and provide a starting point for kernel authors, not to be deployed unmodified in production systems.

## Reproducibility

The GEMM performance numbers in this repository are reproduced against the [`gfx950-tutorial-v1.1`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v1.1) annotated tag in `triton-lang/triton`; the attention numbers against [`gfx950-tutorial-v2.0`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v2.0), which the attention kernels require. the committed IR/assembly dumps predate the v1.1 re-pin and are reproduced against [`gfx950-tutorial-v1.0`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v1.0) (regeneration against v1.1 is pending). Those tags are immutable — they will not be moved or deleted. **The current pin is
[`gfx950-tutorial-v2.1`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v2.1)**; the numbers above have not yet been re-measured against it (see
`CHANGELOG.md`), so they are quoted with the tag they were taken on. Building Triton from the relevant tag (or any commit reachable from it) reproduces the measurements within run-to-run noise.

Later commits on the [`gfx950-tutorial`](https://github.com/triton-lang/triton/tree/gfx950-tutorial) development branch may shift absolute numbers as the compiler evolves; the relative structure (`base` vs `llirSched` vs `llirSched + amdgcnas`) is expected to remain stable.

## Upstream trajectory

The three components the tutorial depends on are on a planned upstreaming path:

- **llirSched** — the LLIR scheduler (out-of-tree LLVM pass plugin, enabled via `LLVM_PASS_PLUGIN_PATH`) — targeted for upstream Triton (`triton-lang/triton`) around June 2026, as an opt-in pass.
- **force-agpr** — the AGPR register-allocation hint (enabled via `TRITON_FORCE_MFMA_AGPR`) — targeted for upstream LLVM around June 2026 as an AGPR allocator policy. Its `amdgpu-mfma-vgpr-form=0` half is a stopgap that forces *all* MFMA C/D into AGPRs; LLVM's upcoming `RewriteMFMAFormStage` pass will instead pick AGPR vs. VGPR form per MFMA by register pressure, and once it defaults on that flag can be dropped from `llvm.cc`.
- **`amdgcnas`** — the post-assembly peephole (out-of-tree plugin, enabled via `TRITON_AMDGCNAS_PLUGIN`) — a longer-term target for an LLVM MachineInstr-level pass.

Once these land upstream, a future revision of this repository will track the corresponding stable Triton/LLVM releases and retire the out-of-tree plugins.

## Issues and pull requests

Issues and pull requests are triaged on a **best-effort basis** by the AMD ML Software Engineering team. There is no service-level commitment.

- **Bug reports** that affect correctness or reproducibility are highest priority.
- **Documentation improvements** (typos, broken links, clearer wording) are welcome.
- **New kernel versions** beyond the existing v0–v9 progression should be discussed in an issue first; this repository is structured as a teaching narrative, and unrelated kernels likely belong in their own repository.

## Hardware

The kernels target **AMD MI350 / MI355** (gfx950). Other MI300-class parts may run the kernels but are not validated and may produce different performance. Earlier `gfx9` parts (Vega, MI50, MI100, MI200) are **not supported** by these kernels — the design relies on gfx950-specific features (e.g. `ds_read_tr`).

## Security

For security-sensitive issues, follow the process in [`SECURITY.md`](SECURITY.md).
