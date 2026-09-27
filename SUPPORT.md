# Support and Compatibility

## Positioning

This repository is **educational reference material**, not a supported product. The kernels demonstrate how to design near-peak-performance GEMM on AMD MI350/MI355 (gfx950) using Gluon. They are intended to teach techniques and provide a starting point for kernel authors, not to be deployed unmodified in production systems.

## Reproducibility

The performance numbers in this repository are reproduced against the [`gfx950-tutorial-v2.2`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v2.2) annotated tag in `triton-lang/triton` — **the current pin** — on a well-performing MI355X (see `CHANGELOG.md`). The committed IR/assembly dumps predate the v1.1 re-pin and are reproduced against [`gfx950-tutorial-v1.0`](https://github.com/triton-lang/triton/releases/tag/gfx950-tutorial-v1.0) (regeneration is pending). Those tags are immutable — they will not be moved or deleted. Building Triton from the relevant tag (or any commit reachable from it) reproduces the measurements within run-to-run noise on the same GPU and day; absolute TFLOPS differ by up to ~17% between MI355X parts and drift between measurement days with the compiler unchanged (see the v2.2 entry in `CHANGELOG.md`), so compare configurations measured together on one GPU.

Later commits on the [`gfx950-tutorial`](https://github.com/triton-lang/triton/tree/gfx950-tutorial) development branch may shift absolute numbers as the compiler evolves; the relative structure (`base` vs `llirSched` vs `llirSched + amdgcnas`) is expected to remain stable.

## Upstream trajectory

The three components the tutorial depends on are on a planned upstreaming path:

- **llirSched** — the LLIR scheduler (out-of-tree LLVM pass plugin, enabled via `LLVM_PASS_PLUGIN_PATH`) — targeted for upstream Triton (`triton-lang/triton`) around June 2026, as an opt-in pass.
- **AGPR-pinned accumulators** (formerly the force-agpr component) — no longer a plugin or a switch: from `a16w16` v7 on, and in `a8w8` and `a4w4`, the kernels pass upstream Gluon's per-call `cd_regclass="a"` ([triton-lang/triton#11792](https://github.com/triton-lang/triton/pull/11792)) to every MFMA. It replaced the process-wide `TRITON_FORCE_MFMA_AGPR` hook in `gfx950-tutorial-v2.2`. LLVM's upcoming `RewriteMFMAFormStage` pass, which picks AGPR vs. VGPR form per MFMA by register pressure, is the longer-term replacement.
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
