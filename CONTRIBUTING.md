# Contributing

Thank you for your interest in this repository. Contributions are welcome — please read the notes below first so your time is well spent.

## What kind of contributions are appropriate

This repository is a **teaching narrative**. Each kernel version exists to demonstrate one specific concept (a layout, a pipeline stage, a scheduling decision), and the README for each version is as much a part of the contribution as the code.

Most welcome:

- **Bug fixes** affecting correctness (including subtle layout / vmcnt / scheduling issues)
- **Documentation improvements** — typos, broken links, clearer wording, better diagrams
- **Reproducibility improvements** — clearer regeneration steps, missing measurement details

Discuss before submitting:

- **New kernel versions** beyond v0–v9 — open an issue first to align on whether the new version fits the existing narrative arc
- **Refactors** that touch multiple kernel versions at once
- **New data types** or new kernel families (FA, MoE, etc.) — likely belong in a separate repository

Out of scope:

- Backports to non-gfx950 hardware — the kernels rely on gfx950-specific features
- Vendoring this code into other projects without attribution

## How to submit

1. **Open an issue** describing what you want to change, especially for non-trivial changes
2. **Fork and branch** off `main`. Use a descriptive branch name (e.g. `fix-v3-bank-conflict-typo`)
3. **Run formatters** before pushing — the CI checks `black` and `ruff`:
   ```bash
   bash scripts/format_fix.sh
   bash scripts/format_check.sh
   ```
4. **Open a pull request** against `main`. Reference the issue number if one exists.
5. **Sign your commits** (DCO sign-off): `git commit -s -m "..."` — required for ROCm-organization contributions.

## Performance changes

If your change affects measured performance:

- Re-run the relevant `scripts/run_perf_table.py` invocation and update the numbers in the affected READMEs
- Disclose the hardware, ROCm version, and Triton commit you measured against
- Run on a healthy GPU (not throttled by neighbors) — see commit history for examples of `HIP_VISIBLE_DEVICES` usage

## Style

- **Black** for Python formatting (line length default)
- **Ruff** for linting (config in `pyproject.toml`)
- **Markdown** — keep paragraphs unwrapped; let the renderer handle line breaks. Use [GitHub-flavored alerts](https://github.com/orgs/community/discussions/16925) (`[!NOTE]`, `[!IMPORTANT]`, `[!TIP]`) where appropriate.

## Review and merge

Review is best-effort. Expect a turnaround of a few business days for documentation changes and longer for kernel changes that need re-measurement. Maintainers will merge once review feedback is addressed and CI is green.

## License

By contributing, you agree that your contributions will be licensed under the repository's [MIT License](LICENSE).
