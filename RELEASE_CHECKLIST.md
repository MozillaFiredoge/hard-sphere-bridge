# Release checklist

## Repository

- [ ] Working tree is clean and synchronized with `origin/main`.
- [ ] No build products, bytecode, checkpoints, or large raw outputs are tracked.
- [ ] Version agrees in CMake, `CITATION.cff`, and `CHANGELOG.md`.
- [ ] User-facing changes since the previous release are in `CHANGELOG.md`.

## Validation

- [ ] GitHub CI passes on GCC, Clang, sanitizers, Python, and visualization.
- [ ] CUDA build and CUDA runtime tests pass on a documented GPU.
- [ ] CPU/CUDA comparison, causal-chain, and adaptive stress reports pass.
- [ ] Euler self-test and grid-refinement check pass.
- [ ] Conservation drift, minimum gap, rollback, and recovery metrics are reviewed.

## Reproducibility

- [ ] Release configuration files and random seeds are frozen.
- [ ] Git commit, compiler, build type, OS, CUDA toolkit/driver, and GPU are recorded.
- [ ] Representative smoke CSV files are attached to the release.
- [ ] Full-scale CSV, PNG, MP4, and manifest are archived outside the Git repository.
- [ ] Artifact hashes and download locations are included in release notes.

## Documentation

- [ ] README quick start succeeds from a fresh clone.
- [ ] CLI help and version output are accurate.
- [ ] Scientific limitations and approximation status are visible.
- [ ] Citation metadata uses the preferred author name.
- [ ] Security and contribution links work after the repository becomes public.

## Publish

- [ ] Create an annotated `v0.1.0` tag.
- [ ] Create GitHub Release notes from the changelog.
- [ ] Attach or link reproducibility artifacts.
- [ ] Make the repository public only after checking commit metadata for private data.
