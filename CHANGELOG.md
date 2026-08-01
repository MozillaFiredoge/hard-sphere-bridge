# Changelog

All notable user-facing changes will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Reproducible rarefaction/contact/compression marker extraction, constrained
  propagation-speed fits, coarse-graining sensitivity output, and publication
  figures for the million-particle shock tube.
- Resumable independent-seed shock-tube batches and Student-t confidence bands
  for macroscopic particle fields against the Euler reference.
- Delete-one-seed jackknife uncertainty for nonlinear wave-marker speeds,
  including an explicit diagnostic warning when a pre-registered marker check
  is not met.
- A resumable factorial particle-count/Knudsen shock-tube scan that preserves
  local number density along the particle-count axis, reports realized local
  Knudsen ranges, and produces field, uncertainty-scaling, and wave-speed
  trend tables and figures.
- Archived results from the complete 36-trajectory CUDA scan, including the
  observed near-\(N^{-1/2}\) confidence-band scaling, local-Kn quantiles,
  field/Euler comparisons, and explicitly flagged wave-marker diagnostics.
- An independent Carnahan--Starling Enskog--Euler reference, paired
  kinetic/mechanical pressure comparisons, a 2048/4096-cell reference check,
  and velocity-error diagnostics that separate normalization, quiet-region
  noise, active-region shape, amplitude, and spatial shift.
- A conservative one-dimensional Enskog Navier--Stokes--Fourier reference
  with first-Sonine shear viscosity, bulk viscosity, and thermal conductivity,
  plus three-model field/error plots, transport diagnostics, and an archived
  512/1024-cell reference check.
- A same-grid Enskog transport ablation that independently controls shear
  viscosity, bulk viscosity, and thermal conductivity, with five-model error
  tables, component-attribution plots, and a refined bulk-increment check.
- A saved-frame Enskog transport evolution comparison that tests whether
  final-time improvements persist, and tracks velocity-pulse amplitude and
  width without launching additional particle trajectories.

### Changed

- Refined the scientific interpretation from a final-frame NSF improvement to
  a time-resolved result: shear transport explains late-time low-Kn
  broadening, while the initial kinetic transient and Kn=0.10 remain outside
  a reliable first-gradient description.
- Reorganized the README around the synchronized three-scale result, headline
  quantitative findings, reproduction entry points, and explicit claim
  boundaries.
- Extended Python self-tests, end-to-end smoke coverage, installation rules,
  and archived source hashes to the Enskog transport comparison tools.

## [0.1.0] - 2026-07-31

### Added

- Open-source project metadata, contribution guidance, security policy, issue
  templates, CMake presets, installation rules, and continuous integration.
- Explicit CLI help, version reporting, smoke command, and unknown-command
  rejection.
- Reproducible Python dependency declaration and visualization metadata.
- CPU exact event-driven hard-sphere reference.
- CUDA structure-of-arrays storage, uniform cell lists, candidate generation,
  continuous collision detection, and deterministic event handling.
- Fixed and adaptive parallel-window experiments with rollback and exact
  short-window recovery.
- CPU/CUDA correctness, causal-chain, random stress, and weak-scaling reports.
- Million-particle mixed-boundary shock-tube evolution.
- Synchronized microscopic, phase-space, and macroscopic diagnostics.
- Offline PNG/MP4 visualization and an independent ideal-gas Euler reference.

[Unreleased]: https://github.com/MozillaFiredoge/hard-sphere-bridge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MozillaFiredoge/hard-sphere-bridge/releases/tag/v0.1.0
