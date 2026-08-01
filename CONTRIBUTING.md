# Contributing to Hard-Sphere Bridge

Thank you for helping improve the simulator, diagnostics, documentation, or
scientific validation.

## Before opening a change

- Use an issue for substantial algorithm or output-schema changes.
- Keep generated `output/`, frame sequences, checkpoints, build trees, and
  Python bytecode out of commits.
- Do not disable conservation, penetration, conflict, or convergence checks to
  make a benchmark appear faster.
- State clearly whether a change affects the exact CPU reference, the
  controlled parallel approximation, CUDA implementation, diagnostics, or
  visualization only.

## Build and test

A CPU-only development build is the shortest validation path:

```bash
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
```

Run the inexpensive command-line experiments:

```bash
./build/cpu-debug/particles_cli smoke
./build/cpu-debug/particles_cli equilibrium \
  configs/equilibrium.json \
  output/contributor-equilibrium
./build/cpu-debug/particles_cli shock-tube \
  configs/shock_tube_smoke.json \
  output/contributor-shock-tube
```

For Python diagnostics:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-visualization.txt
python -m py_compile tools/*.py
python tools/euler_reference.py \
  --self-test \
  --grid-check \
  configs/shock_tube.json
python tools/analyze_wave_kinematics.py --self-test
python tools/analyze_multiseed_fields.py --self-test
python tools/run_n_kn_scan.py --self-test
python tools/analyze_n_kn_scan.py --self-test
python tools/enskog_reference.py --self-test configs/shock_tube.json
python tools/analyze_enskog_comparison.py --self-test
python tools/enskog_nsf_reference.py --self-test configs/shock_tube.json
python tools/analyze_enskog_nsf_comparison.py --self-test
python tools/analyze_enskog_transport_ablation.py --self-test
python tools/analyze_enskog_transport_evolution.py --self-test
```

On a CUDA machine:

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release
ctest --preset cuda-release
```

Include the GPU model, CUDA toolkit and driver versions, selected CUDA
architectures, and complete test output in the pull request.

## Scientific changes

Changes to collision selection, response, boundaries, adaptive windows, or
macroscopic interpretation need proportionate evidence:

- an analytic or exact-reference test;
- CPU/CUDA comparison where applicable;
- momentum and energy drift;
- minimum surface gap and overlap count;
- time-window or spatial-grid refinement;
- fixed seeds and the exact configuration;
- uncertainty or multiple seeds for statistical claims.

Do not describe the large-scale window-parallel method as globally exact. The
project's limitations in `README.md` and `docs/experiment-design.md` must stay
consistent with the implementation.

## Style

- C++ uses C++20 and builds with `-Wall -Wextra -Wpedantic -Wconversion
  -Wshadow` on GCC and Clang.
- Prefer deterministic ordering and explicit units.
- Keep public declarations documented when behavior, tolerances, ownership,
  or failure modes are not obvious.
- Python visualization must remain offline and must not alter particle
  trajectories or source CSV data.
- Update documentation whenever a command, configuration field, output field,
  default, or acceptance criterion changes.

## Pull requests

Keep each pull request focused. In its description include:

- the motivation and scope;
- exact reproduction commands;
- tests and configurations used;
- performance effects, when relevant;
- any change in numerical or scientific interpretation.

By submitting a contribution, you agree that it is licensed under the
repository's Apache-2.0 license.
