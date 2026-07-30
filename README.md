# particles

CUDA-accelerated Newtonian hard-sphere simulation, with synchronized
microscopic particle dynamics, mesoscopic phase-space statistics, and
macroscopic fluid fields.

The experiment and validation requirements are defined in
[`docs/experiment-design.md`](docs/experiment-design.md).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA is enabled automatically when `nvcc` is available. A CPU-only build can
be requested explicitly:

```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPARTICLES_ENABLE_CUDA=OFF
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

The default CUDA build targets compute capability 7.5 and retains PTX for
forward JIT compatibility. For performance measurements, configure for the
actual GPU, for example:

```bash
cmake -S . -B build-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native
```

## Current milestone

The current scaffold provides:

- dimensionless shock-tube configuration;
- CPU continuous collision-time prediction for equal hard spheres;
- CPU elastic collision response;
- conservation and analytic collision tests;
- an optional CUDA target that verifies the local CUDA toolchain.

The next milestone is the small-scale CPU event-driven reference simulator.
