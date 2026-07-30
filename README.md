# particles

CUDA-accelerated Newtonian hard-sphere simulation, with synchronized
microscopic particle dynamics, mesoscopic phase-space statistics, and
macroscopic fluid fields.

The experiment and validation requirements are defined in
[`docs/experiment-design.md`](docs/experiment-design.md).

## Requirements

- CMake 3.24 or newer;
- a C++20 compiler;
- Boost 1.75 or newer with Boost.JSON headers;
- optionally CUDA 12 or newer.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the small event-driven equilibrium reference experiment from the repository
root:

```bash
./build/particles_cli equilibrium \
  configs/equilibrium.json \
  output/equilibrium
```

This writes `summary.csv` and `velocity_histogram.csv` in the selected output
directory.

On a machine with a visible CUDA device, compare the GPU SoA/cell-list result
against the CPU reference:

```bash
./build/particles_cli cuda-grid configs/equilibrium.json
```

Run the first controlled window-parallel convergence experiment:

```bash
./build/particles_cli window-convergence \
  configs/equilibrium.json \
  output/window-convergence.csv
```

The command reuses one initialized state, runs the small \(O(N^2)\)
event-driven oracle once, and then evaluates the mutual-earliest window
approximation at the configured \(\Delta t,\Delta t/2,\Delta t/4,\ldots\).
The CSV records periodic position/velocity errors, collision-count error,
contended events, candidate-horizon clipping, conservation drift, penetration,
and wall-clock time. A CUDA build with a visible device writes paired `cpu`
and `cuda` rows and includes their RMS state difference.

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
- validated JSON loading for shock-tube and equilibrium experiments;
- reproducible non-overlapping spatial initialization;
- Maxwell velocity sampling with exact sampled-momentum correction;
- CPU continuous collision-time prediction for equal hard spheres;
- CPU elastic collision response;
- a small-scale, `O(N^2)` CPU event-driven reference simulator;
- reflecting walls and periodic boundary crossings;
- hard-sphere collisions through periodic images;
- shared CPU/CUDA uniform-grid indexing;
- CUDA structure-of-arrays particle storage;
- CUDA in-place ballistic free flight;
- CUDA cell-id generation and CUB radix sorting;
- dense cell start/end ranges with explicit empty-cell sentinels;
- two-pass CUDA neighbor-pair generation using CUB exclusive scan;
- periodic 27-cell traversal with duplicate-cell and duplicate-pair suppression;
- conservative candidate-validity horizon from cell width, speed, and walls;
- CPU/CUDA continuous collision-time evaluation over periodic images;
- compact valid-event arrays produced by a second CUB exclusive scan;
- CUB minimum reduction for the globally earliest collision time;
- deterministic earliest-time batches anchored by an absolute tolerance;
- per-particle participation counts that separate parallel-safe events from
  shared-particle collision chains;
- deterministic minimum-pair serialization for shared-particle chains;
- zero-time event rebuilds that propagate Newton-cradle-style impulses;
- CPU/CUDA advancement to the shared earliest-batch time;
- race-free equal-mass impulses for parallel-safe event indices;
- post-collision periodic wrapping and stale-event invalidation;
- batch momentum, kinetic-energy, state-match, and penetration diagnostics;
- per-particle reflecting-wall times with multi-axis corner masks;
- repeated CUDA cell/event rebuilds until a requested time window completes;
- explicit `collision_conflict`, `mixed_event_tie`, and `batch_limit` stops;
- a CPU prototype of the fixed-window mutual-earliest collision rule;
- a CUDA implementation using per-particle minimum-time and deterministic
  pair-key selection;
- race-free CUDA response for disjoint accepted events, including each
  event's post-collision remainder of the window;
- device-side endpoint gap and overlap reductions;
- automatic \(\Delta t,\Delta t/2,\Delta t/4\) comparison with the exact
  event-driven reference;
- convergence CSV output with state error, collision-count, conflict,
  conservation, candidate-horizon, penetration, and runtime diagnostics;
- conservation and analytic collision tests;
- an optional CUDA target that verifies the local CUDA toolchain.

The CPU implementation intentionally recomputes all candidate events after
every event. It is a correctness oracle, not the path to million-particle
performance.

The CUDA correctness controller now repeatedly rebuilds events, resolves
parallel-safe collisions, deterministically serializes collision chains, and
handles reflecting walls until its requested time or an explicit stop
condition. It still performs a global earliest-event reduction per batch, so
it is not the final million-particle throughput path. The window-parallel rule
and convergence report now run on both CPU and CUDA. The next M3 step is to
exercise deterministic conflict stress cases on a real GPU, then profile and
remove per-window host synchronization before scaling toward \(2^{20}\)
particles.
