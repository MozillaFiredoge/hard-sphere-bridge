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

Run the deterministic causal-chain convergence experiment:

```bash
./build/particles_cli window-scenario \
  configs/causal_chain.json \
  output/causal-chain-convergence.csv
```

This three-sphere case is deliberately nontrivial. With the configured
\(\Delta t=0.5\), the first collision creates a second collision that was not
present in the start-of-window event list, so the run must stop with
`overlap_detected`. At \(\Delta t=0.25\) and \(0.125\), the intervening rebuild
finds the causal collision and both runs should complete with two collisions
and no overlap. A coarse-level stop is therefore the expected convergence
signal, not a test failure.

Run the multi-seed random collision stress experiment:

```bash
./build/particles_cli window-stress \
  configs/window_stress.json \
  output/window-stress
```

The default scan uses 256 particles, four fixed seeds, and eight refinements
from \(\Delta t=0.06\) to \(0.00046875\). It writes `runs.csv` with every
seed/backend/level and `summary.csv` with completion rate, overlap-stop rate,
collision-count error, contended-event rate, state error, conservation drift,
and timing. Failed runs are compared with the exact reference at their actual
stop time.

Run the transactional adaptive-window experiment on the same seeds:

```bash
./build/particles_cli window-adaptive-stress \
  configs/window_stress.json \
  output/window-adaptive-stress
```

Each step begins with a particle-state checkpoint. If endpoint diagnostics
find penetration, the attempted state is discarded and the local window is
halved. A reduced successful window is retained and doubled only after four
consecutive successful steps. `runs.csv` records attempts, rollbacks, growths,
maximum refinement depth, minimum accepted window, rejected work,
exact-reference error, and CPU/GPU state difference. The default CPU baseline
completes all four seeds with exact collision counts and reduces the rollback
rate from 70.2% for reset-to-maximum control to 17.9%.

Run the CUDA weak-scaling benchmark:

```bash
./build/particles_cli cuda-scaling \
  configs/cuda_scaling.json \
  output/cuda-scaling.csv
```

The default levels double from \(2^{10}\) to \(2^{20}\) particles while
holding number density, diameter, temperature, and packing fraction fixed.
The periodic cubic domain grows as \(N^{1/3}\). Each row records initialization,
allocation and transfer time; grid and candidate density; adaptive attempts,
rollbacks and collisions; committed and attempted particle-window throughput;
observed device memory; conservation drift; and endpoint penetration status.
This is a scaling and capacity experiment, not an exact-reference comparison.
If a device runs out of memory, completed lower levels remain in the flushed
CSV and the failing level is printed.

Generate the first micro-to-meso-to-macro snapshot from the million-particle
shock-tube initial condition:

```bash
./build/particles_cli multiscale-snapshot \
  configs/shock_tube.json \
  output/multiscale-snapshot
```

All three layers are derived from the same particle array. The command writes
stable representative particle samples to `microscopic.csv`, the
\(x-v_x\) histogram to `phase_space.csv`, slab-wise
\(\rho,u,T,p\) to `macroscopic.csv`, and cross-layer conservation checks to
`summary.csv`. The default million-particle snapshot takes about 0.22 seconds
to measure on the current CPU baseline, excluding initialization and CSV I/O.

Run the complete evolving shock-tube pipeline first with the 4096-particle
smoke configuration:

```bash
./build/particles_cli shock-tube \
  configs/shock_tube_smoke.json \
  output/shock-tube-smoke
```

Then run the configured million-particle experiment:

```bash
./build/particles_cli shock-tube \
  configs/shock_tube.json \
  output/shock-tube
```

An optional fourth argument overrides only the end time for a short recovery
or performance check. For example, this crosses the first observed dense
contact chain without waiting for the full first output interval:

```bash
./build/particles_cli shock-tube \
  configs/shock_tube.json \
  output/shock-tube-recovery-check \
  0.0012
```

The command uses CUDA when a device is visible and otherwise falls back to the
CPU implementation. It advances adaptive transactional windows between output
times, with reflecting walls on \(x\) and periodic boundaries on \(y,z\).
If a near-zero-time collision chain still overlaps at the configured minimum
window, the controller restores its checkpoint, advances that short interval
with deterministic exact-event batches, and then resumes parallel windows.
Every output time appends synchronized rows to `microscopic.csv`,
`phase_space.csv`, `macroscopic.csv`, and `summary.csv`. `evolution.csv`
records attempts, rollbacks, exact recoveries, rejected and committed minimum
gaps, collisions, energy drift, transverse momentum drift, and per-frame
simulation time.

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
- mixed periodic/reflecting parallel windows in which wall events compete with
  particle collisions and boundary-aware free flight handles post-event
  motion;
- an exact short-window recovery path for zero-time collision chains that
  cannot be eliminated by further parallel-window bisection;
- race-free CUDA response for disjoint accepted events, including each
  event's post-collision remainder of the window;
- device-side endpoint gap and overlap reductions;
- a configured causal collision-chain experiment whose coarse window fails
  and refined windows recover the exact two-collision trajectory;
- a configured multi-seed, high-collision-rate stress scan with detailed and
  aggregated CPU/CUDA reports;
- transactional CPU/CUDA particle-state checkpoints with overlap-triggered
  rollback and local window bisection;
- an adaptive stress report that separates committed collision work from
  rejected attempts;
- a constant-density CUDA weak-scaling scan from \(2^{10}\) through
  \(2^{20}\) particles with throughput, memory, conservation, and penetration
  diagnostics;
- a shared-state micro/meso/macro snapshot pipeline with representative
  particles, \(x-v_x\), slab fields, and moment-reconstruction checks;
- a multi-frame shock-tube runner with CPU fallback, CUDA execution, adaptive
  rollback, and a per-interval evolution report;
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
and fixed/adaptive reports now run on both CPU and CUDA. The next experiment is
to run the mixed-boundary shock tube on a real GPU, inspect its multi-frame
macroscopic profiles, and tune the output and window intervals from measured
rollback and throughput data.
