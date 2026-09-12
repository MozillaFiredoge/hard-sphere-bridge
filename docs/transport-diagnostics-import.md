# Transport diagnostics: draft source import

## Provenance and scope

Base: `2f0ffbe9af90a710cb20409f7b752ab1d705b8a6`.
Contributor: 刘华林 (Hualin Liu), `hualinliu@cjlu.edu.cn`.
Imported with permission on 2026-09-12 from the supplied
`paper1-reproducibility/` source snapshot. The eight imported source/header files
match that snapshot. No original development history was supplied, so this import
does not reconstruct earlier commits or identify the production binary's source
version. Subsequent upstream tests are recorded separately from contributor code.

This is a **draft for review**, not a release or validation of a manuscript.
It contains no production data, manuscript, internal review correspondence,
cluster submission scripts, or Python paper-analysis scripts. Those analyses need
a separate review, including consistency of their transport-coefficient formulas.

## Changes visible to users

- Shock-tube initialization accepts optional `initial_condition` fields
  `transverse_shear_amplitude` (default 0) and `transverse_shear_mode` (default 1).
  They add `A sin(2 pi mode x / Lx)` to the local y velocity. The intended
  wall-free shear experiment requires a periodic x boundary; this import does
  not enforce that condition.
- `macroscopic.csv` appends `velocity_x_squared_sum` and
  `velocity_x_y_product_sum`. These are raw sums of vx squared and vx*vy, not
  centered stresses. With slab volume V, the kinetic pressure components are
  `Pxx_k = m*sum(vx^2)/V - rho*ux^2` and
  `Pxy_k = m*sum(vx*vy)/V - rho*ux*uy`. An xx viscous residual additionally
  subtracts the chosen equilibrium pressure. The supplied xy header comment
  omits the convective subtraction and should not be used as a general formula.
- Shock-tube evolution writes `interface_fluxes.csv` on CPU and CUDA paths.
  Columns: frame, time, interface, x, mass_flux, momentum_x_flux, energy_flux.
  Values are normalized by elapsed time and interface area, positive toward +x.
  This is a **frame-endpoint estimator**, not exact event-by-event crossing data.
- CUDA shock-tube evolution additionally writes `collisional_flux.csv` with
  xx momentum and energy transfer across x interfaces. It adds device buffers,
  download/reset APIs, and rollback checkpoint copies.
- Existing output schemas gain columns/files; consumers should use column names.
  Shock-tube runs also incur particle-copy, allocation and I/O overhead. No
  performance-neutrality claim is made.

## Known limitations: do not merge as validated diagnostics yet

1. Endpoint crossing estimates miss round trips and cannot reconstruct collisions
   between frames. Periodic x wrapping also needs explicit handling. Use the
   volume moments for instantaneous kinetic stress; do not describe the endpoint
   CSV as an exact control-surface conservation ledger.
2. CUDA transfer accumulation passes a hard-coded particle mass of 1. Results
   must not be used for non-unit masses before this is corrected and tested.
3. CUDA CSV normalization uses the configured frame interval, not the actual
   last interval. A shortened final frame is consequently misnormalized.
4. The collision observer is attached to the accepted parallel event kernel.
   Coverage of exact-recovery and other collision paths, periodic seams, and
   local momentum/energy balance still needs dedicated validation. No CPU
   collisional-transfer counterpart is included.
5. The collisional CSV is opened in append mode while header state is local to
   each run. Reusing an output directory can mix old/new records and headers.
   Use a fresh output directory. Stream precision/error handling needs review.
6. Public flux APIs need input validation (particle-array lengths, positive time,
   interface count at least 2, repeated device setup). Shear mode parsing casts
   a floating-point number to size_t without validating a positive integer.
7. No new CUDA tests accompanied the snapshot. CPU build/test success cannot
   establish GPU correctness, production reproducibility, or physical validity.

These issues are deliberately documented rather than silently changing the
contributor snapshot. Fixes should be separate, reviewable commits before merging
this branch into the main development line.

## Verification plan

Build a clean checkout (do not rely on unrelated uncommitted local changes):

```sh
cmake --preset cpu-debug
cmake --build --preset cpu-debug
ctest --preset cpu-debug
./build/cpu-debug/particles_cli smoke
./build/cpu-debug/particles_cli shock-tube configs/shock_tube_smoke.json output/transport-smoke
```

Upstream regression tests cover analytic velocity second moments, ballistic
bidirectional crossing signs/normalization, and default shear fields. They do
not certify the limitations above. Before merging, add CUDA single-collision,
rollback/recovery, mass-scaling, final-frame normalization, and local conservation
tests, then validate representative production configurations on a GPU.

## Local import verification (2026-09-12)

Contributor source snapshot: commit `833e76f`. Verification used a clean export
of the staged repository, including the upstream regression tests but excluding
all unrelated worktree edits. Build and run artifacts were placed under
`/tmp/transport-import-nklVTU/`, not committed.

- GNU C++ 13.3 / Boost 1.83, Debug CPU build: passed.
- CPU CTest: 4/4 passed (including the expanded unit-test executable).
- CLI `smoke`: passed.
- CPU `shock-tube` with `configs/shock_tube_smoke.json`: 4096 particles,
  two completed evolution intervals; maximum reported absolute relative energy
  drift about 3.13e-16. New moment columns and endpoint-flux CSV were emitted.
- CUDA 12.8.93 Release build targeting architecture 86: all targets compiled and
  linked, including the CUDA tests. This architecture was an explicit compile
  target, not a detected local GPU.
- GPU execution was not performed: the environment reported GPU access blocked.
  Compilation is not numerical validation of the new CUDA observer.
- `git diff --check`: passed.

The eight imported source/header files were preserved unchanged from the supplied
snapshot; no scientific or runtime limitations listed above were silently fixed.
