# Scientific context and references

Hard-Sphere Bridge is a numerical research and visualization project. It is
inspired by rigorous work connecting Newtonian hard-sphere dynamics,
Boltzmann's kinetic equation, and fluid equations, but it does not implement
those proofs and must not be cited as a proof of Hilbert's sixth problem.

## Mathematical context

1. Yu Deng, Zaher Hani, and Xiao Ma,
   [Long time derivation of the Boltzmann equation from hard sphere dynamics](https://arxiv.org/abs/2408.07818),
   arXiv:2408.07818.
2. Yu Deng, Zaher Hani, and Xiao Ma,
   [Hilbert's sixth problem: derivation of fluid equations via Boltzmann's kinetic theory](https://arxiv.org/abs/2503.01800),
   arXiv:2503.01800.
3. Thierry Bodineau, Isabelle Gallagher, Laure Saint-Raymond, and Sergio
   Simonella,
   [Derivation of the Boltzmann equation from hard-sphere dynamics (after Y. Deng, Z. Hani, and X. Ma)](https://arxiv.org/abs/2602.04407),
   arXiv:2602.04407.
4. Oscar E. Lanford III, *Time evolution of large classical systems*,
   Dynamical Systems, Theory and Applications, Lecture Notes in Physics 38,
   1–111, 1975.

## Numerical context

- The CPU reference is a deliberately small, globally event-driven
  hard-sphere implementation that rebuilds affected event information for
  correctness rather than throughput.
- The large CUDA path uses continuous collision-time prediction inside
  adaptive parallel windows. It is a controlled numerical approximation with
  rollback and deterministic exact short-window recovery, not an
  implementation of the limiting argument in the papers above.
- The one-dimensional Euler reference uses primitive-variable MUSCL
  reconstruction, a minmod limiter, local Lax--Friedrichs/Rusanov fluxes, and
  SSP-RK2 time integration. It is a numerical comparison curve and never
  advances or corrects particle states.

## Dense hard-sphere context

1. Norman F. Carnahan and Kenneth E. Starling,
   [Equation of State for Nonattracting Rigid Spheres](https://doi.org/10.1063/1.1672048),
   *Journal of Chemical Physics* 51, 635--636, 1969.
2. José María Montanero and Andrés Santos,
   [Monte Carlo simulation method for the Enskog equation](https://doi.org/10.1103/PhysRevE.54.438),
   *Physical Review E* 54, 438--444, 1996.

The first checked-in Enskog comparison is an inviscid hard-sphere
hydrodynamic reference. It uses the Carnahan--Starling compressibility factor

\[
Z(\eta)=\frac{1+\eta+\eta^2-\eta^3}{(1-\eta)^3},\qquad
\eta=\frac{\pi}{6}nd^3,
\]

and advances Euler conservation laws with total mechanical pressure
\(p=nk_BT Z\). The particle CSV's existing `pressure` column is only the
kinetic moment \(nk_BT\); for the matched dense-gas comparison, the analyzer
applies the same Carnahan--Starling closure to each independent particle seed
before averaging.

The second comparison adds the Navier--Stokes-order elastic-hard-sphere
transport coefficients in equations (2.15)--(2.21) of Montanero and Santos:
first-Sonine shear viscosity, bulk viscosity, and thermal conductivity. It
uses the normal stress
\(\tau_{xx}=(4\mu/3+\zeta)\partial_xu\) and Fourier heat flux
\(q_x=-\kappa\partial_xT\) in a conservative one-dimensional finite-volume
solver. This reference is still a first-gradient continuum closure, not a
numerical solution of the full nonlocal Enskog kinetic equation. In
particular, it should not be expected to remain quantitatively accurate once
the local Knudsen number is no longer small.

## Citation of this software

Use the metadata in the repository's `CITATION.cff` and cite an immutable
release tag. When discussing a generated result, also report its configuration
file, random seed, Git commit, compiler/build type, CUDA environment, GPU, and
diagnostic resolutions.
