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

## Citation of this software

Use the metadata in the repository's `CITATION.cff` and cite an immutable
release tag. When discussing a generated result, also report its configuration
file, random seed, Git commit, compiler/build type, CUDA environment, GPU, and
diagnostic resolutions.
