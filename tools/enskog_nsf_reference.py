#!/usr/bin/env python3
"""One-dimensional Enskog Navier--Stokes--Fourier reference.

The equilibrium pressure is the Carnahan--Starling hard-sphere closure from
``enskog_reference.py``.  The dissipative flux uses the standard first-Sonine
Enskog shear viscosity, bulk viscosity, and thermal conductivity for elastic
hard spheres.  This is a Navier--Stokes-order continuum model, not a solver for
the full nonlocal Enskog kinetic equation.

The conservative equations are advanced with the same MUSCL/Rusanov/SSP-RK2
hyperbolic discretization as the inviscid Enskog-Euler reference.  Centered
face gradients add the one-dimensional normal viscous stress

    tau_xx = (4 mu / 3 + zeta) du/dx

and heat conduction ``kappa dT/dx``.  The explicit time step satisfies both a
hyperbolic CFL bound and a conservative parabolic stability bound.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from enskog_reference import (
    EnskogEulerParameters,
    _conserved_to_primitive,
    _load_initial_state,
    _make_frame,
    _maximum_signal_speed,
    _reflecting_ghost_primitives,
    _spatial_operator,
    _state_is_physical,
    contact_pair_correlation,
    packing_fraction,
    solve_enskog_euler_reference,
)


@dataclass(frozen=True)
class EnskogNsfParameters:
    gamma: float = 5.0 / 3.0
    cells: int = 1024
    hyperbolic_cfl: float = 0.40
    diffusion_cfl: float = 0.20
    limiter_theta: float = 1.5
    maximum_packing_fraction: float = 0.49
    transport_scale: float = 1.0
    shear_viscosity_scale: float = 1.0
    bulk_viscosity_scale: float = 1.0
    thermal_conductivity_scale: float = 1.0


@dataclass(frozen=True)
class EnskogTransport:
    shear_viscosity: np.ndarray
    bulk_viscosity: np.ndarray
    thermal_conductivity: np.ndarray
    longitudinal_viscosity: np.ndarray
    momentum_diffusivity: np.ndarray
    thermal_diffusivity: np.ndarray


@dataclass
class EnskogNsfSolution:
    frames: list
    parameters: EnskogNsfParameters
    particle_diameter: float
    steps: int
    hyperbolic_limited_steps: int
    diffusion_limited_steps: int
    positivity_retries: int
    minimum_time_step: float
    initial_mass: float
    final_mass: float
    initial_energy: float
    final_energy: float
    minimum_density: float
    minimum_temperature: float
    maximum_packing_fraction: float
    maximum_momentum_diffusivity: float
    maximum_thermal_diffusivity: float

    @property
    def relative_mass_drift(self) -> float:
        return _relative_difference(self.final_mass, self.initial_mass)

    @property
    def relative_energy_drift(self) -> float:
        return _relative_difference(self.final_energy, self.initial_energy)


def _relative_difference(value: float, reference: float) -> float:
    scale = max(abs(reference), np.finfo(np.float64).tiny)
    return (value - reference) / scale


def enskog_transport_coefficients(
    number_density: np.ndarray,
    temperature: np.ndarray,
    diameter: float,
    particle_mass: float = 1.0,
    boltzmann_constant: float = 1.0,
    gamma: float = 5.0 / 3.0,
    transport_scale: float = 1.0,
    shear_viscosity_scale: float = 1.0,
    bulk_viscosity_scale: float = 1.0,
    thermal_conductivity_scale: float = 1.0,
) -> EnskogTransport:
    """Return elastic-hard-sphere Enskog transport coefficients.

    The formulas follow Montanero & Santos, Physical Review E 54, 438
    (1996), equations (2.15)--(2.21), including their first-Sonine numerical
    corrections 1.01603 and 1.02513 to the dilute Boltzmann coefficients.
    """
    number_density = np.asarray(number_density, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    if diameter <= 0.0 or not math.isfinite(diameter):
        raise ValueError("Enskog transport requires a positive finite diameter")
    if particle_mass <= 0.0 or boltzmann_constant <= 0.0:
        raise ValueError("particle mass and Boltzmann constant must be positive")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    scales = {
        "transport": transport_scale,
        "shear-viscosity": shear_viscosity_scale,
        "bulk-viscosity": bulk_viscosity_scale,
        "thermal-conductivity": thermal_conductivity_scale,
    }
    for name, scale in scales.items():
        if scale < 0.0 or not math.isfinite(scale):
            raise ValueError(f"{name} scale must be finite and nonnegative")
    if (
        np.any(number_density <= 0.0)
        or np.any(temperature <= 0.0)
        or not np.all(np.isfinite(number_density))
        or not np.all(np.isfinite(temperature))
    ):
        raise ValueError("Enskog transport state must be positive and finite")

    eta = packing_fraction(number_density, diameter)
    chi = contact_pair_correlation(eta)
    root = np.sqrt(
        particle_mass * boltzmann_constant * temperature / math.pi
    )
    dilute_shear = 1.01603 * (5.0 / 16.0) * root / diameter**2
    pi_n_d3_chi = math.pi * number_density * diameter**3 * chi
    shear_factor = 1.0 + (4.0 / 15.0) * pi_n_d3_chi
    bulk = (
        (4.0 / 9.0)
        * number_density**2
        * diameter**4
        * chi
        * np.sqrt(math.pi * particle_mass * boltzmann_constant * temperature)
    )
    shear = dilute_shear * shear_factor**2 / chi + (3.0 / 5.0) * bulk
    dilute_thermal = (
        1.02513
        * (15.0 / 4.0)
        * (boltzmann_constant / particle_mass)
        * dilute_shear
    )
    thermal_factor = 1.0 + (2.0 / 5.0) * pi_n_d3_chi
    thermal = (
        dilute_thermal * thermal_factor**2 / chi
        + (3.0 / 2.0) * (boltzmann_constant / particle_mass) * bulk
    )
    shear *= transport_scale * shear_viscosity_scale
    bulk *= transport_scale * bulk_viscosity_scale
    thermal *= transport_scale * thermal_conductivity_scale
    longitudinal = (4.0 / 3.0) * shear + bulk
    mass_density = particle_mass * number_density
    specific_heat_volume = (
        boltzmann_constant / particle_mass / (gamma - 1.0)
    )
    momentum_diffusivity = longitudinal / mass_density
    thermal_diffusivity = thermal / (mass_density * specific_heat_volume)
    arrays = (
        shear,
        bulk,
        thermal,
        longitudinal,
        momentum_diffusivity,
        thermal_diffusivity,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("Enskog transport coefficient is non-finite")
    if any(np.any(value < 0.0) for value in arrays):
        raise ValueError("Enskog transport coefficient is negative")
    return EnskogTransport(*arrays)


def _euler_parameters(parameters: EnskogNsfParameters) -> EnskogEulerParameters:
    return EnskogEulerParameters(
        gamma=parameters.gamma,
        cells=parameters.cells,
        cfl=parameters.hyperbolic_cfl,
        limiter_theta=parameters.limiter_theta,
        maximum_packing_fraction=parameters.maximum_packing_fraction,
    )


def _diffusive_operator(
    conserved: np.ndarray,
    dx: float,
    parameters: EnskogNsfParameters,
    particle_mass: float,
    boltzmann_constant: float,
    diameter: float,
) -> tuple[np.ndarray, float, float]:
    primitive = _conserved_to_primitive(
        conserved,
        particle_mass,
        boltzmann_constant,
        parameters.gamma,
    )
    extended = _reflecting_ghost_primitives(primitive)
    left = extended[1:-2]
    right = extended[2:-1]
    face_density = 0.5 * (left[:, 0] + right[:, 0])
    face_temperature = 0.5 * (left[:, 2] + right[:, 2])
    face_velocity = 0.5 * (left[:, 1] + right[:, 1])
    transport = enskog_transport_coefficients(
        face_density / particle_mass,
        face_temperature,
        diameter,
        particle_mass,
        boltzmann_constant,
        parameters.gamma,
        transport_scale=parameters.transport_scale,
        shear_viscosity_scale=parameters.shear_viscosity_scale,
        bulk_viscosity_scale=parameters.bulk_viscosity_scale,
        thermal_conductivity_scale=parameters.thermal_conductivity_scale,
    )
    velocity_gradient = (right[:, 1] - left[:, 1]) / dx
    temperature_gradient = (right[:, 2] - left[:, 2]) / dx
    normal_stress = transport.longitudinal_viscosity * velocity_gradient
    conductive_energy_flux = (
        transport.thermal_conductivity * temperature_gradient
    )
    diffusive_flux = np.zeros((len(left), 3), dtype=np.float64)
    diffusive_flux[:, 1] = normal_stress
    diffusive_flux[:, 2] = (
        face_velocity * normal_stress + conductive_energy_flux
    )
    operator = (diffusive_flux[1:] - diffusive_flux[:-1]) / dx
    return (
        operator,
        float(np.max(transport.momentum_diffusivity)),
        float(np.max(transport.thermal_diffusivity)),
    )


def _combined_operator(
    conserved: np.ndarray,
    dx: float,
    parameters: EnskogNsfParameters,
    particle_mass: float,
    boltzmann_constant: float,
    diameter: float,
    density_floor: float,
    temperature_floor: float,
) -> np.ndarray:
    hyperbolic = _spatial_operator(
        conserved,
        dx,
        _euler_parameters(parameters),
        particle_mass,
        boltzmann_constant,
        diameter,
        density_floor,
        temperature_floor,
    )
    diffusive, _, _ = _diffusive_operator(
        conserved,
        dx,
        parameters,
        particle_mass,
        boltzmann_constant,
        diameter,
    )
    return hyperbolic + diffusive


def _advance_one_step(
    conserved: np.ndarray,
    dt: float,
    dx: float,
    parameters: EnskogNsfParameters,
    particle_mass: float,
    boltzmann_constant: float,
    diameter: float,
    density_floor: float,
    temperature_floor: float,
) -> np.ndarray | None:
    arguments = (
        dx,
        parameters,
        particle_mass,
        boltzmann_constant,
        diameter,
        density_floor,
        temperature_floor,
    )
    first = conserved + dt * _combined_operator(conserved, *arguments)
    euler = _euler_parameters(parameters)
    if not _state_is_physical(
        first,
        euler,
        particle_mass,
        boltzmann_constant,
        diameter,
        density_floor,
        temperature_floor,
    ):
        return None
    second = 0.5 * conserved + 0.5 * (
        first + dt * _combined_operator(first, *arguments)
    )
    if not _state_is_physical(
        second,
        euler,
        particle_mass,
        boltzmann_constant,
        diameter,
        density_floor,
        temperature_floor,
    ):
        return None
    return second


def solve_enskog_nsf_reference(
    config_path: Path,
    frame_times: list[tuple[int, float]],
    *,
    particle_count: int | None = None,
    particle_diameter: float | None = None,
    parameters: EnskogNsfParameters = EnskogNsfParameters(),
) -> EnskogNsfSolution:
    """Solve the configured Enskog NSF problem at exact output times."""
    if parameters.cells < 8:
        raise ValueError("cell count must be at least eight")
    if parameters.gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    if not 0.0 < parameters.hyperbolic_cfl < 1.0:
        raise ValueError("hyperbolic CFL must lie in (0, 1)")
    if not 0.0 < parameters.diffusion_cfl <= 0.25:
        raise ValueError("diffusion CFL must lie in (0, 0.25]")
    if not 1.0 <= parameters.limiter_theta <= 2.0:
        raise ValueError("limiter theta must lie in [1, 2]")
    for name, scale in (
        ("transport", parameters.transport_scale),
        ("shear-viscosity", parameters.shear_viscosity_scale),
        ("bulk-viscosity", parameters.bulk_viscosity_scale),
        ("thermal-conductivity", parameters.thermal_conductivity_scale),
    ):
        if scale < 0.0 or not math.isfinite(scale):
            raise ValueError(f"{name} scale must be finite and nonnegative")
    if not frame_times:
        raise ValueError("at least one output time is required")
    ordered_targets = sorted(
        ((int(frame), float(time)) for frame, time in frame_times),
        key=lambda item: (item[1], item[0]),
    )
    if ordered_targets[0][1] < 0.0:
        raise ValueError("output times must be nonnegative")

    (
        x,
        conserved,
        dx,
        transverse_area,
        particle_mass,
        boltzmann_constant,
        diameter,
    ) = _load_initial_state(
        config_path,
        _euler_parameters(parameters),
        particle_count,
        particle_diameter,
    )
    primitive = _conserved_to_primitive(
        conserved,
        particle_mass,
        boltzmann_constant,
        parameters.gamma,
    )
    density_floor = max(
        np.finfo(np.float64).tiny,
        1.0e-12 * float(np.min(primitive[:, 0])),
    )
    temperature_floor = max(
        np.finfo(np.float64).tiny,
        1.0e-12 * float(np.min(primitive[:, 2])),
    )
    initial_mass = float(np.sum(conserved[:, 0]) * dx * transverse_area)
    initial_energy = float(np.sum(conserved[:, 2]) * dx * transverse_area)
    minimum_density = float(np.min(primitive[:, 0]))
    minimum_temperature = float(np.min(primitive[:, 2]))
    maximum_eta = float(
        np.max(packing_fraction(primitive[:, 0] / particle_mass, diameter))
    )
    maximum_momentum_diffusivity = 0.0
    maximum_thermal_diffusivity = 0.0
    minimum_time_step = math.inf
    current_time = 0.0
    steps = 0
    hyperbolic_limited_steps = 0
    diffusion_limited_steps = 0
    positivity_retries = 0
    frames = []

    for frame, target_time in ordered_targets:
        if target_time < current_time - 1.0e-14:
            raise ValueError("output times moved backward")
        while current_time < target_time - 2.0e-15:
            maximum_speed = _maximum_signal_speed(
                conserved,
                _euler_parameters(parameters),
                particle_mass,
                boltzmann_constant,
                diameter,
            )
            if not math.isfinite(maximum_speed) or maximum_speed <= 0.0:
                raise RuntimeError("invalid Enskog NSF signal speed")
            _, momentum_diffusivity, thermal_diffusivity = _diffusive_operator(
                conserved,
                dx,
                parameters,
                particle_mass,
                boltzmann_constant,
                diameter,
            )
            maximum_momentum_diffusivity = max(
                maximum_momentum_diffusivity, momentum_diffusivity
            )
            maximum_thermal_diffusivity = max(
                maximum_thermal_diffusivity, thermal_diffusivity
            )
            hyperbolic_dt = parameters.hyperbolic_cfl * dx / maximum_speed
            maximum_diffusivity = max(momentum_diffusivity, thermal_diffusivity)
            diffusion_dt = (
                parameters.diffusion_cfl * dx**2 / maximum_diffusivity
                if maximum_diffusivity > 0.0
                else math.inf
            )
            if diffusion_dt < hyperbolic_dt:
                diffusion_limited_steps += 1
            else:
                hyperbolic_limited_steps += 1
            dt = min(hyperbolic_dt, diffusion_dt, target_time - current_time)
            accepted = None
            for _ in range(20):
                accepted = _advance_one_step(
                    conserved,
                    dt,
                    dx,
                    parameters,
                    particle_mass,
                    boltzmann_constant,
                    diameter,
                    density_floor,
                    temperature_floor,
                )
                if accepted is not None:
                    break
                positivity_retries += 1
                dt *= 0.5
            if accepted is None:
                raise RuntimeError(
                    f"Enskog NSF positivity retries exhausted at t={current_time:g}"
                )
            conserved = accepted
            current_time += dt
            minimum_time_step = min(minimum_time_step, dt)
            steps += 1
        current_time = target_time
        primitive = _conserved_to_primitive(
            conserved,
            particle_mass,
            boltzmann_constant,
            parameters.gamma,
        )
        eta = packing_fraction(primitive[:, 0] / particle_mass, diameter)
        minimum_density = min(minimum_density, float(np.min(primitive[:, 0])))
        minimum_temperature = min(
            minimum_temperature, float(np.min(primitive[:, 2]))
        )
        maximum_eta = max(maximum_eta, float(np.max(eta)))
        frames.append(
            _make_frame(
                frame,
                target_time,
                x,
                conserved,
                _euler_parameters(parameters),
                particle_mass,
                boltzmann_constant,
                diameter,
            )
        )

    final_mass = float(np.sum(conserved[:, 0]) * dx * transverse_area)
    final_energy = float(np.sum(conserved[:, 2]) * dx * transverse_area)
    return EnskogNsfSolution(
        frames=frames,
        parameters=parameters,
        particle_diameter=diameter,
        steps=steps,
        hyperbolic_limited_steps=hyperbolic_limited_steps,
        diffusion_limited_steps=diffusion_limited_steps,
        positivity_retries=positivity_retries,
        minimum_time_step=minimum_time_step,
        initial_mass=initial_mass,
        final_mass=final_mass,
        initial_energy=initial_energy,
        final_energy=final_energy,
        minimum_density=minimum_density,
        minimum_temperature=minimum_temperature,
        maximum_packing_fraction=maximum_eta,
        maximum_momentum_diffusivity=maximum_momentum_diffusivity,
        maximum_thermal_diffusivity=maximum_thermal_diffusivity,
    )


def run_self_test(config_path: Path) -> int:
    density = np.array([0.1, 1.0, 10.0], dtype=np.float64)
    temperature = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    transport = enskog_transport_coefficients(
        density,
        temperature,
        0.1,
    )
    for value in (
        transport.shear_viscosity,
        transport.bulk_viscosity,
        transport.thermal_conductivity,
        transport.longitudinal_viscosity,
    ):
        if np.any(value <= 0.0) or not np.all(np.isfinite(value)):
            raise AssertionError("transport coefficient self-test failed")

    shear_only = enskog_transport_coefficients(
        density,
        temperature,
        0.1,
        bulk_viscosity_scale=0.0,
        thermal_conductivity_scale=0.0,
    )
    if not np.array_equal(
        shear_only.shear_viscosity, transport.shear_viscosity
    ):
        raise AssertionError("shear-only scaling changed shear viscosity")
    if np.any(shear_only.bulk_viscosity != 0.0):
        raise AssertionError("shear-only scaling retained bulk viscosity")
    if np.any(shear_only.thermal_conductivity != 0.0):
        raise AssertionError("shear-only scaling retained thermal conductivity")
    if not np.array_equal(
        shear_only.longitudinal_viscosity,
        (4.0 / 3.0) * shear_only.shear_viscosity,
    ):
        raise AssertionError("shear-only longitudinal viscosity is inconsistent")

    thermal_only = enskog_transport_coefficients(
        density,
        temperature,
        0.1,
        shear_viscosity_scale=0.0,
        bulk_viscosity_scale=0.0,
    )
    if np.any(thermal_only.momentum_diffusivity != 0.0):
        raise AssertionError("thermal-only scaling retained momentum diffusion")
    if np.any(thermal_only.thermal_diffusivity <= 0.0):
        raise AssertionError("thermal-only scaling removed heat diffusion")

    times = [(0, 0.0), (1, 0.02)]
    zero_transport = solve_enskog_nsf_reference(
        config_path,
        times,
        parameters=EnskogNsfParameters(cells=256, transport_scale=0.0),
    )
    inviscid = solve_enskog_euler_reference(
        config_path,
        times,
        parameters=EnskogEulerParameters(cells=256, cfl=0.40),
    )
    maximum_zero_transport_difference = 0.0
    for field in ("number_density", "bulk_velocity_x", "temperature", "pressure"):
        maximum_zero_transport_difference = max(
            maximum_zero_transport_difference,
            float(
                np.max(
                    np.abs(
                        zero_transport.frames[-1].fields[field]
                        - inviscid.frames[-1].fields[field]
                    )
                )
            ),
        )
    if maximum_zero_transport_difference > 1.0e-12:
        raise AssertionError("zero-transport NSF did not reduce to Enskog-Euler")

    viscous = solve_enskog_nsf_reference(
        config_path,
        [(0, 0.0), (1, 0.01), (2, 0.02)],
        parameters=EnskogNsfParameters(cells=256),
    )
    if viscous.minimum_density <= 0.0 or viscous.minimum_temperature <= 0.0:
        raise AssertionError("Enskog NSF self-test lost positivity")
    if abs(viscous.relative_mass_drift) > 1.0e-11:
        raise AssertionError("Enskog NSF self-test lost mass conservation")
    if abs(viscous.relative_energy_drift) > 1.0e-11:
        raise AssertionError("Enskog NSF self-test lost energy conservation")
    print("Enskog Navier--Stokes--Fourier reference self-test")
    print("  independent component scaling: passed")
    print(f"  zero-transport maximum difference: {maximum_zero_transport_difference:.3e}")
    print(f"  viscous steps: {viscous.steps}")
    print(f"  diffusion-limited steps: {viscous.diffusion_limited_steps}")
    print(f"  positivity retries: {viscous.positivity_retries}")
    print(f"  relative mass drift: {viscous.relative_mass_drift:.3e}")
    print(f"  relative energy drift: {viscous.relative_energy_drift:.3e}")
    print(f"  minimum temperature: {viscous.minimum_temperature:.9g}")
    print("  status: passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/shock_tube.json"),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate transport coefficients, inviscid limit, and conservation",
    )
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("use --self-test or import this module from an analyzer")
    return run_self_test(arguments.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
