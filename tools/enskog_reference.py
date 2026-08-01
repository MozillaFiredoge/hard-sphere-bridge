#!/usr/bin/env python3
"""One-dimensional hard-sphere Euler reference with a Carnahan--Starling EOS.

This is the equilibrium (zeroth-order hydrodynamic) reference associated with
the revised Enskog description of elastic hard spheres.  It changes the
mechanical pressure and sound speed through the hard-sphere compressibility
factor, but it does not solve the Enskog kinetic equation and does not include
Enskog viscosity or heat conduction.

For packing fraction eta = pi n d^3 / 6, the closure is

    p = n k_B T Z(eta),
    Z = (1 + eta + eta^2 - eta^3) / (1 - eta)^3.

Elastic hard spheres have no configurational internal energy away from their
instantaneous collisions, so e = k_B T / (m (gamma - 1)) remains the
monatomic kinetic internal energy.  The resulting isentropic sound speed is

    c^2 = (k_B T / m) [Z + eta Z' + (gamma - 1) Z^2].

The finite-volume discretization matches tools/euler_reference.py: MUSCL with
minmod limiting, Rusanov fluxes, SSP-RK2, and reflecting x boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from euler_reference import EulerFrame


@dataclass(frozen=True)
class EnskogEulerParameters:
    gamma: float = 5.0 / 3.0
    cells: int = 2048
    cfl: float = 0.45
    limiter_theta: float = 1.5
    maximum_packing_fraction: float = 0.49


@dataclass
class EnskogEulerSolution:
    frames: list[EulerFrame]
    parameters: EnskogEulerParameters
    particle_diameter: float
    steps: int
    initial_mass: float
    final_mass: float
    initial_energy: float
    final_energy: float
    minimum_density: float
    minimum_temperature: float
    maximum_packing_fraction: float

    @property
    def relative_mass_drift(self) -> float:
        return relative_difference(self.final_mass, self.initial_mass)

    @property
    def relative_energy_drift(self) -> float:
        return relative_difference(self.final_energy, self.initial_energy)


def relative_difference(value: float, reference: float) -> float:
    scale = max(abs(reference), np.finfo(np.float64).tiny)
    return (value - reference) / scale


def packing_fraction(number_density: np.ndarray, diameter: float) -> np.ndarray:
    if diameter < 0.0 or not math.isfinite(diameter):
        raise ValueError("hard-sphere diameter must be finite and nonnegative")
    return (math.pi / 6.0) * number_density * diameter**3


def carnahan_starling_compressibility(
    eta: np.ndarray | float,
) -> np.ndarray:
    value = np.asarray(eta, dtype=np.float64)
    if np.any(value < 0.0) or np.any(value >= 1.0):
        raise ValueError("Carnahan--Starling packing fraction must be in [0, 1)")
    return (1.0 + value + value**2 - value**3) / (1.0 - value) ** 3


def carnahan_starling_derivative(
    eta: np.ndarray | float,
) -> np.ndarray:
    value = np.asarray(eta, dtype=np.float64)
    if np.any(value < 0.0) or np.any(value >= 1.0):
        raise ValueError("Carnahan--Starling packing fraction must be in [0, 1)")
    numerator = 1.0 + value + value**2 - value**3
    numerator_derivative = 1.0 + 2.0 * value - 3.0 * value**2
    return (
        numerator_derivative / (1.0 - value) ** 3
        + 3.0 * numerator / (1.0 - value) ** 4
    )


def contact_pair_correlation(eta: np.ndarray | float) -> np.ndarray:
    value = np.asarray(eta, dtype=np.float64)
    if np.any(value < 0.0) or np.any(value >= 1.0):
        raise ValueError("contact pair correlation requires eta in [0, 1)")
    return (1.0 - 0.5 * value) / (1.0 - value) ** 3


def enskog_pressure(
    number_density: np.ndarray,
    temperature: np.ndarray,
    diameter: float,
    boltzmann_constant: float = 1.0,
) -> np.ndarray:
    eta = packing_fraction(number_density, diameter)
    return (
        number_density
        * boltzmann_constant
        * temperature
        * carnahan_starling_compressibility(eta)
    )


def _primitive_to_conserved(
    primitive: np.ndarray,
    particle_mass: float,
    boltzmann_constant: float,
    gamma: float,
) -> np.ndarray:
    density = primitive[..., 0]
    velocity = primitive[..., 1]
    temperature = primitive[..., 2]
    gas_constant = boltzmann_constant / particle_mass
    conserved = np.empty_like(primitive)
    conserved[..., 0] = density
    conserved[..., 1] = density * velocity
    conserved[..., 2] = (
        density * gas_constant * temperature / (gamma - 1.0)
        + 0.5 * density * velocity**2
    )
    return conserved


def _conserved_to_primitive(
    conserved: np.ndarray,
    particle_mass: float,
    boltzmann_constant: float,
    gamma: float,
) -> np.ndarray:
    density = conserved[..., 0]
    velocity = conserved[..., 1] / density
    internal_energy_density = (
        conserved[..., 2] - 0.5 * density * velocity**2
    )
    gas_constant = boltzmann_constant / particle_mass
    temperature = (
        (gamma - 1.0) * internal_energy_density / (density * gas_constant)
    )
    primitive = np.empty_like(conserved)
    primitive[..., 0] = density
    primitive[..., 1] = velocity
    primitive[..., 2] = temperature
    return primitive


def _thermodynamic_fields(
    primitive: np.ndarray,
    particle_mass: float,
    boltzmann_constant: float,
    diameter: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    density = primitive[..., 0]
    temperature = primitive[..., 2]
    number_density = density / particle_mass
    eta = packing_fraction(number_density, diameter)
    compressibility = carnahan_starling_compressibility(eta)
    derivative = carnahan_starling_derivative(eta)
    pressure = number_density * boltzmann_constant * temperature * compressibility
    gas_constant = boltzmann_constant / particle_mass
    sound_speed_squared = gas_constant * temperature * (
        compressibility
        + eta * derivative
        + (gamma - 1.0) * compressibility**2
    )
    return pressure, eta, compressibility, sound_speed_squared


def _physical_flux(
    primitive: np.ndarray,
    particle_mass: float,
    boltzmann_constant: float,
    diameter: float,
    gamma: float,
) -> np.ndarray:
    conserved = _primitive_to_conserved(
        primitive, particle_mass, boltzmann_constant, gamma
    )
    pressure, _, _, _ = _thermodynamic_fields(
        primitive,
        particle_mass,
        boltzmann_constant,
        diameter,
        gamma,
    )
    density = primitive[..., 0]
    velocity = primitive[..., 1]
    flux = np.empty_like(primitive)
    flux[..., 0] = density * velocity
    flux[..., 1] = density * velocity**2 + pressure
    flux[..., 2] = velocity * (conserved[..., 2] + pressure)
    return flux


def _minmod(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    same_sign = ((a > 0.0) & (b > 0.0) & (c > 0.0)) | (
        (a < 0.0) & (b < 0.0) & (c < 0.0)
    )
    magnitude = np.minimum(np.minimum(np.abs(a), np.abs(b)), np.abs(c))
    return np.where(same_sign, np.sign(a) * magnitude, 0.0)


def _reflecting_ghost_primitives(primitive: np.ndarray) -> np.ndarray:
    cells = primitive.shape[0]
    extended = np.empty((cells + 4, 3), dtype=np.float64)
    extended[2:-2] = primitive
    extended[1] = primitive[0]
    extended[0] = primitive[1]
    extended[1, 1] *= -1.0
    extended[0, 1] *= -1.0
    extended[-2] = primitive[-1]
    extended[-1] = primitive[-2]
    extended[-2, 1] *= -1.0
    extended[-1, 1] *= -1.0
    return extended


def _rusanov_flux(
    left: np.ndarray,
    right: np.ndarray,
    particle_mass: float,
    boltzmann_constant: float,
    diameter: float,
    gamma: float,
) -> np.ndarray:
    left_conserved = _primitive_to_conserved(
        left, particle_mass, boltzmann_constant, gamma
    )
    right_conserved = _primitive_to_conserved(
        right, particle_mass, boltzmann_constant, gamma
    )
    left_flux = _physical_flux(
        left, particle_mass, boltzmann_constant, diameter, gamma
    )
    right_flux = _physical_flux(
        right, particle_mass, boltzmann_constant, diameter, gamma
    )
    *_, left_sound_squared = _thermodynamic_fields(
        left, particle_mass, boltzmann_constant, diameter, gamma
    )
    *_, right_sound_squared = _thermodynamic_fields(
        right, particle_mass, boltzmann_constant, diameter, gamma
    )
    left_speed = np.abs(left[:, 1]) + np.sqrt(left_sound_squared)
    right_speed = np.abs(right[:, 1]) + np.sqrt(right_sound_squared)
    maximum_speed = np.maximum(left_speed, right_speed)
    return 0.5 * (left_flux + right_flux) - 0.5 * maximum_speed[:, None] * (
        right_conserved - left_conserved
    )


def _spatial_operator(
    conserved: np.ndarray,
    dx: float,
    parameters: EnskogEulerParameters,
    particle_mass: float,
    boltzmann_constant: float,
    diameter: float,
    density_floor: float,
    temperature_floor: float,
) -> np.ndarray:
    primitive = _conserved_to_primitive(
        conserved,
        particle_mass,
        boltzmann_constant,
        parameters.gamma,
    )
    extended = _reflecting_ghost_primitives(primitive)
    slope = np.zeros_like(extended)
    backward = extended[1:-1] - extended[:-2]
    centered = 0.5 * (extended[2:] - extended[:-2])
    forward = extended[2:] - extended[1:-1]
    slope[1:-1] = _minmod(
        parameters.limiter_theta * backward,
        centered,
        parameters.limiter_theta * forward,
    )
    left = extended[1:-2] + 0.5 * slope[1:-2]
    right = extended[2:-1] - 0.5 * slope[2:-1]
    left[:, 0] = np.maximum(left[:, 0], density_floor)
    right[:, 0] = np.maximum(right[:, 0], density_floor)
    left[:, 2] = np.maximum(left[:, 2], temperature_floor)
    right[:, 2] = np.maximum(right[:, 2], temperature_floor)
    interface_flux = _rusanov_flux(
        left,
        right,
        particle_mass,
        boltzmann_constant,
        diameter,
        parameters.gamma,
    )
    return -(interface_flux[1:] - interface_flux[:-1]) / dx


def _state_is_physical(
    conserved: np.ndarray,
    parameters: EnskogEulerParameters,
    particle_mass: float,
    boltzmann_constant: float,
    diameter: float,
    density_floor: float,
    temperature_floor: float,
) -> bool:
    if not np.all(np.isfinite(conserved)):
        return False
    primitive = _conserved_to_primitive(
        conserved,
        particle_mass,
        boltzmann_constant,
        parameters.gamma,
    )
    if not (
        np.all(np.isfinite(primitive))
        and np.all(primitive[:, 0] > density_floor)
        and np.all(primitive[:, 2] > temperature_floor)
    ):
        return False
    eta = packing_fraction(primitive[:, 0] / particle_mass, diameter)
    return bool(np.all(eta < parameters.maximum_packing_fraction))


def _maximum_signal_speed(
    conserved: np.ndarray,
    parameters: EnskogEulerParameters,
    particle_mass: float,
    boltzmann_constant: float,
    diameter: float,
) -> float:
    primitive = _conserved_to_primitive(
        conserved,
        particle_mass,
        boltzmann_constant,
        parameters.gamma,
    )
    *_, sound_speed_squared = _thermodynamic_fields(
        primitive,
        particle_mass,
        boltzmann_constant,
        diameter,
        parameters.gamma,
    )
    return float(np.max(np.abs(primitive[:, 1]) + np.sqrt(sound_speed_squared)))


def _advance_one_step(
    conserved: np.ndarray,
    dt: float,
    dx: float,
    parameters: EnskogEulerParameters,
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
    first = conserved + dt * _spatial_operator(conserved, *arguments)
    if not _state_is_physical(
        first,
        parameters,
        particle_mass,
        boltzmann_constant,
        diameter,
        density_floor,
        temperature_floor,
    ):
        return None
    second = 0.5 * conserved + 0.5 * (
        first + dt * _spatial_operator(first, *arguments)
    )
    if not _state_is_physical(
        second,
        parameters,
        particle_mass,
        boltzmann_constant,
        diameter,
        density_floor,
        temperature_floor,
    ):
        return None
    return second


def _load_initial_state(
    config_path: Path,
    parameters: EnskogEulerParameters,
    particle_count_override: int | None,
    particle_diameter_override: float | None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    extent = [float(value) for value in config["domain"]["extent"]]
    if len(extent) != 3 or any(value <= 0.0 for value in extent):
        raise ValueError("domain.extent must contain three positive values")
    if config["domain"]["boundary"]["x"] != "reflecting":
        raise ValueError("Enskog-Euler reference requires a reflecting x boundary")
    particle_count = (
        int(particle_count_override)
        if particle_count_override is not None
        else int(config["particles"]["count"])
    )
    diameter = (
        float(particle_diameter_override)
        if particle_diameter_override is not None
        else float(config["particles"]["diameter"])
    )
    particle_mass = float(config["units"]["particle_mass"])
    boltzmann_constant = float(config["units"]["boltzmann_constant"])
    if particle_count <= 0:
        raise ValueError("particle count must be positive")
    if particle_mass <= 0.0 or boltzmann_constant <= 0.0:
        raise ValueError("particle mass and Boltzmann constant must be positive")
    if diameter < 0.0 or not math.isfinite(diameter):
        raise ValueError("particle diameter must be finite and nonnegative")

    initial = config["initial_condition"]
    interface_x = float(initial["interface_x"])
    transition_width = float(initial["transition_width"])
    if transition_width <= 0.0:
        raise ValueError("initial transition_width must be positive")
    left = initial["left"]
    right = initial["right"]
    dx = extent[0] / parameters.cells
    x = (np.arange(parameters.cells, dtype=np.float64) + 0.5) * dx
    blend = 0.5 * (1.0 + np.tanh((x - interface_x) / transition_width))
    relative_number_density = (
        (1.0 - blend) * float(left["relative_number_density"])
        + blend * float(right["relative_number_density"])
    )
    transverse_area = extent[1] * extent[2]
    number_density_scale = particle_count / (
        transverse_area * dx * float(np.sum(relative_number_density))
    )
    number_density = number_density_scale * relative_number_density
    density = particle_mass * number_density
    temperature = (1.0 - blend) * float(left["temperature"]) + blend * float(
        right["temperature"]
    )
    velocity = (
        (1.0 - blend) * float(left["bulk_velocity"][0])
        + blend * float(right["bulk_velocity"][0])
    )
    primitive = np.column_stack((density, velocity, temperature))
    conserved = _primitive_to_conserved(
        primitive, particle_mass, boltzmann_constant, parameters.gamma
    )
    eta = packing_fraction(number_density, diameter)
    if np.max(eta) >= parameters.maximum_packing_fraction:
        raise ValueError(
            "initial packing fraction exceeds Enskog-Euler safety limit"
        )
    return (
        x,
        conserved,
        dx,
        transverse_area,
        particle_mass,
        boltzmann_constant,
        diameter,
    )


def _make_frame(
    frame: int,
    time: float,
    x: np.ndarray,
    conserved: np.ndarray,
    parameters: EnskogEulerParameters,
    particle_mass: float,
    boltzmann_constant: float,
    diameter: float,
) -> EulerFrame:
    primitive = _conserved_to_primitive(
        conserved,
        particle_mass,
        boltzmann_constant,
        parameters.gamma,
    )
    mass_density = primitive[:, 0].copy()
    velocity = primitive[:, 1].copy()
    temperature = primitive[:, 2].copy()
    number_density = mass_density / particle_mass
    pressure, eta, compressibility, _ = _thermodynamic_fields(
        primitive,
        particle_mass,
        boltzmann_constant,
        diameter,
        parameters.gamma,
    )
    kinetic_pressure = number_density * boltzmann_constant * temperature
    return EulerFrame(
        frame=frame,
        time=time,
        fields={
            "x_center": x.copy(),
            "number_density": number_density,
            "mass_density": mass_density,
            "bulk_velocity_x": velocity,
            "temperature": temperature,
            "pressure": pressure,
            "kinetic_pressure": kinetic_pressure,
            "packing_fraction": eta,
            "compressibility_factor": compressibility,
        },
    )


def solve_enskog_euler_reference(
    config_path: Path,
    frame_times: list[tuple[int, float]],
    *,
    particle_count: int | None = None,
    particle_diameter: float | None = None,
    parameters: EnskogEulerParameters = EnskogEulerParameters(),
) -> EnskogEulerSolution:
    """Solve inviscid hard-sphere hydrodynamics at exact output times."""
    if parameters.gamma <= 1.0:
        raise ValueError("gamma must be greater than one")
    if parameters.cells < 8:
        raise ValueError("cell count must be at least eight")
    if not 0.0 < parameters.cfl < 1.0:
        raise ValueError("CFL must lie strictly between zero and one")
    if not 1.0 <= parameters.limiter_theta <= 2.0:
        raise ValueError("limiter theta must lie in [1, 2]")
    if not 0.0 < parameters.maximum_packing_fraction < 1.0:
        raise ValueError("maximum packing fraction must lie in (0, 1)")
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
        parameters,
        particle_count,
        particle_diameter,
    )
    initial_primitive = _conserved_to_primitive(
        conserved,
        particle_mass,
        boltzmann_constant,
        parameters.gamma,
    )
    density_floor = max(
        np.finfo(np.float64).tiny,
        1.0e-12 * float(np.min(initial_primitive[:, 0])),
    )
    temperature_floor = max(
        np.finfo(np.float64).tiny,
        1.0e-12 * float(np.min(initial_primitive[:, 2])),
    )
    initial_mass = float(np.sum(conserved[:, 0]) * dx * transverse_area)
    initial_energy = float(np.sum(conserved[:, 2]) * dx * transverse_area)
    minimum_density = float(np.min(initial_primitive[:, 0]))
    minimum_temperature = float(np.min(initial_primitive[:, 2]))
    maximum_eta = float(
        np.max(packing_fraction(initial_primitive[:, 0] / particle_mass, diameter))
    )
    current_time = 0.0
    steps = 0
    frames: list[EulerFrame] = []

    for frame, target_time in ordered_targets:
        if target_time < current_time - 1.0e-14:
            raise ValueError("output times moved backward")
        while current_time < target_time - 2.0e-15:
            maximum_speed = _maximum_signal_speed(
                conserved,
                parameters,
                particle_mass,
                boltzmann_constant,
                diameter,
            )
            if not math.isfinite(maximum_speed) or maximum_speed <= 0.0:
                raise RuntimeError("invalid Enskog-Euler signal speed")
            dt = min(parameters.cfl * dx / maximum_speed, target_time - current_time)
            accepted: np.ndarray | None = None
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
                dt *= 0.5
            if accepted is None:
                raise RuntimeError(
                    f"Enskog-Euler positivity retries exhausted at t={current_time:g}"
                )
            conserved = accepted
            current_time += dt
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
                parameters,
                particle_mass,
                boltzmann_constant,
                diameter,
            )
        )

    final_mass = float(np.sum(conserved[:, 0]) * dx * transverse_area)
    final_energy = float(np.sum(conserved[:, 2]) * dx * transverse_area)
    return EnskogEulerSolution(
        frames=frames,
        parameters=parameters,
        particle_diameter=diameter,
        steps=steps,
        initial_mass=initial_mass,
        final_mass=final_mass,
        initial_energy=initial_energy,
        final_energy=final_energy,
        minimum_density=minimum_density,
        minimum_temperature=minimum_temperature,
        maximum_packing_fraction=maximum_eta,
    )


def run_self_test(config_path: Path) -> int:
    eta = np.array([0.0, 0.01, 0.03], dtype=np.float64)
    z = carnahan_starling_compressibility(eta)
    chi = contact_pair_correlation(eta)
    if not np.allclose(z, 1.0 + 4.0 * eta * chi, rtol=1.0e-14, atol=1.0e-14):
        raise AssertionError("Carnahan--Starling/Enskog identity failed")
    step = 1.0e-7
    numeric = (
        carnahan_starling_compressibility(eta + step)
        - carnahan_starling_compressibility(eta)
    ) / step
    analytic = carnahan_starling_derivative(eta)
    if not np.allclose(numeric, analytic, rtol=2.0e-6, atol=2.0e-6):
        raise AssertionError("Carnahan--Starling derivative self-test failed")

    dilute = solve_enskog_euler_reference(
        config_path,
        [(0, 0.0), (1, 0.05)],
        particle_diameter=0.0,
        parameters=EnskogEulerParameters(cells=256),
    )
    dense = solve_enskog_euler_reference(
        config_path,
        [(0, 0.0), (1, 0.05), (2, 0.2)],
        parameters=EnskogEulerParameters(cells=512),
    )
    for solution in (dilute, dense):
        if solution.minimum_density <= 0.0 or solution.minimum_temperature <= 0.0:
            raise AssertionError("Enskog-Euler self-test lost positivity")
        if abs(solution.relative_mass_drift) > 1.0e-11:
            raise AssertionError("Enskog-Euler self-test lost mass conservation")
        if abs(solution.relative_energy_drift) > 1.0e-11:
            raise AssertionError("Enskog-Euler self-test lost energy conservation")
    print("Enskog-Euler reference self-test")
    print(f"  CS Z(eta=0.03): {z[-1]:.12g}")
    print(f"  dense shock-tube steps: {dense.steps}")
    print(f"  relative mass drift: {dense.relative_mass_drift:.3e}")
    print(f"  relative energy drift: {dense.relative_energy_drift:.3e}")
    print(f"  maximum packing fraction: {dense.maximum_packing_fraction:.9g}")
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
        help="validate the EOS, sound-speed derivative, and finite-volume solver",
    )
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("use --self-test or import this module from an analyzer")
    return run_self_test(arguments.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
