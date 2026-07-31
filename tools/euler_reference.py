#!/usr/bin/env python3
"""One-dimensional ideal-gas Euler reference for the shock-tube experiment.

This module is deliberately independent of the hard-sphere time integrator.
It reconstructs the configured smooth macroscopic initial condition and
advances it with a conservative finite-volume method:

* monatomic ideal gas with gamma = 5/3 by default;
* MUSCL primitive-variable reconstruction with a minmod limiter;
* local Lax--Friedrichs (Rusanov) interface flux;
* second-order SSP Runge--Kutta time integration;
* reflecting boundaries on the longitudinal axis.

The result is a numerical continuum reference, not an analytic truth and not
an input to particle advancement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EulerParameters:
    gamma: float = 5.0 / 3.0
    cells: int = 2048
    cfl: float = 0.45
    limiter_theta: float = 1.5


@dataclass
class EulerFrame:
    frame: int
    time: float
    fields: dict[str, np.ndarray]


@dataclass
class EulerSolution:
    frames: list[EulerFrame]
    parameters: EulerParameters
    steps: int
    initial_mass: float
    final_mass: float
    initial_energy: float
    final_energy: float
    minimum_density: float
    minimum_pressure: float

    @property
    def relative_mass_drift(self) -> float:
        return _relative_difference(self.final_mass, self.initial_mass)

    @property
    def relative_energy_drift(self) -> float:
        return _relative_difference(self.final_energy, self.initial_energy)


def _relative_difference(value: float, reference: float) -> float:
    scale = max(abs(reference), np.finfo(np.float64).tiny)
    return (value - reference) / scale


def _primitive_to_conserved(
    primitive: np.ndarray, gamma: float
) -> np.ndarray:
    density = primitive[..., 0]
    velocity = primitive[..., 1]
    pressure = primitive[..., 2]
    conserved = np.empty_like(primitive)
    conserved[..., 0] = density
    conserved[..., 1] = density * velocity
    conserved[..., 2] = pressure / (gamma - 1.0) + 0.5 * density * velocity**2
    return conserved


def _conserved_to_primitive(
    conserved: np.ndarray, gamma: float
) -> np.ndarray:
    density = conserved[..., 0]
    velocity = conserved[..., 1] / density
    pressure = (gamma - 1.0) * (
        conserved[..., 2] - 0.5 * density * velocity**2
    )
    primitive = np.empty_like(conserved)
    primitive[..., 0] = density
    primitive[..., 1] = velocity
    primitive[..., 2] = pressure
    return primitive


def _physical_flux(primitive: np.ndarray, gamma: float) -> np.ndarray:
    conserved = _primitive_to_conserved(primitive, gamma)
    density = primitive[..., 0]
    velocity = primitive[..., 1]
    pressure = primitive[..., 2]
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
    left: np.ndarray, right: np.ndarray, gamma: float
) -> np.ndarray:
    left_conserved = _primitive_to_conserved(left, gamma)
    right_conserved = _primitive_to_conserved(right, gamma)
    left_flux = _physical_flux(left, gamma)
    right_flux = _physical_flux(right, gamma)
    left_speed = np.abs(left[:, 1]) + np.sqrt(
        gamma * left[:, 2] / left[:, 0]
    )
    right_speed = np.abs(right[:, 1]) + np.sqrt(
        gamma * right[:, 2] / right[:, 0]
    )
    maximum_speed = np.maximum(left_speed, right_speed)
    return 0.5 * (left_flux + right_flux) - 0.5 * maximum_speed[:, None] * (
        right_conserved - left_conserved
    )


def _spatial_operator(
    conserved: np.ndarray,
    dx: float,
    parameters: EulerParameters,
    density_floor: float,
    pressure_floor: float,
) -> np.ndarray:
    primitive = _conserved_to_primitive(conserved, parameters.gamma)
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
    left[:, 2] = np.maximum(left[:, 2], pressure_floor)
    right[:, 2] = np.maximum(right[:, 2], pressure_floor)

    interface_flux = _rusanov_flux(left, right, parameters.gamma)
    return -(interface_flux[1:] - interface_flux[:-1]) / dx


def _state_is_physical(
    conserved: np.ndarray,
    gamma: float,
    density_floor: float,
    pressure_floor: float,
) -> bool:
    if not np.all(np.isfinite(conserved)):
        return False
    primitive = _conserved_to_primitive(conserved, gamma)
    return bool(
        np.all(np.isfinite(primitive))
        and np.all(primitive[:, 0] > density_floor)
        and np.all(primitive[:, 2] > pressure_floor)
    )


def _maximum_signal_speed(conserved: np.ndarray, gamma: float) -> float:
    primitive = _conserved_to_primitive(conserved, gamma)
    sound_speed = np.sqrt(gamma * primitive[:, 2] / primitive[:, 0])
    return float(np.max(np.abs(primitive[:, 1]) + sound_speed))


def _advance_one_step(
    conserved: np.ndarray,
    dt: float,
    dx: float,
    parameters: EulerParameters,
    density_floor: float,
    pressure_floor: float,
) -> np.ndarray | None:
    first = conserved + dt * _spatial_operator(
        conserved,
        dx,
        parameters,
        density_floor,
        pressure_floor,
    )
    if not _state_is_physical(
        first, parameters.gamma, density_floor, pressure_floor
    ):
        return None
    second = 0.5 * conserved + 0.5 * (
        first
        + dt
        * _spatial_operator(
            first,
            dx,
            parameters,
            density_floor,
            pressure_floor,
        )
    )
    if not _state_is_physical(
        second, parameters.gamma, density_floor, pressure_floor
    ):
        return None
    return second


def _load_initial_state(
    config_path: Path,
    parameters: EulerParameters,
    particle_count_override: int | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
]:
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)

    extent = [float(value) for value in config["domain"]["extent"]]
    if len(extent) != 3 or any(value <= 0.0 for value in extent):
        raise ValueError("domain.extent must contain three positive values")
    if config["domain"]["boundary"]["x"] != "reflecting":
        raise ValueError("Euler reference currently requires a reflecting x boundary")

    particle_count = (
        int(particle_count_override)
        if particle_count_override is not None
        else int(config["particles"]["count"])
    )
    if particle_count <= 0:
        raise ValueError("particle count must be positive")
    particle_mass = float(config["units"]["particle_mass"])
    boltzmann_constant = float(config["units"]["boltzmann_constant"])
    if particle_mass <= 0.0 or boltzmann_constant <= 0.0:
        raise ValueError("particle mass and Boltzmann constant must be positive")

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
    left_velocity = float(left["bulk_velocity"][0])
    right_velocity = float(right["bulk_velocity"][0])
    velocity = (1.0 - blend) * left_velocity + blend * right_velocity
    pressure = number_density * boltzmann_constant * temperature

    primitive = np.column_stack((density, velocity, pressure))
    conserved = _primitive_to_conserved(primitive, parameters.gamma)
    return (
        x,
        conserved,
        dx,
        transverse_area,
        particle_mass,
        boltzmann_constant,
        number_density_scale,
    )


def _make_frame(
    frame: int,
    time: float,
    x: np.ndarray,
    conserved: np.ndarray,
    gamma: float,
    particle_mass: float,
    boltzmann_constant: float,
) -> EulerFrame:
    primitive = _conserved_to_primitive(conserved, gamma)
    mass_density = primitive[:, 0].copy()
    velocity = primitive[:, 1].copy()
    pressure = primitive[:, 2].copy()
    number_density = mass_density / particle_mass
    temperature = pressure / (number_density * boltzmann_constant)
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
        },
    )


def solve_euler_reference(
    config_path: Path,
    frame_times: list[tuple[int, float]],
    *,
    particle_count: int | None = None,
    parameters: EulerParameters = EulerParameters(),
) -> EulerSolution:
    """Solve the configured Euler problem and return states at exact targets."""
    if parameters.gamma <= 1.0:
        raise ValueError("Euler gamma must be greater than one")
    if parameters.cells < 8:
        raise ValueError("Euler cell count must be at least eight")
    if not 0.0 < parameters.cfl < 1.0:
        raise ValueError("Euler CFL must lie strictly between zero and one")
    if not 1.0 <= parameters.limiter_theta <= 2.0:
        raise ValueError("limiter theta must lie in [1, 2]")
    if not frame_times:
        raise ValueError("at least one Euler output time is required")

    ordered_targets = sorted(
        ((int(frame), float(time)) for frame, time in frame_times),
        key=lambda item: (item[1], item[0]),
    )
    if ordered_targets[0][1] < 0.0:
        raise ValueError("Euler output times must be nonnegative")
    if any(
        ordered_targets[index][1] < ordered_targets[index - 1][1]
        for index in range(1, len(ordered_targets))
    ):
        raise ValueError("Euler output times must be nondecreasing")

    (
        x,
        conserved,
        dx,
        transverse_area,
        particle_mass,
        boltzmann_constant,
        _,
    ) = _load_initial_state(config_path, parameters, particle_count)
    initial_primitive = _conserved_to_primitive(conserved, parameters.gamma)
    density_floor = max(
        np.finfo(np.float64).tiny,
        1.0e-12 * float(np.min(initial_primitive[:, 0])),
    )
    pressure_floor = max(
        np.finfo(np.float64).tiny,
        1.0e-12 * float(np.min(initial_primitive[:, 2])),
    )

    initial_mass = float(np.sum(conserved[:, 0]) * dx * transverse_area)
    initial_energy = float(np.sum(conserved[:, 2]) * dx * transverse_area)
    minimum_density = float(np.min(initial_primitive[:, 0]))
    minimum_pressure = float(np.min(initial_primitive[:, 2]))
    current_time = 0.0
    steps = 0
    frames: list[EulerFrame] = []

    for frame, target_time in ordered_targets:
        if target_time < current_time - 1.0e-14:
            raise ValueError("Euler target times moved backward")
        while current_time < target_time - 2.0e-15:
            maximum_speed = _maximum_signal_speed(conserved, parameters.gamma)
            if not math.isfinite(maximum_speed) or maximum_speed <= 0.0:
                raise RuntimeError("Euler solver encountered an invalid signal speed")
            nominal_dt = parameters.cfl * dx / maximum_speed
            dt = min(nominal_dt, target_time - current_time)
            accepted: np.ndarray | None = None
            for _ in range(20):
                accepted = _advance_one_step(
                    conserved,
                    dt,
                    dx,
                    parameters,
                    density_floor,
                    pressure_floor,
                )
                if accepted is not None:
                    break
                dt *= 0.5
            if accepted is None:
                raise RuntimeError(
                    f"Euler positivity retries exhausted at t={current_time:g}"
                )
            conserved = accepted
            current_time += dt
            steps += 1
        current_time = target_time
        primitive = _conserved_to_primitive(conserved, parameters.gamma)
        minimum_density = min(minimum_density, float(np.min(primitive[:, 0])))
        minimum_pressure = min(minimum_pressure, float(np.min(primitive[:, 2])))
        frames.append(
            _make_frame(
                frame,
                target_time,
                x,
                conserved,
                parameters.gamma,
                particle_mass,
                boltzmann_constant,
            )
        )

    final_mass = float(np.sum(conserved[:, 0]) * dx * transverse_area)
    final_energy = float(np.sum(conserved[:, 2]) * dx * transverse_area)
    return EulerSolution(
        frames=frames,
        parameters=parameters,
        steps=steps,
        initial_mass=initial_mass,
        final_mass=final_mass,
        initial_energy=initial_energy,
        final_energy=final_energy,
        minimum_density=minimum_density,
        minimum_pressure=minimum_pressure,
    )


def write_euler_csv(solution: EulerSolution, output_path: Path) -> None:
    fields = (
        "frame",
        "time",
        "cell",
        "x_center",
        "number_density",
        "mass_density",
        "bulk_velocity_x",
        "temperature",
        "pressure",
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame in solution.frames:
            count = len(frame.fields["x_center"])
            for cell in range(count):
                writer.writerow(
                    {
                        "frame": frame.frame,
                        "time": f"{frame.time:.17g}",
                        "cell": cell,
                        "x_center": f"{frame.fields['x_center'][cell]:.17g}",
                        "number_density": (
                            f"{frame.fields['number_density'][cell]:.17g}"
                        ),
                        "mass_density": (
                            f"{frame.fields['mass_density'][cell]:.17g}"
                        ),
                        "bulk_velocity_x": (
                            f"{frame.fields['bulk_velocity_x'][cell]:.17g}"
                        ),
                        "temperature": (
                            f"{frame.fields['temperature'][cell]:.17g}"
                        ),
                        "pressure": f"{frame.fields['pressure'][cell]:.17g}",
                    }
                )


def _run_self_test(config_path: Path) -> None:
    parameters = EulerParameters(cells=512)
    uniform = np.column_stack(
        (
            np.full(parameters.cells, 2.0),
            np.zeros(parameters.cells),
            np.full(parameters.cells, 3.0),
        )
    )
    conserved = _primitive_to_conserved(uniform, parameters.gamma)
    density_floor = 1.0e-12
    pressure_floor = 1.0e-12
    dx = 4.0 / parameters.cells
    current = conserved.copy()
    for _ in range(20):
        signal_speed = _maximum_signal_speed(current, parameters.gamma)
        dt = 0.4 * dx / signal_speed
        advanced = _advance_one_step(
            current,
            dt,
            dx,
            parameters,
            density_floor,
            pressure_floor,
        )
        if advanced is None:
            raise AssertionError("uniform-state self-test lost positivity")
        current = advanced
    uniform_error = float(np.max(np.abs(current - conserved)))
    if uniform_error > 1.0e-12:
        raise AssertionError(
            f"uniform reflecting state drifted by {uniform_error:.3e}"
        )

    shock = solve_euler_reference(
        config_path,
        [(0, 0.0), (1, 0.05), (2, 0.2)],
        parameters=parameters,
    )
    if shock.minimum_density <= 0.0 or shock.minimum_pressure <= 0.0:
        raise AssertionError("shock-tube self-test lost positivity")
    if abs(shock.relative_mass_drift) > 1.0e-11:
        raise AssertionError(
            f"shock-tube mass drift is {shock.relative_mass_drift:.3e}"
        )
    if abs(shock.relative_energy_drift) > 1.0e-11:
        raise AssertionError(
            f"shock-tube energy drift is {shock.relative_energy_drift:.3e}"
        )
    print("Euler reference self-test")
    print(f"  uniform-state maximum drift: {uniform_error:.3e}")
    print(f"  shock-tube steps: {shock.steps}")
    print(f"  relative mass drift: {shock.relative_mass_drift:.3e}")
    print(f"  relative energy drift: {shock.relative_energy_drift:.3e}")
    print(f"  minimum density: {shock.minimum_density:.9g}")
    print(f"  minimum pressure: {shock.minimum_pressure:.9g}")
    print("  status: passed")


def _run_grid_check(config_path: Path) -> None:
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    length_x = float(config["domain"]["extent"][0])
    sample_count = int(config["diagnostics"]["macroscopic_x_slabs"])
    sample_x = (
        np.arange(sample_count, dtype=np.float64) + 0.5
    ) * length_x / sample_count
    grids = (1024, 2048, 4096)
    solutions = {
        cells: solve_euler_reference(
            config_path,
            [(0, 0.2)],
            parameters=EulerParameters(cells=cells),
        )
        for cells in grids
    }
    reference = solutions[4096].frames[-1]
    fields = (
        "number_density",
        "bulk_velocity_x",
        "temperature",
        "pressure",
    )
    print("Euler reference grid check at t=0.2")
    print("  relative L2 differences against 4096 cells:")
    for cells in (1024, 2048):
        frame = solutions[cells].frames[-1]
        errors: list[str] = []
        for field in fields:
            values = np.interp(
                sample_x,
                frame.fields["x_center"],
                frame.fields[field],
            )
            reference_values = np.interp(
                sample_x,
                reference.fields["x_center"],
                reference.fields[field],
            )
            rms_difference = float(
                np.sqrt(np.mean((values - reference_values) ** 2))
            )
            reference_l2 = float(np.sqrt(np.mean(reference_values**2)))
            relative_l2 = rms_difference / max(
                reference_l2, np.finfo(np.float64).tiny
            )
            errors.append(f"{field}={relative_l2:.6g}")
        print(f"    {cells}: " + ", ".join(errors))
    print(
        "  steps: "
        + ", ".join(
            f"{cells}={solutions[cells].steps}" for cells in grids
        )
    )
    print(
        f"  4096-cell relative mass/energy drift: "
        f"{solutions[4096].relative_mass_drift:.3e} / "
        f"{solutions[4096].relative_energy_drift:.3e}"
    )
    print("  status: passed")


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
        help="run uniform-state and configured shock-tube checks",
    )
    parser.add_argument(
        "--grid-check",
        action="store_true",
        help="compare 1024/2048-cell results against 4096 cells",
    )
    args = parser.parse_args()
    if not args.self_test and not args.grid_check:
        parser.error(
            "use --self-test or --grid-check, "
            "or import this module from the renderer"
        )
    config_path = args.config.resolve()
    if args.self_test:
        _run_self_test(config_path)
    if args.grid_check:
        _run_grid_check(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
