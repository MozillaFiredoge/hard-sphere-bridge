#!/usr/bin/env python3
"""Prepare and run a factorial particle-count/Knudsen shock-tube scan."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_multiseed_shock_tube import (
    deep_merge,
    inspect_run,
    load_json_object,
    require_integer,
    require_number,
    write_json_if_compatible,
)


@dataclass(frozen=True)
class ScanConfig:
    path: Path
    raw: dict[str, Any]
    base_config_path: Path
    base_config: dict[str, Any]
    particle_counts: tuple[int, ...]
    knudsen_numbers: tuple[float, ...]
    seeds: tuple[int, ...]
    required_backend: str
    reference_particle_count: int
    characteristic_length: float
    density_reference: str
    overrides: dict[str, Any]
    statistics: dict[str, Any]
    acceptance: dict[str, Any]
    maximum_initial_packing_fraction: float
    maximum_transverse_mean_free_path_ratio: float


@dataclass(frozen=True)
class ScanCell:
    cell_id: str
    particle_count: int
    target_knudsen: float
    particle_diameter: float
    extent_y: float
    extent_z: float
    transverse_area: float
    reference_number_density: float
    maximum_number_density: float
    mean_free_path: float
    realized_knudsen: float
    maximum_initial_packing_fraction: float
    transverse_mean_free_path_ratio: float
    overrides: dict[str, Any]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        help="shock_tube_n_kn_scan experiment JSON",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        help="particles_cli executable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="scan output directory",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write all derived cell and seed configs without simulations",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        help="process only the first N cells in row-major order",
    )
    parser.add_argument(
        "--cell",
        action="append",
        dest="cell_ids",
        help="process one cell id; repeat to select multiple cells",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        help="process only the first N seeds in every selected cell",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate the independent N/Kn parameterization and exit",
    )
    return parser.parse_args()


def require_unique_integers(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{name} must contain at least two integers")
    result = tuple(
        require_integer(item, f"{name}[{index}]", minimum=1)
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} values must be unique")
    return result


def require_unique_numbers(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{name} must contain at least two numbers")
    result = tuple(
        require_number(item, f"{name}[{index}]", positive=True)
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{name} values must be unique")
    return result


def nested_value(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def load_scan_config(path: Path) -> ScanConfig:
    resolved = path.resolve()
    raw = load_json_object(resolved)
    if raw.get("experiment") != "shock_tube_n_kn_scan":
        raise ValueError("experiment must be shock_tube_n_kn_scan")
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")

    base_name = raw.get("base_config")
    if not isinstance(base_name, str) or not base_name:
        raise ValueError("base_config must be a non-empty path string")
    base_path = (resolved.parent / base_name).resolve()
    base = load_json_object(base_path)
    if base.get("experiment") != "shock_tube":
        raise ValueError("base_config must be a shock_tube config")

    particle_counts = require_unique_integers(
        raw.get("particle_counts"),
        "particle_counts",
    )
    knudsen_numbers = require_unique_numbers(
        raw.get("knudsen_numbers"),
        "knudsen_numbers",
    )
    seeds = require_unique_integers(raw.get("seeds"), "seeds")
    required_backend = raw.get("required_backend")
    if required_backend not in {"any", "cuda"}:
        raise ValueError("required_backend must be any or cuda")

    parameterization = raw.get("parameterization")
    overrides = raw.get("overrides")
    statistics = raw.get("statistics")
    acceptance = raw.get("acceptance")
    if not all(
        isinstance(value, dict)
        for value in (
            parameterization,
            overrides,
            statistics,
            acceptance,
        )
    ):
        raise ValueError(
            "parameterization, overrides, statistics, and acceptance "
            "must be objects"
        )
    assert isinstance(parameterization, dict)
    assert isinstance(overrides, dict)
    assert isinstance(statistics, dict)
    assert isinstance(acceptance, dict)

    if nested_value(overrides, "particles", "count") is not None:
        raise ValueError("overrides must not set particles.count")
    if nested_value(overrides, "particles", "diameter") is not None:
        raise ValueError("overrides must not set particles.diameter")
    if nested_value(overrides, "domain", "extent") is not None:
        raise ValueError("overrides must not set domain.extent")
    if parameterization.get("scale_transverse_area_with_particle_count") is not True:
        raise ValueError(
            "scale_transverse_area_with_particle_count must be true"
        )
    density_reference = parameterization.get("density_reference")
    if density_reference not in {"left", "right"}:
        raise ValueError("density_reference must be left or right")

    confidence = require_number(
        statistics.get("confidence_level"),
        "statistics.confidence_level",
        positive=True,
    )
    if confidence != 0.95:
        raise ValueError("only a 0.95 confidence level is supported")
    require_integer(
        statistics.get("euler_cells"),
        "statistics.euler_cells",
        minimum=8,
    )
    minimum_completed = require_integer(
        statistics.get("minimum_completed_seeds"),
        "statistics.minimum_completed_seeds",
        minimum=2,
    )
    if minimum_completed > len(seeds):
        raise ValueError("minimum_completed_seeds exceeds configured seeds")
    require_number(
        acceptance.get("maximum_absolute_relative_energy_drift"),
        "acceptance.maximum_absolute_relative_energy_drift",
        positive=True,
    )
    require_number(
        acceptance.get("minimum_committed_surface_gap"),
        "acceptance.minimum_committed_surface_gap",
    )

    return ScanConfig(
        path=resolved,
        raw=raw,
        base_config_path=base_path,
        base_config=base,
        particle_counts=particle_counts,
        knudsen_numbers=knudsen_numbers,
        seeds=seeds,
        required_backend=required_backend,
        reference_particle_count=require_integer(
            parameterization.get("reference_particle_count"),
            "parameterization.reference_particle_count",
            minimum=2,
        ),
        characteristic_length=require_number(
            parameterization.get("characteristic_length"),
            "parameterization.characteristic_length",
            positive=True,
        ),
        density_reference=density_reference,
        overrides=copy.deepcopy(overrides),
        statistics=copy.deepcopy(statistics),
        acceptance=copy.deepcopy(acceptance),
        maximum_initial_packing_fraction=require_number(
            acceptance.get("maximum_initial_packing_fraction"),
            "acceptance.maximum_initial_packing_fraction",
            positive=True,
        ),
        maximum_transverse_mean_free_path_ratio=require_number(
            acceptance.get("maximum_transverse_mean_free_path_ratio"),
            "acceptance.maximum_transverse_mean_free_path_ratio",
            positive=True,
        ),
    )


def log_cosh(value: float) -> float:
    magnitude = abs(value)
    return magnitude + math.log1p(math.exp(-2.0 * magnitude)) - math.log(2.0)


def integrated_relative_number_density(config: dict[str, Any]) -> float:
    domain = config["domain"]
    initial = config["initial_condition"]
    assert isinstance(domain, dict)
    assert isinstance(initial, dict)
    extent = domain["extent"]
    left = initial["left"]
    right = initial["right"]
    assert isinstance(extent, list)
    assert isinstance(left, dict)
    assert isinstance(right, dict)
    length = float(extent[0])
    interface = float(initial["interface_x"])
    width = float(initial["transition_width"])
    left_density = float(left["relative_number_density"])
    right_density = float(right["relative_number_density"])
    if width <= 0.0:
        raise ValueError("initial_condition.transition_width must be positive")
    blend_integral = 0.5 * length + 0.5 * width * (
        log_cosh((length - interface) / width)
        - log_cosh(-interface / width)
    )
    result = (
        left_density * length
        + (right_density - left_density) * blend_integral
    )
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("integrated relative number density is invalid")
    return result


def cell_identifier(particle_count: int, knudsen: float) -> str:
    encoded_knudsen = f"{knudsen:.6f}".replace(".", "p")
    return f"n-{particle_count:09d}_kn-{encoded_knudsen}"


def derive_cells(scan: ScanConfig) -> tuple[ScanCell, ...]:
    effective_reference = copy.deepcopy(scan.base_config)
    deep_merge(effective_reference, scan.overrides)
    domain = effective_reference.get("domain")
    initial = effective_reference.get("initial_condition")
    if not isinstance(domain, dict) or not isinstance(initial, dict):
        raise ValueError("effective domain and initial_condition must be objects")
    extent = domain.get("extent")
    if (
        not isinstance(extent, list)
        or len(extent) != 3
        or any(float(value) <= 0.0 for value in extent)
    ):
        raise ValueError("domain.extent must have three positive values")
    left = initial.get("left")
    right = initial.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("initial left and right states must be objects")
    relative = {
        "left": require_number(
            left.get("relative_number_density"),
            "initial_condition.left.relative_number_density",
            positive=True,
        ),
        "right": require_number(
            right.get("relative_number_density"),
            "initial_condition.right.relative_number_density",
            positive=True,
        ),
    }

    base_y = float(extent[1])
    base_z = float(extent[2])
    base_area = base_y * base_z
    relative_integral = integrated_relative_number_density(
        effective_reference
    )
    number_density_scale = scan.reference_particle_count / (
        base_area * relative_integral
    )
    reference_density = (
        number_density_scale * relative[scan.density_reference]
    )
    maximum_density = number_density_scale * max(relative.values())

    cells: list[ScanCell] = []
    identifiers: set[str] = set()
    for particle_count in scan.particle_counts:
        area_scale = particle_count / scan.reference_particle_count
        transverse_scale = math.sqrt(area_scale)
        extent_y = base_y * transverse_scale
        extent_z = base_z * transverse_scale
        area = extent_y * extent_z
        cell_density_scale = particle_count / (
            area * relative_integral
        )
        if not math.isclose(
            cell_density_scale,
            number_density_scale,
            rel_tol=2.0e-14,
            abs_tol=0.0,
        ):
            raise AssertionError("transverse scaling did not preserve density")
        for target_knudsen in scan.knudsen_numbers:
            mean_free_path = (
                target_knudsen * scan.characteristic_length
            )
            diameter = math.sqrt(
                1.0
                / (
                    math.sqrt(2.0)
                    * math.pi
                    * reference_density
                    * mean_free_path
                )
            )
            realized_knudsen = 1.0 / (
                math.sqrt(2.0)
                * math.pi
                * reference_density
                * diameter
                * diameter
                * scan.characteristic_length
            )
            packing_fraction = (
                math.pi / 6.0 * maximum_density * diameter**3
            )
            transverse_ratio = mean_free_path / min(extent_y, extent_z)
            if packing_fraction > scan.maximum_initial_packing_fraction:
                raise ValueError(
                    f"N={particle_count}, Kn={target_knudsen:g}: initial "
                    f"packing fraction {packing_fraction:g} exceeds "
                    f"{scan.maximum_initial_packing_fraction:g}"
                )
            if (
                transverse_ratio
                > scan.maximum_transverse_mean_free_path_ratio
            ):
                raise ValueError(
                    f"N={particle_count}, Kn={target_knudsen:g}: transverse "
                    f"mean-free-path ratio {transverse_ratio:g} exceeds "
                    f"{scan.maximum_transverse_mean_free_path_ratio:g}"
                )
            if diameter * 2.0 >= min(extent_y, extent_z):
                raise ValueError("periodic transverse extent is too small")

            derived_overrides = copy.deepcopy(scan.overrides)
            deep_merge(
                derived_overrides,
                {
                    "domain": {
                        "extent": [float(extent[0]), extent_y, extent_z]
                    },
                    "particles": {
                        "count": particle_count,
                        "diameter": diameter,
                    },
                },
            )
            identifier = cell_identifier(particle_count, target_knudsen)
            if identifier in identifiers:
                raise ValueError("scan cell identifiers are not unique")
            identifiers.add(identifier)
            cells.append(
                ScanCell(
                    cell_id=identifier,
                    particle_count=particle_count,
                    target_knudsen=target_knudsen,
                    particle_diameter=diameter,
                    extent_y=extent_y,
                    extent_z=extent_z,
                    transverse_area=area,
                    reference_number_density=reference_density,
                    maximum_number_density=maximum_density,
                    mean_free_path=mean_free_path,
                    realized_knudsen=realized_knudsen,
                    maximum_initial_packing_fraction=packing_fraction,
                    transverse_mean_free_path_ratio=transverse_ratio,
                    overrides=derived_overrides,
                )
            )
    return tuple(cells)


def batch_document(scan: ScanConfig, cell: ScanCell) -> dict[str, Any]:
    return {
        "experiment": "shock_tube_multiseed",
        "schema_version": 1,
        "base_config": "base-config.json",
        "seeds": list(scan.seeds),
        "required_backend": scan.required_backend,
        "overrides": cell.overrides,
        "statistics": scan.statistics,
        "acceptance": {
            "maximum_absolute_relative_energy_drift": scan.acceptance[
                "maximum_absolute_relative_energy_drift"
            ],
            "minimum_committed_surface_gap": scan.acceptance[
                "minimum_committed_surface_gap"
            ],
        },
    }


def cell_metadata(scan: ScanConfig, cell: ScanCell) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    try:
        scan_path = str(scan.path.relative_to(repository))
    except ValueError:
        scan_path = str(scan.path)
    return {
        "schema_version": 1,
        "experiment": "shock_tube_n_kn_scan_cell",
        "scan_config": scan_path,
        "cell_id": cell.cell_id,
        "particle_count": cell.particle_count,
        "target_knudsen": cell.target_knudsen,
        "realized_knudsen": cell.realized_knudsen,
        "characteristic_length": scan.characteristic_length,
        "density_reference": scan.density_reference,
        "reference_number_density": cell.reference_number_density,
        "maximum_number_density": cell.maximum_number_density,
        "mean_free_path": cell.mean_free_path,
        "particle_diameter": cell.particle_diameter,
        "transverse_extent": [cell.extent_y, cell.extent_z],
        "transverse_area": cell.transverse_area,
        "maximum_initial_packing_fraction": (
            cell.maximum_initial_packing_fraction
        ),
        "transverse_mean_free_path_ratio": (
            cell.transverse_mean_free_path_ratio
        ),
    }


def prepare_scan(
    scan: ScanConfig,
    cells: tuple[ScanCell, ...],
    output: Path,
) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    write_json_if_compatible(output / "scan-config.json", scan.raw)
    batch_paths: dict[str, Path] = {}
    for cell in cells:
        directory = output / "cells" / cell.cell_id
        directory.mkdir(parents=True, exist_ok=True)
        write_json_if_compatible(
            directory / "base-config.json",
            scan.base_config,
        )
        batch_path = directory / "batch-input.json"
        write_json_if_compatible(
            batch_path,
            batch_document(scan, cell),
        )
        write_json_if_compatible(
            directory / "cell-metadata.json",
            cell_metadata(scan, cell),
        )
        batch_paths[cell.cell_id] = batch_path
    write_scan_manifest(output / "scan-manifest.csv", cells)
    return batch_paths


def write_scan_manifest(path: Path, cells: tuple[ScanCell, ...]) -> None:
    fields = [
        "cell_id",
        "particle_count",
        "target_knudsen",
        "realized_knudsen",
        "particle_diameter",
        "mean_free_path",
        "reference_number_density",
        "maximum_number_density",
        "extent_y",
        "extent_z",
        "transverse_area",
        "maximum_initial_packing_fraction",
        "transverse_mean_free_path_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    field: getattr(cell, field)
                    for field in fields
                }
            )


def inspect_cell(
    scan: ScanConfig,
    cell: ScanCell,
    directory: Path,
) -> dict[str, Any]:
    simulation = copy.deepcopy(scan.base_config)
    deep_merge(simulation, cell.overrides)
    end_time = float(simulation["simulation"]["end_time"])
    runs = [
        inspect_run(directory / f"seed-{seed}", seed, end_time)
        for seed in scan.seeds
    ]
    completed = sum(run.completed for run in runs)
    backend_matches = all(
        not run.completed
        or scan.required_backend == "any"
        or run.backend == scan.required_backend
        for run in runs
    )
    acceptance_matches = all(
        not run.completed
        or (
            run.maximum_absolute_relative_energy_drift
            <= float(
                scan.acceptance[
                    "maximum_absolute_relative_energy_drift"
                ]
            )
            and run.minimum_committed_surface_gap
            >= float(scan.acceptance["minimum_committed_surface_gap"])
        )
        for run in runs
    )
    return {
        "cell_id": cell.cell_id,
        "particle_count": cell.particle_count,
        "target_knudsen": cell.target_knudsen,
        "completed_seeds": completed,
        "configured_seeds": len(scan.seeds),
        "backend_matches": backend_matches,
        "acceptance_matches": acceptance_matches,
        "completed": (
            completed >= int(scan.statistics["minimum_completed_seeds"])
            and backend_matches
            and acceptance_matches
        ),
    }


def write_scan_summary(
    path: Path,
    scan: ScanConfig,
    cells: tuple[ScanCell, ...],
    output: Path,
) -> None:
    inspections = [
        inspect_cell(scan, cell, output / "cells" / cell.cell_id)
        for cell in cells
    ]
    completed = sum(bool(item["completed"]) for item in inspections)
    document = {
        "schema_version": 1,
        "experiment": "shock_tube_n_kn_scan",
        "status": "completed" if completed == len(cells) else "incomplete",
        "configured_cells": len(cells),
        "completed_cells": completed,
        "particle_counts": list(scan.particle_counts),
        "knudsen_numbers": list(scan.knudsen_numbers),
        "configured_seeds_per_cell": len(scan.seeds),
        "cells": inspections,
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def run_self_test() -> int:
    path = Path(__file__).resolve().parents[1] / "configs" / (
        "shock_tube_n_kn_scan_smoke.json"
    )
    scan = load_scan_config(path)
    cells = derive_cells(scan)
    expected = len(scan.particle_counts) * len(scan.knudsen_numbers)
    if len(cells) != expected:
        raise AssertionError("factorial scan cell count is wrong")
    reference_densities = {
        round(cell.reference_number_density, 12) for cell in cells
    }
    if len(reference_densities) != 1:
        raise AssertionError("particle-count axis changed number density")
    for cell in cells:
        if not math.isclose(
            cell.realized_knudsen,
            cell.target_knudsen,
            rel_tol=2.0e-14,
        ):
            raise AssertionError("diameter did not realize target Knudsen")
    areas = {
        cell.particle_count: cell.transverse_area
        for cell in cells
    }
    low, high = sorted(areas)
    if not math.isclose(
        areas[high] / areas[low],
        high / low,
        rel_tol=2.0e-14,
    ):
        raise AssertionError("transverse area did not scale with N")
    print("particle-count/Knudsen scan parameterization self-test")
    print(f"  cells: {len(cells)}")
    print(f"  invariant reference density: {cells[0].reference_number_density:.9g}")
    print("  status: passed")
    return 0


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    arguments = parse_arguments()
    if arguments.self_test:
        return run_self_test()
    if (
        arguments.config is None
        or arguments.executable is None
        or arguments.output is None
    ):
        raise ValueError(
            "config, --executable, and --output are required"
        )
    if arguments.max_cells is not None and arguments.max_cells <= 0:
        raise ValueError("--max-cells must be positive")
    if arguments.max_cells is not None and arguments.cell_ids:
        raise ValueError("--max-cells and --cell cannot be combined")
    if arguments.max_seeds is not None and arguments.max_seeds <= 0:
        raise ValueError("--max-seeds must be positive")
    executable = arguments.executable.resolve()
    if not executable.is_file():
        raise ValueError(f"executable not found: {executable}")

    scan = load_scan_config(arguments.config)
    all_cells = derive_cells(scan)
    output = arguments.output.resolve()
    batch_paths = prepare_scan(scan, all_cells, output)
    selected = all_cells
    if arguments.cell_ids:
        requested = set(arguments.cell_ids)
        known = {cell.cell_id for cell in all_cells}
        unknown = sorted(requested.difference(known))
        if unknown:
            raise ValueError("unknown scan cells: " + ", ".join(unknown))
        selected = tuple(
            cell for cell in all_cells if cell.cell_id in requested
        )
    elif arguments.max_cells is not None:
        selected = selected[: arguments.max_cells]

    print("particle-count/Knudsen factorial scan")
    print(f"  config: {scan.path}")
    print(f"  particle levels: {len(scan.particle_counts)}")
    print(f"  Knudsen levels: {len(scan.knudsen_numbers)}")
    print(f"  configured cells: {len(all_cells)}")
    print(f"  selected cells: {len(selected)}")
    print(f"  seeds per cell: {len(scan.seeds)}")
    print(f"  required backend: {scan.required_backend}")
    print(f"  output: {output}")

    repository = Path(__file__).resolve().parents[1]
    runner = Path(__file__).resolve().parent / "run_multiseed_shock_tube.py"
    failures = []
    for index, cell in enumerate(selected, start=1):
        print(
            f"  cell {index}/{len(selected)}: {cell.cell_id}, "
            f"d={cell.particle_diameter:.9g}, "
            f"area={cell.transverse_area:.9g}"
        )
        command = [
            sys.executable,
            str(runner),
            str(batch_paths[cell.cell_id]),
            "--executable",
            str(executable),
            "--output",
            str(output / "cells" / cell.cell_id),
        ]
        if arguments.prepare_only:
            command.append("--prepare-only")
        if arguments.max_seeds is not None:
            command.extend(["--max-seeds", str(arguments.max_seeds)])
        completed = subprocess.run(command, cwd=repository, check=False)
        if completed.returncode != 0:
            failures.append(cell.cell_id)
        write_scan_summary(
            output / "scan-summary.json",
            scan,
            all_cells,
            output,
        )
    if failures:
        print("  failed cells: " + ", ".join(failures))
        return 1
    print("  selected cells processed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, csv.Error) as exception:
        raise SystemExit(f"N/Kn scan runner: {exception}") from exception
