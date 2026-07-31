#!/usr/bin/env python3
"""Prepare and run independent shock-tube seeds with resumable bookkeeping."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BatchConfig:
    path: Path
    raw: dict[str, Any]
    base_config_path: Path
    effective_config: dict[str, Any]
    seeds: tuple[int, ...]
    required_backend: str
    confidence_level: float
    euler_cells: int
    minimum_completed_seeds: int
    maximum_absolute_relative_energy_drift: float
    minimum_committed_surface_gap: float

    @property
    def end_time(self) -> float:
        simulation = self.effective_config["simulation"]
        assert isinstance(simulation, dict)
        return float(simulation["end_time"])

    @property
    def particle_count(self) -> int:
        particles = self.effective_config["particles"]
        assert isinstance(particles, dict)
        return int(particles["count"])


@dataclass(frozen=True)
class RunInspection:
    seed: int
    completed: bool
    backend: str | None
    final_time: float
    final_status: str
    maximum_absolute_relative_energy_drift: float
    minimum_committed_surface_gap: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="shock_tube_multiseed experiment JSON",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        required=True,
        help="particles_cli executable",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="batch output directory",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write per-seed configs without launching simulations",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        help="process only the first N configured seeds",
    )
    return parser.parse_args()


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require_integer(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def require_number(
    value: Any,
    name: str,
    *,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(base.get(key), dict)
        ):
            nested = base[key]
            assert isinstance(nested, dict)
            deep_merge(nested, value)
        else:
            base[key] = copy.deepcopy(value)


def load_batch_config(path: Path) -> BatchConfig:
    resolved_path = path.resolve()
    raw = load_json_object(resolved_path)
    if raw.get("experiment") != "shock_tube_multiseed":
        raise ValueError("experiment must be shock_tube_multiseed")
    if raw.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")

    base_name = raw.get("base_config")
    if not isinstance(base_name, str) or not base_name:
        raise ValueError("base_config must be a non-empty path string")
    base_path = (resolved_path.parent / base_name).resolve()
    effective = load_json_object(base_path)
    if effective.get("experiment") != "shock_tube":
        raise ValueError("base_config must be a shock_tube config")
    overrides = raw.get("overrides")
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be an object")
    deep_merge(effective, overrides)

    seeds_value = raw.get("seeds")
    if not isinstance(seeds_value, list) or len(seeds_value) < 2:
        raise ValueError("seeds must contain at least two integers")
    seeds = tuple(
        require_integer(seed, f"seeds[{index}]", minimum=0)
        for index, seed in enumerate(seeds_value)
    )
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    required_backend = raw.get("required_backend")
    if required_backend not in {"any", "cuda"}:
        raise ValueError("required_backend must be any or cuda")
    statistics = raw.get("statistics")
    acceptance = raw.get("acceptance")
    if not isinstance(statistics, dict) or not isinstance(acceptance, dict):
        raise ValueError("statistics and acceptance must be objects")
    confidence_level = require_number(
        statistics.get("confidence_level"),
        "statistics.confidence_level",
        positive=True,
    )
    if confidence_level != 0.95:
        raise ValueError("only a 0.95 confidence level is currently supported")
    euler_cells = require_integer(
        statistics.get("euler_cells"),
        "statistics.euler_cells",
        minimum=8,
    )
    minimum_completed_seeds = require_integer(
        statistics.get("minimum_completed_seeds"),
        "statistics.minimum_completed_seeds",
        minimum=2,
    )
    if minimum_completed_seeds > len(seeds):
        raise ValueError("minimum_completed_seeds exceeds configured seeds")

    particles = effective.get("particles")
    diagnostics = effective.get("diagnostics")
    simulation = effective.get("simulation")
    if not all(
        isinstance(value, dict)
        for value in (particles, diagnostics, simulation)
    ):
        raise ValueError(
            "effective particles, diagnostics, and simulation must be objects"
        )
    assert isinstance(particles, dict)
    assert isinstance(diagnostics, dict)
    assert isinstance(simulation, dict)
    require_integer(particles.get("count"), "particles.count", minimum=2)
    require_number(
        particles.get("diameter"),
        "particles.diameter",
        positive=True,
    )
    require_integer(
        diagnostics.get("macroscopic_x_slabs"),
        "diagnostics.macroscopic_x_slabs",
        minimum=16,
    )
    require_number(
        simulation.get("end_time"),
        "simulation.end_time",
        positive=True,
    )
    require_number(
        simulation.get("frame_interval"),
        "simulation.frame_interval",
        positive=True,
    )

    return BatchConfig(
        path=resolved_path,
        raw=raw,
        base_config_path=base_path,
        effective_config=effective,
        seeds=seeds,
        required_backend=required_backend,
        confidence_level=confidence_level,
        euler_cells=euler_cells,
        minimum_completed_seeds=minimum_completed_seeds,
        maximum_absolute_relative_energy_drift=require_number(
            acceptance.get("maximum_absolute_relative_energy_drift"),
            "acceptance.maximum_absolute_relative_energy_drift",
            positive=True,
        ),
        minimum_committed_surface_gap=require_number(
            acceptance.get("minimum_committed_surface_gap"),
            "acceptance.minimum_committed_surface_gap",
        ),
    )


def write_json_if_compatible(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        existing = load_json_object(path)
        if existing != value:
            raise ValueError(
                f"{path} already exists with different experiment data"
            )
        return
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def seed_directory(root: Path, seed: int) -> Path:
    return root / f"seed-{seed}"


def prepare_batch(
    batch: BatchConfig,
    output_root: Path,
) -> dict[int, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_if_compatible(
        output_root / "experiment-config.json",
        batch.raw,
    )
    write_json_if_compatible(
        output_root / "effective-config.json",
        batch.effective_config,
    )
    paths = {}
    for seed in batch.seeds:
        directory = seed_directory(output_root, seed)
        directory.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(batch.effective_config)
        config["random_seed"] = seed
        config_path = directory / "run-config.json"
        write_json_if_compatible(config_path, config)
        paths[seed] = config_path
    return paths


def finite_float(value: str) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def inspect_run(
    directory: Path,
    seed: int,
    end_time: float,
) -> RunInspection:
    evolution_path = directory / "evolution.csv"
    if not evolution_path.is_file():
        return RunInspection(
            seed=seed,
            completed=False,
            backend=None,
            final_time=0.0,
            final_status="missing",
            maximum_absolute_relative_energy_drift=math.inf,
            minimum_committed_surface_gap=-math.inf,
        )
    with evolution_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{evolution_path}: no rows")
    last = rows[-1]
    drifts = [
        abs(value)
        for row in rows
        if (value := finite_float(row["relative_energy_drift"])) is not None
    ]
    gaps = [
        value
        for row in rows
        if (value := finite_float(row["minimum_surface_gap"])) is not None
    ]
    final_time = float(last["time"])
    final_status = last["status"]
    completed = (
        final_status == "completed"
        and final_time >= end_time - 1.0e-12
    )
    return RunInspection(
        seed=seed,
        completed=completed,
        backend=last["backend"],
        final_time=final_time,
        final_status=final_status,
        maximum_absolute_relative_energy_drift=max(drifts, default=math.inf),
        minimum_committed_surface_gap=min(gaps, default=-math.inf),
    )


def write_batch_summary(
    path: Path,
    batch: BatchConfig,
    inspections: list[RunInspection],
) -> None:
    def json_number(value: float) -> float | None:
        return value if math.isfinite(value) else None

    completed = sum(item.completed for item in inspections)
    value = {
        "schema_version": 1,
        "experiment": "shock_tube_multiseed",
        "particle_count": batch.particle_count,
        "configured_seeds": list(batch.seeds),
        "completed_seeds": completed,
        "minimum_completed_seeds": batch.minimum_completed_seeds,
        "runs": [
            {
                "seed": item.seed,
                "completed": item.completed,
                "backend": item.backend,
                "final_time": item.final_time,
                "final_status": item.final_status,
                "maximum_absolute_relative_energy_drift": (
                    json_number(
                        item.maximum_absolute_relative_energy_drift
                    )
                ),
                "minimum_committed_surface_gap": (
                    json_number(item.minimum_committed_surface_gap)
                ),
            }
            for item in inspections
        ],
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def cuda_preflight(executable: Path, repository_root: Path) -> None:
    config = repository_root / "configs" / "equilibrium.json"
    print("CUDA preflight")
    completed = subprocess.run(
        [str(executable), "cuda-grid", str(config)],
        cwd=repository_root,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CUDA is required, but particles_cli cuda-grid did not pass"
        )


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    arguments = parse_arguments()
    batch = load_batch_config(arguments.config)
    executable = arguments.executable.resolve()
    if not executable.is_file():
        raise ValueError(f"executable not found: {executable}")
    if arguments.max_seeds is not None and arguments.max_seeds <= 0:
        raise ValueError("--max-seeds must be positive")

    output_root = arguments.output.resolve()
    config_paths = prepare_batch(batch, output_root)
    selected_seeds = batch.seeds
    if arguments.max_seeds is not None:
        selected_seeds = selected_seeds[: arguments.max_seeds]

    print("multi-seed shock-tube batch")
    print(f"  config: {batch.path}")
    print(f"  particles per seed: {batch.particle_count}")
    print(f"  configured seeds: {len(batch.seeds)}")
    print(f"  selected seeds: {len(selected_seeds)}")
    print(f"  required backend: {batch.required_backend}")
    print(f"  output: {output_root}")
    if arguments.prepare_only:
        print("  status: prepared")
        return 0

    pending = [
        seed
        for seed in selected_seeds
        if not inspect_run(
            seed_directory(output_root, seed),
            seed,
            batch.end_time,
        ).completed
    ]
    repository_root = Path(__file__).resolve().parents[1]
    if pending and batch.required_backend == "cuda":
        cuda_preflight(executable, repository_root)

    selected_inspections: list[RunInspection] = []
    for index, seed in enumerate(selected_seeds, start=1):
        directory = seed_directory(output_root, seed)
        inspection = inspect_run(directory, seed, batch.end_time)
        if inspection.completed:
            print(
                f"  seed {seed} ({index}/{len(selected_seeds)}): "
                "already completed"
            )
        else:
            print(
                f"  seed {seed} ({index}/{len(selected_seeds)}): running"
            )
            start = time.perf_counter()
            completed = subprocess.run(
                [
                    str(executable),
                    "shock-tube",
                    str(config_paths[seed]),
                    str(directory),
                ],
                cwd=repository_root,
                check=False,
            )
            wall_seconds = time.perf_counter() - start
            inspection = inspect_run(
                directory,
                seed,
                batch.end_time,
            )
            print(
                f"    return code={completed.returncode}, "
                f"wall={wall_seconds:.6g} s, "
                f"status={inspection.final_status}"
            )
        selected_inspections.append(inspection)
        all_inspections = [
            inspect_run(
                seed_directory(output_root, configured_seed),
                configured_seed,
                batch.end_time,
            )
            for configured_seed in batch.seeds
        ]
        write_batch_summary(
            output_root / "batch-summary.json",
            batch,
            all_inspections,
        )

    failures = [
        item for item in selected_inspections if not item.completed
    ]
    if failures:
        print(
            "  incomplete seeds: "
            + ", ".join(str(item.seed) for item in failures)
        )
        return 1
    print("  selected seeds completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, csv.Error) as exception:
        raise SystemExit(f"multi-seed runner: {exception}") from exception
