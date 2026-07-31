#!/usr/bin/env python3
"""Aggregate independent shock-tube seeds into field confidence bands."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "particles-matplotlib")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from euler_reference import (  # noqa: E402
    EulerParameters,
    solve_euler_reference,
)
from run_multiseed_shock_tube import (  # noqa: E402
    BatchConfig,
    inspect_run,
    load_batch_config,
    seed_directory,
)


FIELDS = (
    "number_density",
    "bulk_velocity_x",
    "temperature",
    "pressure",
)
FIELD_LABELS = {
    "number_density": r"$n(x)$",
    "bulk_velocity_x": r"$u_x(x)$",
    "temperature": r"$T(x)$",
    "pressure": r"$p(x)$",
}
BACKGROUND = "#090d18"
PANEL = "#111827"
FOREGROUND = "#e5e7eb"
MUTED = "#94a3b8"
GRID = "#334155"
PARTICLE_COLOR = "#38bdf8"
EULER_COLOR = "#f59e0b"


@dataclass(frozen=True)
class Ensemble:
    seeds: tuple[int, ...]
    frames: np.ndarray
    times: np.ndarray
    slabs: np.ndarray
    x: np.ndarray
    values: np.ndarray


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        help="shock_tube_multiseed experiment JSON",
    )
    parser.add_argument(
        "batch_output",
        type=Path,
        nargs="?",
        help="directory produced by run_multiseed_shock_tube.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="statistics output directory; defaults to BATCH_OUTPUT/statistics",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate Student-t intervals and aggregation arithmetic",
    )
    return parser.parse_args()


def student_t_critical_95(degrees_of_freedom: int) -> float:
    values = {
        1: 12.7062047364,
        2: 4.30265272975,
        3: 3.18244630528,
        4: 2.7764451052,
        5: 2.57058183564,
        6: 2.44691184879,
        7: 2.36462425101,
        8: 2.3060041352,
        9: 2.26215716285,
        10: 2.22813885196,
        11: 2.20098516008,
        12: 2.17881282966,
        13: 2.16036865646,
        14: 2.14478668792,
        15: 2.13144954556,
        16: 2.11990529922,
        17: 2.10981557783,
        18: 2.10092204024,
        19: 2.09302405441,
        20: 2.08596344727,
        21: 2.07961384473,
        22: 2.0738730679,
        23: 2.06865761042,
        24: 2.06389856163,
        25: 2.05953855275,
        26: 2.05552943864,
        27: 2.05183051648,
        28: 2.0484071418,
        29: 2.04522964213,
        30: 2.0422724563,
    }
    if degrees_of_freedom < 1:
        raise ValueError("Student-t interval requires at least two seeds")
    if degrees_of_freedom in values:
        return values[degrees_of_freedom]
    return 1.95996398454


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_macroscopic_rows(
    path: Path,
) -> tuple[list[tuple[int, int, float, float]], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"frame", "time", "slab", "x_center", *FIELDS}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no rows")
    rows.sort(key=lambda row: (int(row["frame"]), int(row["slab"])))
    keys = [
        (
            int(row["frame"]),
            int(row["slab"]),
            float(row["time"]),
            float(row["x_center"]),
        )
        for row in rows
    ]
    if len({(frame, slab) for frame, slab, _, _ in keys}) != len(keys):
        raise ValueError(f"{path}: duplicate frame/slab rows")
    values = np.array(
        [[float(row[field]) for field in FIELDS] for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: non-finite field value")
    return keys, values


def load_ensemble(
    batch: BatchConfig,
    batch_output: Path,
) -> Ensemble:
    completed_seeds = []
    reference_keys: list[tuple[int, int, float, float]] | None = None
    arrays = []
    for seed in batch.seeds:
        directory = seed_directory(batch_output, seed)
        inspection = inspect_run(directory, seed, batch.end_time)
        if not inspection.completed:
            continue
        if (
            batch.required_backend == "cuda"
            and inspection.backend != "cuda"
        ):
            raise ValueError(f"seed {seed} did not use the CUDA backend")
        if (
            inspection.maximum_absolute_relative_energy_drift
            > batch.maximum_absolute_relative_energy_drift
        ):
            raise ValueError(
                f"seed {seed} energy drift "
                f"{inspection.maximum_absolute_relative_energy_drift:g} "
                "exceeds acceptance"
            )
        if (
            inspection.minimum_committed_surface_gap
            < batch.minimum_committed_surface_gap
        ):
            raise ValueError(
                f"seed {seed} minimum gap "
                f"{inspection.minimum_committed_surface_gap:g} "
                "is below acceptance"
            )
        keys, values = load_macroscopic_rows(directory / "macroscopic.csv")
        if reference_keys is None:
            reference_keys = keys
        else:
            if len(keys) != len(reference_keys):
                raise ValueError(f"seed {seed} field row count differs")
            for reference, candidate in zip(reference_keys, keys, strict=True):
                if reference[:2] != candidate[:2] or not np.allclose(
                    reference[2:],
                    candidate[2:],
                    rtol=0.0,
                    atol=1.0e-13,
                ):
                    raise ValueError(
                        f"seed {seed} frame/slab grid differs"
                    )
        completed_seeds.append(seed)
        arrays.append(values)
    if len(completed_seeds) < batch.minimum_completed_seeds:
        raise ValueError(
            f"only {len(completed_seeds)} seeds completed; "
            f"{batch.minimum_completed_seeds} required"
        )
    assert reference_keys is not None

    frame_ids = sorted({key[0] for key in reference_keys})
    slab_ids = sorted({key[1] for key in reference_keys})
    frame_count = len(frame_ids)
    slab_count = len(slab_ids)
    if len(reference_keys) != frame_count * slab_count:
        raise ValueError("macroscopic grid is not rectangular")
    if frame_ids != list(range(frame_count)):
        raise ValueError("frame indices must be contiguous from zero")
    if slab_ids != list(range(slab_count)):
        raise ValueError("slab indices must be contiguous from zero")

    times = np.empty(frame_count, dtype=np.float64)
    x = np.empty(slab_count, dtype=np.float64)
    for frame in frame_ids:
        frame_times = {
            key[2] for key in reference_keys if key[0] == frame
        }
        if len(frame_times) != 1:
            raise ValueError(f"frame {frame} has inconsistent times")
        times[frame] = frame_times.pop()
    for slab in slab_ids:
        slab_x = {key[3] for key in reference_keys if key[1] == slab}
        if len(slab_x) != 1:
            raise ValueError(f"slab {slab} has inconsistent x centers")
        x[slab] = slab_x.pop()

    stacked = np.stack(arrays).reshape(
        len(arrays),
        frame_count,
        slab_count,
        len(FIELDS),
    )
    return Ensemble(
        seeds=tuple(completed_seeds),
        frames=np.array(frame_ids, dtype=np.int64),
        times=times,
        slabs=np.array(slab_ids, dtype=np.int64),
        x=x,
        values=stacked,
    )


def euler_at_particle_grid(
    effective_config: Path,
    batch: BatchConfig,
    ensemble: Ensemble,
) -> np.ndarray:
    solution = solve_euler_reference(
        effective_config,
        [
            (int(frame), float(time))
            for frame, time in zip(
                ensemble.frames,
                ensemble.times,
                strict=True,
            )
        ],
        particle_count=batch.particle_count,
        parameters=EulerParameters(cells=batch.euler_cells),
    )
    values = np.empty(
        (
            len(ensemble.frames),
            len(ensemble.slabs),
            len(FIELDS),
        ),
        dtype=np.float64,
    )
    for frame_index, frame in enumerate(solution.frames):
        euler_x = frame.fields["x_center"]
        for field_index, field in enumerate(FIELDS):
            values[frame_index, :, field_index] = np.interp(
                ensemble.x,
                euler_x,
                frame.fields[field],
            )
    return values


def confidence_statistics(
    ensemble: Ensemble,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    sample_count = len(ensemble.seeds)
    critical = student_t_critical_95(sample_count - 1)
    mean = np.mean(ensemble.values, axis=0)
    standard_deviation = np.std(ensemble.values, axis=0, ddof=1)
    standard_error = standard_deviation / math.sqrt(sample_count)
    half_width = critical * standard_error
    if not all(
        np.all(np.isfinite(value))
        for value in (mean, standard_deviation, standard_error, half_width)
    ):
        raise ValueError("confidence statistics contain non-finite values")
    return mean, standard_deviation, half_width, critical


def write_seed_summary(
    path: Path,
    batch: BatchConfig,
    batch_output: Path,
    seeds: tuple[int, ...],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "seed",
                "backend",
                "final_time",
                "maximum_absolute_relative_energy_drift",
                "minimum_committed_surface_gap",
            ]
        )
        for seed in seeds:
            inspection = inspect_run(
                seed_directory(batch_output, seed),
                seed,
                batch.end_time,
            )
            writer.writerow(
                [
                    seed,
                    inspection.backend,
                    inspection.final_time,
                    inspection.maximum_absolute_relative_energy_drift,
                    inspection.minimum_committed_surface_gap,
                ]
            )


def write_field_bands(
    path: Path,
    ensemble: Ensemble,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    half_width: np.ndarray,
    euler: np.ndarray,
) -> None:
    sample_count = len(ensemble.seeds)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "frame",
                "time",
                "slab",
                "x_center",
                "field",
                "seed_count",
                "mean",
                "sample_standard_deviation",
                "standard_error",
                "ci95_lower",
                "ci95_upper",
                "ci95_half_width",
                "euler_value",
                "mean_minus_euler",
                "euler_inside_ci95",
            ]
        )
        for frame_index, frame in enumerate(ensemble.frames):
            for slab_index, slab in enumerate(ensemble.slabs):
                for field_index, field in enumerate(FIELDS):
                    field_mean = mean[frame_index, slab_index, field_index]
                    field_std = standard_deviation[
                        frame_index,
                        slab_index,
                        field_index,
                    ]
                    band = half_width[frame_index, slab_index, field_index]
                    euler_value = euler[
                        frame_index,
                        slab_index,
                        field_index,
                    ]
                    writer.writerow(
                        [
                            int(frame),
                            ensemble.times[frame_index],
                            int(slab),
                            ensemble.x[slab_index],
                            field,
                            sample_count,
                            field_mean,
                            field_std,
                            field_std / math.sqrt(sample_count),
                            field_mean - band,
                            field_mean + band,
                            band,
                            euler_value,
                            field_mean - euler_value,
                            (
                                "yes"
                                if field_mean - band
                                <= euler_value
                                <= field_mean + band
                                else "no"
                            ),
                        ]
                    )


def frame_summary_rows(
    ensemble: Ensemble,
    mean: np.ndarray,
    half_width: np.ndarray,
    euler: np.ndarray,
) -> list[dict[str, float | int | str]]:
    rows = []
    tiny = np.finfo(np.float64).tiny
    for frame_index, frame in enumerate(ensemble.frames):
        for field_index, field in enumerate(FIELDS):
            difference = (
                mean[frame_index, :, field_index]
                - euler[frame_index, :, field_index]
            )
            reference = euler[frame_index, :, field_index]
            band = half_width[frame_index, :, field_index]
            reference_rms = float(np.sqrt(np.mean(np.square(reference))))
            difference_rms = float(np.sqrt(np.mean(np.square(difference))))
            band_rms = float(np.sqrt(np.mean(np.square(band))))
            inside = np.abs(difference) <= band
            rows.append(
                {
                    "frame": int(frame),
                    "time": float(ensemble.times[frame_index]),
                    "field": field,
                    "relative_l2_mean_vs_euler": (
                        difference_rms / max(reference_rms, tiny)
                    ),
                    "maximum_absolute_mean_vs_euler": float(
                        np.max(np.abs(difference))
                    ),
                    "relative_rms_ci95_half_width": (
                        band_rms / max(reference_rms, tiny)
                    ),
                    "euler_coverage_fraction": float(np.mean(inside)),
                }
            )
    return rows


def write_frame_summary(
    path: Path,
    rows: list[dict[str, float | int | str]],
) -> None:
    fieldnames = [
        "frame",
        "time",
        "field",
        "relative_l2_mean_vs_euler",
        "maximum_absolute_mean_vs_euler",
        "relative_rms_ci95_half_width",
        "euler_coverage_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=FOREGROUND)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.grid(color=GRID, alpha=0.45, linewidth=0.8)
    axis.xaxis.label.set_color(FOREGROUND)
    axis.yaxis.label.set_color(FOREGROUND)
    axis.title.set_color(FOREGROUND)


def write_final_plot(
    path: Path,
    particle_count: int,
    ensemble: Ensemble,
    mean: np.ndarray,
    half_width: np.ndarray,
    euler: np.ndarray,
) -> None:
    frame_index = len(ensemble.frames) - 1
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 7.2),
        facecolor=BACKGROUND,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.87,
        bottom=0.09,
        hspace=0.3,
        wspace=0.2,
    )
    for axis, field_index, field in zip(
        axes.flat,
        range(len(FIELDS)),
        FIELDS,
        strict=True,
    ):
        style_axis(axis)
        field_mean = mean[frame_index, :, field_index]
        band = half_width[frame_index, :, field_index]
        axis.fill_between(
            ensemble.x,
            field_mean - band,
            field_mean + band,
            color=PARTICLE_COLOR,
            alpha=0.22,
            linewidth=0.0,
            label="hard spheres — 95% mean CI",
        )
        axis.plot(
            ensemble.x,
            field_mean,
            color=PARTICLE_COLOR,
            linewidth=1.8,
            label="hard-sphere seed mean",
        )
        axis.plot(
            ensemble.x,
            euler[frame_index, :, field_index],
            color=EULER_COLOR,
            linestyle="--",
            linewidth=1.5,
            label="ideal Euler",
        )
        axis.set_title(FIELD_LABELS[field])
        axis.set_xlabel("x")
        axis.legend(
            facecolor=PANEL,
            edgecolor=GRID,
            labelcolor=FOREGROUND,
            fontsize=8,
        )
    figure.suptitle(
        (
            f"Independent-seed field bands at t="
            f"{ensemble.times[frame_index]:g} "
            f"({len(ensemble.seeds)} seeds, "
            f"{particle_count:,} particles/seed)"
        ),
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def write_summary_json(
    path: Path,
    batch: BatchConfig,
    batch_output: Path,
    ensemble: Ensemble,
    critical: float,
    frame_rows: list[dict[str, float | int | str]],
) -> None:
    final_frame = int(ensemble.frames[-1])
    final_rows = [
        row for row in frame_rows if row["frame"] == final_frame
    ]
    sources = []
    for seed in ensemble.seeds:
        directory = seed_directory(batch_output, seed)
        sources.append(
            {
                "seed": seed,
                "macroscopic_sha256": sha256_file(
                    directory / "macroscopic.csv"
                ),
                "evolution_sha256": sha256_file(
                    directory / "evolution.csv"
                ),
            }
        )
    value = {
        "schema_version": 1,
        "status": "passed",
        "method": {
            "independent_random_seeds": True,
            "confidence_interval": (
                "two-sided Student-t interval for the field mean"
            ),
            "confidence_level": batch.confidence_level,
            "degrees_of_freedom": len(ensemble.seeds) - 1,
            "critical_value": critical,
            "seed_count": len(ensemble.seeds),
            "euler_cells": batch.euler_cells,
            "analysis_tool_sha256": sha256_file(Path(__file__)),
        },
        "experiment": {
            "particle_count_per_seed": batch.particle_count,
            "seeds": list(ensemble.seeds),
            "frame_count": len(ensemble.frames),
            "slab_count": len(ensemble.slabs),
            "end_time": float(ensemble.times[-1]),
        },
        "final_frame": {
            str(row["field"]): {
                key: value
                for key, value in row.items()
                if key not in {"frame", "time", "field"}
            }
            for row in final_rows
        },
        "sources": sources,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def run_self_test() -> int:
    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
        dtype=np.float64,
    ).reshape(4, 1, 1, 4)
    ensemble = Ensemble(
        seeds=(1, 2, 3, 4),
        frames=np.array([0]),
        times=np.array([0.1]),
        slabs=np.array([0]),
        x=np.array([0.5]),
        values=values,
    )
    mean, standard_deviation, half_width, critical = (
        confidence_statistics(ensemble)
    )
    expected_mean = np.array([2.5, 3.5, 4.5, 5.5])
    expected_std = math.sqrt(5.0 / 3.0)
    if not np.allclose(mean[0, 0], expected_mean):
        raise AssertionError("multi-seed mean self-test failed")
    if not np.allclose(standard_deviation[0, 0], expected_std):
        raise AssertionError("multi-seed standard deviation self-test failed")
    if not math.isclose(critical, 3.18244630528):
        raise AssertionError("Student-t critical value self-test failed")
    expected_half_width = critical * expected_std / 2.0
    if not np.allclose(half_width[0, 0], expected_half_width):
        raise AssertionError("confidence half-width self-test failed")
    print("multi-seed field statistics self-test")
    print(f"  seeds: {len(ensemble.seeds)}")
    print(f"  degrees of freedom: {len(ensemble.seeds) - 1}")
    print(f"  t critical: {critical:.12g}")
    print("  status: passed")
    return 0


def main() -> int:
    arguments = parse_arguments()
    if arguments.self_test:
        return run_self_test()
    if arguments.config is None or arguments.batch_output is None:
        raise ValueError(
            "config and batch_output are required unless --self-test is used"
        )
    batch = load_batch_config(arguments.config)
    batch_output = arguments.batch_output.resolve()
    effective_config = batch_output / "effective-config.json"
    if not effective_config.is_file():
        raise ValueError(
            f"{effective_config} is missing; prepare the batch first"
        )
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else batch_output / "statistics"
    )
    output.mkdir(parents=True, exist_ok=True)

    ensemble = load_ensemble(batch, batch_output)
    mean, standard_deviation, half_width, critical = (
        confidence_statistics(ensemble)
    )
    euler = euler_at_particle_grid(
        effective_config,
        batch,
        ensemble,
    )
    frame_rows = frame_summary_rows(
        ensemble,
        mean,
        half_width,
        euler,
    )

    seed_summary_path = output / "seed-summary.csv"
    bands_path = output / "field-bands.csv"
    frame_summary_path = output / "frame-summary.csv"
    summary_path = output / "summary.json"
    plot_path = output / "final-field-bands.png"
    write_seed_summary(
        seed_summary_path,
        batch,
        batch_output,
        ensemble.seeds,
    )
    write_field_bands(
        bands_path,
        ensemble,
        mean,
        standard_deviation,
        half_width,
        euler,
    )
    write_frame_summary(frame_summary_path, frame_rows)
    write_summary_json(
        summary_path,
        batch,
        batch_output,
        ensemble,
        critical,
        frame_rows,
    )
    write_final_plot(
        plot_path,
        batch.particle_count,
        ensemble,
        mean,
        half_width,
        euler,
    )

    print("multi-seed shock-tube field bands")
    print(f"  particles per seed: {batch.particle_count}")
    print(f"  completed seeds: {len(ensemble.seeds)}")
    print(
        f"  95% Student-t critical value: {critical:.9g} "
        f"(df={len(ensemble.seeds) - 1})"
    )
    print(f"  frames/slabs: {len(ensemble.frames)} / {len(ensemble.slabs)}")
    print(f"  final time: {ensemble.times[-1]:g}")
    for row in frame_rows:
        if row["frame"] != int(ensemble.frames[-1]):
            continue
        print(
            f"  final {row['field']}: relative L2 mean/Euler="
            f"{float(row['relative_l2_mean_vs_euler']):.6g}, "
            f"relative rms CI half-width="
            f"{float(row['relative_rms_ci95_half_width']):.6g}, "
            f"Euler coverage="
            f"{float(row['euler_coverage_fraction']):.6g}"
        )
    print(f"  seed summary: {seed_summary_path}")
    print(f"  field bands: {bands_path}")
    print(f"  frame summary: {frame_summary_path}")
    print(f"  summary JSON: {summary_path}")
    print(f"  plot: {plot_path}")
    print("  overall status: passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, csv.Error) as exception:
        raise SystemExit(f"multi-seed analysis: {exception}") from exception
