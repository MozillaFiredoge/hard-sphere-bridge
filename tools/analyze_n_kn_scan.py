#!/usr/bin/env python3
"""Analyze a factorial particle-count/Knudsen shock-tube scan."""

from __future__ import annotations

import argparse
import csv
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

from analyze_multiseed_fields import (  # noqa: E402
    FIELDS,
    FIELD_LABELS,
    aggregate_wave_speeds,
    confidence_statistics,
    euler_at_particle_grid,
    frame_summary_rows,
    load_ensemble,
    portable_path,
    sha256_file,
)
from analyze_wave_kinematics import FEATURE_LABELS  # noqa: E402
from run_multiseed_shock_tube import (  # noqa: E402
    inspect_run,
    load_batch_config,
    seed_directory,
)
from run_n_kn_scan import (  # noqa: E402
    ScanCell,
    ScanConfig,
    derive_cells,
    load_scan_config,
)


BACKGROUND = "#090d18"
PANEL = "#111827"
FOREGROUND = "#e5e7eb"
MUTED = "#94a3b8"
GRID = "#334155"
COLORS = ("#38bdf8", "#a78bfa", "#22c55e", "#f97316")
EULER_COLOR = "#f59e0b"


@dataclass(frozen=True)
class CellAnalysis:
    cell: ScanCell
    seed_count: int
    final_time: float
    maximum_energy_drift: float
    minimum_surface_gap: float
    final_local_knudsen_p05: float
    final_local_knudsen_median: float
    final_local_knudsen_p95: float
    field_metrics: dict[str, dict[str, float]]
    wave_metrics: dict[str, dict[str, float | bool]] | None
    wave_error: str | None
    source_hashes: tuple[dict[str, Any], ...]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        help="shock_tube_n_kn_scan experiment JSON",
    )
    parser.add_argument(
        "scan_output",
        type=Path,
        nargs="?",
        help="directory produced by run_n_kn_scan.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="analysis directory; defaults to SCAN_OUTPUT/analysis",
    )
    parser.add_argument(
        "--wave-config",
        type=Path,
        help="optional frozen wave-marker configuration",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="analyze complete cells and report missing cells instead of failing",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate log-log trend arithmetic and exit",
    )
    return parser.parse_args()


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=FOREGROUND)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.grid(color=GRID, alpha=0.45, linewidth=0.8)
    axis.xaxis.label.set_color(FOREGROUND)
    axis.yaxis.label.set_color(FOREGROUND)
    axis.title.set_color(FOREGROUND)


def color_tick_labels(axis: plt.Axes) -> None:
    axis.tick_params(colors=FOREGROUND, which="both")
    for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        label.set_color(FOREGROUND)


def log_log_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("log-log slope requires paired one-dimensional data")
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("log-log slope requires positive values")
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def monotonic_direction(values: list[float]) -> str:
    if all(b <= a for a, b in zip(values, values[1:])):
        return "nonincreasing"
    if all(b >= a for a, b in zip(values, values[1:])):
        return "nondecreasing"
    return "nonmonotonic"


def cell_is_complete(
    scan: ScanConfig,
    cell: ScanCell,
    cell_output: Path,
) -> bool:
    batch_path = cell_output / "batch-input.json"
    if not batch_path.is_file():
        return False
    batch = load_batch_config(batch_path)
    completed = 0
    for seed in scan.seeds:
        inspection = inspect_run(
            seed_directory(cell_output, seed),
            seed,
            batch.end_time,
        )
        if not inspection.completed:
            continue
        if (
            scan.required_backend == "cuda"
            and inspection.backend != "cuda"
        ):
            continue
        if (
            inspection.maximum_absolute_relative_energy_drift
            > float(
                scan.acceptance[
                    "maximum_absolute_relative_energy_drift"
                ]
            )
            or inspection.minimum_committed_surface_gap
            < float(scan.acceptance["minimum_committed_surface_gap"])
        ):
            continue
        completed += 1
    return completed >= int(scan.statistics["minimum_completed_seeds"])


def analyze_cell(
    scan: ScanConfig,
    cell: ScanCell,
    cell_output: Path,
    wave_config: Path | None,
) -> CellAnalysis:
    batch_path = cell_output / "batch-input.json"
    batch = load_batch_config(batch_path)
    effective_path = cell_output / "effective-config.json"
    ensemble = load_ensemble(batch, cell_output)
    mean, _, half_width, critical = confidence_statistics(ensemble)
    euler, euler_solution = euler_at_particle_grid(
        effective_path,
        batch,
        ensemble,
    )
    rows = frame_summary_rows(ensemble, mean, half_width, euler)
    final_frame = int(ensemble.frames[-1])
    final_number_density = mean[final_frame, :, 0]
    final_local_knudsen = 1.0 / (
        math.sqrt(2.0)
        * math.pi
        * final_number_density
        * cell.particle_diameter**2
        * scan.characteristic_length
    )
    if not np.all(np.isfinite(final_local_knudsen)):
        raise ValueError(f"{cell.cell_id}: non-finite final local Knudsen")
    field_metrics: dict[str, dict[str, float]] = {}
    for row in rows:
        if int(row["frame"]) != final_frame:
            continue
        field = str(row["field"])
        field_metrics[field] = {
            key: float(value)
            for key, value in row.items()
            if key not in {"frame", "time", "field"}
        }

    wave_metrics: dict[str, dict[str, float | bool]] | None = None
    wave_error: str | None = None
    if wave_config is not None:
        try:
            _, wave_metrics = aggregate_wave_speeds(
                effective_path,
                ensemble,
                euler_solution,
                wave_config,
                critical,
            )
        except (ValueError, RuntimeError) as exception:
            wave_error = str(exception)

    inspections = [
        inspect_run(
            seed_directory(cell_output, seed),
            seed,
            batch.end_time,
        )
        for seed in ensemble.seeds
    ]
    hashes = []
    for seed in ensemble.seeds:
        directory = seed_directory(cell_output, seed)
        hashes.append(
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
    return CellAnalysis(
        cell=cell,
        seed_count=len(ensemble.seeds),
        final_time=float(ensemble.times[-1]),
        maximum_energy_drift=max(
            item.maximum_absolute_relative_energy_drift
            for item in inspections
        ),
        minimum_surface_gap=min(
            item.minimum_committed_surface_gap for item in inspections
        ),
        final_local_knudsen_p05=float(
            np.quantile(final_local_knudsen, 0.05)
        ),
        final_local_knudsen_median=float(
            np.median(final_local_knudsen)
        ),
        final_local_knudsen_p95=float(
            np.quantile(final_local_knudsen, 0.95)
        ),
        field_metrics=field_metrics,
        wave_metrics=wave_metrics,
        wave_error=wave_error,
        source_hashes=tuple(hashes),
    )


def field_rows(
    analyses: list[CellAnalysis],
) -> list[dict[str, Any]]:
    rows = []
    for analysis in analyses:
        cell = analysis.cell
        for field in FIELDS:
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "particle_count": cell.particle_count,
                    "target_knudsen": cell.target_knudsen,
                    "particle_diameter": cell.particle_diameter,
                    "seed_count": analysis.seed_count,
                    "field": field,
                    **analysis.field_metrics[field],
                }
            )
    return rows


def wave_rows(
    analyses: list[CellAnalysis],
) -> list[dict[str, Any]]:
    rows = []
    for analysis in analyses:
        if analysis.wave_metrics is None:
            continue
        cell = analysis.cell
        for feature in FEATURE_LABELS:
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "particle_count": cell.particle_count,
                    "target_knudsen": cell.target_knudsen,
                    "particle_diameter": cell.particle_diameter,
                    "seed_count": analysis.seed_count,
                    "feature": feature,
                    **analysis.wave_metrics[feature],
                }
            )
    return rows


def cell_rows(
    analyses: list[CellAnalysis],
) -> list[dict[str, Any]]:
    return [
        {
            "cell_id": analysis.cell.cell_id,
            "particle_count": analysis.cell.particle_count,
            "target_knudsen": analysis.cell.target_knudsen,
            "realized_initial_reference_knudsen": (
                analysis.cell.realized_knudsen
            ),
            "particle_diameter": analysis.cell.particle_diameter,
            "transverse_area": analysis.cell.transverse_area,
            "maximum_initial_packing_fraction": (
                analysis.cell.maximum_initial_packing_fraction
            ),
            "transverse_mean_free_path_ratio": (
                analysis.cell.transverse_mean_free_path_ratio
            ),
            "seed_count": analysis.seed_count,
            "final_time": analysis.final_time,
            "maximum_absolute_relative_energy_drift": (
                analysis.maximum_energy_drift
            ),
            "minimum_committed_surface_gap": (
                analysis.minimum_surface_gap
            ),
            "final_local_knudsen_p05": (
                analysis.final_local_knudsen_p05
            ),
            "final_local_knudsen_median": (
                analysis.final_local_knudsen_median
            ),
            "final_local_knudsen_p95": (
                analysis.final_local_knudsen_p95
            ),
            "wave_status": (
                "not_requested"
                if analysis.wave_metrics is None
                and analysis.wave_error is None
                else "failed"
                if analysis.wave_error is not None
                else "warning"
                if analysis.wave_metrics is not None
                and any(
                    not bool(
                        metrics["full_ensemble_marker_check_passed"]
                    )
                    for metrics in analysis.wave_metrics.values()
                )
                else "completed"
            ),
        }
        for analysis in analyses
    ]


def trend_rows(
    scan: ScanConfig,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    metrics = (
        "relative_l2_mean_vs_euler",
        "relative_rms_ci95_half_width",
    )
    for knudsen in scan.knudsen_numbers:
        for field in FIELDS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if math.isclose(
                        float(row["target_knudsen"]),
                        knudsen,
                        rel_tol=0.0,
                        abs_tol=1.0e-14,
                    )
                    and row["field"] == field
                ),
                key=lambda row: int(row["particle_count"]),
            )
            if len(selected) < 2:
                continue
            x = np.array(
                [float(row["particle_count"]) for row in selected]
            )
            for metric in metrics:
                values = [float(row[metric]) for row in selected]
                result.append(
                    {
                        "scan_axis": "particle_count",
                        "fixed_value": knudsen,
                        "field": field,
                        "metric": metric,
                        "sample_count": len(values),
                        "log_log_slope": log_log_slope(
                            x,
                            np.array(values),
                        ),
                        "direction_with_increasing_axis": (
                            monotonic_direction(values)
                        ),
                        "first_value": values[0],
                        "last_value": values[-1],
                        "last_over_first": values[-1] / values[0],
                    }
                )
    for particle_count in scan.particle_counts:
        for field in FIELDS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if int(row["particle_count"]) == particle_count
                    and row["field"] == field
                ),
                key=lambda row: float(row["target_knudsen"]),
            )
            if len(selected) < 2:
                continue
            x = np.array(
                [float(row["target_knudsen"]) for row in selected]
            )
            for metric in metrics:
                values = [float(row[metric]) for row in selected]
                result.append(
                    {
                        "scan_axis": "knudsen",
                        "fixed_value": particle_count,
                        "field": field,
                        "metric": metric,
                        "sample_count": len(values),
                        "log_log_slope": log_log_slope(
                            x,
                            np.array(values),
                        ),
                        "direction_with_increasing_axis": (
                            monotonic_direction(values)
                        ),
                        "first_value": values[0],
                        "last_value": values[-1],
                        "last_over_first": values[-1] / values[0],
                    }
                )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        "yes"
                        if value is True
                        else "no"
                        if value is False
                        else value
                    )
                    for key, value in row.items()
                }
            )


def plot_field_scan(
    path: Path,
    scan: ScanConfig,
    rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 7.5),
        facecolor=BACKGROUND,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.87,
        bottom=0.09,
        hspace=0.3,
        wspace=0.22,
    )
    for axis, field in zip(axes.flat, FIELDS, strict=True):
        style_axis(axis)
        for index, particle_count in enumerate(scan.particle_counts):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["field"] == field
                    and int(row["particle_count"]) == particle_count
                ),
                key=lambda row: float(row["target_knudsen"]),
            )
            if not selected:
                continue
            x = [float(row["target_knudsen"]) for row in selected]
            error = [
                float(row["relative_l2_mean_vs_euler"])
                for row in selected
            ]
            uncertainty = [
                float(row["relative_rms_ci95_half_width"])
                for row in selected
            ]
            color = COLORS[index % len(COLORS)]
            axis.plot(
                x,
                error,
                color=color,
                marker="o",
                linewidth=1.8,
                label=f"N={particle_count:,}  error",
            )
            axis.plot(
                x,
                uncertainty,
                color=color,
                linestyle="--",
                marker=".",
                linewidth=1.2,
                alpha=0.8,
                label=f"N={particle_count:,}  CI half-width",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        color_tick_labels(axis)
        axis.set_xlabel("target Knudsen number")
        axis.set_ylabel("relative RMS metric")
        axis.set_title(FIELD_LABELS[field], color=FOREGROUND)
        axis.legend(
            facecolor=PANEL,
            edgecolor=GRID,
            labelcolor=FOREGROUND,
            fontsize=7,
            ncol=2,
        )
    figure.suptitle(
        "Particle/Euler error and independent-seed uncertainty",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def plot_uncertainty_scaling(
    path: Path,
    scan: ScanConfig,
    rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 7.5),
        facecolor=BACKGROUND,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.87,
        bottom=0.09,
        hspace=0.3,
        wspace=0.22,
    )
    for axis, field in zip(axes.flat, FIELDS, strict=True):
        style_axis(axis)
        for index, knudsen in enumerate(scan.knudsen_numbers):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["field"] == field
                    and math.isclose(
                        float(row["target_knudsen"]),
                        knudsen,
                        rel_tol=0.0,
                        abs_tol=1.0e-14,
                    )
                ),
                key=lambda row: int(row["particle_count"]),
            )
            if not selected:
                continue
            x = np.array(
                [float(row["particle_count"]) for row in selected]
            )
            y = np.array(
                [
                    float(row["relative_rms_ci95_half_width"])
                    for row in selected
                ]
            )
            slope = log_log_slope(x, y) if len(x) >= 2 else math.nan
            axis.plot(
                x,
                y,
                color=COLORS[index % len(COLORS)],
                marker="o",
                linewidth=1.8,
                label=f"Kn={knudsen:g}, slope={slope:.2f}",
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        color_tick_labels(axis)
        axis.set_xlabel("particle count (transverse area scaled)")
        axis.set_ylabel("relative RMS 95% CI half-width")
        axis.set_title(FIELD_LABELS[field], color=FOREGROUND)
        axis.legend(
            facecolor=PANEL,
            edgecolor=GRID,
            labelcolor=FOREGROUND,
            fontsize=8,
        )
    figure.suptitle(
        "Finite-sample uncertainty scaling at fixed Kn",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def plot_wave_scan(
    path: Path,
    scan: ScanConfig,
    rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(13.5, 4.8),
        facecolor=BACKGROUND,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.82,
        bottom=0.15,
        wspace=0.25,
    )
    for axis, feature in zip(axes, FEATURE_LABELS, strict=True):
        style_axis(axis)
        euler_values = []
        for index, particle_count in enumerate(scan.particle_counts):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["feature"] == feature
                    and int(row["particle_count"]) == particle_count
                ),
                key=lambda row: float(row["target_knudsen"]),
            )
            if not selected:
                continue
            x = np.array(
                [float(row["target_knudsen"]) for row in selected]
            )
            speed = np.array(
                [float(row["full_ensemble_speed"]) for row in selected]
            )
            half = np.array(
                [float(row["ci95_half_width"]) for row in selected]
            )
            euler_values.extend(float(row["euler_speed"]) for row in selected)
            axis.errorbar(
                x,
                speed,
                yerr=half,
                color=COLORS[index % len(COLORS)],
                marker="o",
                capsize=4,
                linewidth=1.6,
                label=f"N={particle_count:,}",
            )
        if euler_values:
            axis.axhline(
                float(np.mean(euler_values)),
                color=EULER_COLOR,
                linestyle="--",
                linewidth=1.5,
                label="ideal Euler",
            )
        axis.set_xscale("log")
        color_tick_labels(axis)
        axis.set_xlabel("target Knudsen number")
        axis.set_ylabel("marker speed")
        axis.set_title(FEATURE_LABELS[feature], color=FOREGROUND)
        axis.legend(
            facecolor=PANEL,
            edgecolor=GRID,
            labelcolor=FOREGROUND,
            fontsize=8,
        )
    figure.suptitle(
        "Wave-marker speed across particle count and Knudsen number",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def write_summary(
    path: Path,
    scan: ScanConfig,
    scan_output: Path,
    analyses: list[CellAnalysis],
    missing: list[str],
    trends: list[dict[str, Any]],
    wave_config: Path | None,
) -> None:
    wave_failures = [
        {
            "cell_id": analysis.cell.cell_id,
            "error": analysis.wave_error,
        }
        for analysis in analyses
        if analysis.wave_error is not None
    ]
    wave_marker_warnings = [
        {
            "cell_id": analysis.cell.cell_id,
            "feature": feature,
        }
        for analysis in analyses
        if analysis.wave_metrics is not None
        for feature, metrics in analysis.wave_metrics.items()
        if not bool(metrics["full_ensemble_marker_check_passed"])
    ]
    status = (
        "partial"
        if missing
        else "completed_with_wave_warnings"
        if wave_failures or wave_marker_warnings
        else "completed"
    )
    document = {
        "schema_version": 1,
        "experiment": "shock_tube_n_kn_scan_analysis",
        "status": status,
        "method": {
            "particle_count_axis": (
                "transverse periodic area scaled in proportion to N; "
                "local number density and target Kn are invariant"
            ),
            "knudsen_definition": (
                "1 / (sqrt(2) * pi * n_reference * d^2 * L_macro)"
            ),
            "density_reference": scan.density_reference,
            "characteristic_length": scan.characteristic_length,
            "field_uncertainty": (
                "two-sided Student-t interval for independent-seed mean"
            ),
            "wave_uncertainty": (
                "delete-one-seed jackknife on the seed-mean field"
                if wave_config is not None
                else None
            ),
            "analysis_tool_sha256": sha256_file(Path(__file__)),
        },
        "inputs": {
            "scan_config": {
                "path": portable_path(scan.path),
                "sha256": sha256_file(scan.path),
            },
            "wave_config": (
                {
                    "path": portable_path(wave_config),
                    "sha256": sha256_file(wave_config),
                }
                if wave_config is not None
                else None
            ),
            "scan_output": portable_path(scan_output),
        },
        "configured_cells": (
            len(scan.particle_counts) * len(scan.knudsen_numbers)
        ),
        "analyzed_cells": len(analyses),
        "missing_cells": missing,
        "wave_failures": wave_failures,
        "wave_marker_warnings": wave_marker_warnings,
        "cells": [
            {
                "cell_id": analysis.cell.cell_id,
                "particle_count": analysis.cell.particle_count,
                "target_knudsen": analysis.cell.target_knudsen,
                "particle_diameter": analysis.cell.particle_diameter,
                "maximum_initial_packing_fraction": (
                    analysis.cell.maximum_initial_packing_fraction
                ),
                "transverse_mean_free_path_ratio": (
                    analysis.cell.transverse_mean_free_path_ratio
                ),
                "seed_count": analysis.seed_count,
                "final_time": analysis.final_time,
                "maximum_absolute_relative_energy_drift": (
                    analysis.maximum_energy_drift
                ),
                "minimum_committed_surface_gap": (
                    analysis.minimum_surface_gap
                ),
                "final_local_knudsen": {
                    "p05": analysis.final_local_knudsen_p05,
                    "median": analysis.final_local_knudsen_median,
                    "p95": analysis.final_local_knudsen_p95,
                },
                "fields": analysis.field_metrics,
                "waves": analysis.wave_metrics,
                "wave_error": analysis.wave_error,
                "sources": list(analysis.source_hashes),
            }
            for analysis in analyses
        ],
        "trends": trends,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_self_test() -> int:
    x = np.array([1.0, 4.0, 16.0])
    y = np.array([1.0, 0.5, 0.25])
    slope = log_log_slope(x, y)
    if not math.isclose(slope, -0.5, abs_tol=1.0e-14):
        raise AssertionError("log-log slope self-test failed")
    if monotonic_direction([3.0, 2.0, 2.0]) != "nonincreasing":
        raise AssertionError("monotonic direction self-test failed")
    if monotonic_direction([1.0, 3.0, 2.0]) != "nonmonotonic":
        raise AssertionError("nonmonotonic direction self-test failed")
    print("particle-count/Knudsen scan analysis self-test")
    print(f"  synthetic N^(-1/2) slope: {slope:.12g}")
    print("  status: passed")
    return 0


def main() -> int:
    arguments = parse_arguments()
    if arguments.self_test:
        return run_self_test()
    if arguments.config is None or arguments.scan_output is None:
        raise ValueError(
            "config and scan_output are required unless --self-test is used"
        )
    scan = load_scan_config(arguments.config)
    cells = derive_cells(scan)
    scan_output = arguments.scan_output.resolve()
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else scan_output / "analysis"
    )
    output.mkdir(parents=True, exist_ok=True)
    wave_config = (
        arguments.wave_config.resolve()
        if arguments.wave_config is not None
        else None
    )

    analyses = []
    missing = []
    for cell in cells:
        cell_output = scan_output / "cells" / cell.cell_id
        if not cell_is_complete(scan, cell, cell_output):
            missing.append(cell.cell_id)
            continue
        print(
            f"analyzing {cell.cell_id} "
            f"(N={cell.particle_count}, Kn={cell.target_knudsen:g})"
        )
        analyses.append(
            analyze_cell(scan, cell, cell_output, wave_config)
        )
    if missing and not arguments.allow_incomplete:
        raise ValueError(
            "incomplete scan cells: " + ", ".join(missing)
        )
    if not analyses:
        raise ValueError("no complete scan cells are available")

    fields = field_rows(analyses)
    waves = wave_rows(analyses)
    cells_summary = cell_rows(analyses)
    trends = trend_rows(scan, fields)
    cells_path = output / "cell-summary.csv"
    fields_path = output / "field-scan.csv"
    waves_path = output / "wave-scan.csv"
    trends_path = output / "trend-summary.csv"
    summary_path = output / "summary.json"
    field_plot_path = output / "field-error-scan.png"
    uncertainty_plot_path = output / "uncertainty-scaling.png"
    wave_plot_path = output / "wave-speed-scan.png"
    write_csv(cells_path, cells_summary)
    write_csv(fields_path, fields)
    if waves:
        write_csv(waves_path, waves)
    write_csv(trends_path, trends)
    write_summary(
        summary_path,
        scan,
        scan_output,
        analyses,
        missing,
        trends,
        wave_config,
    )
    plot_field_scan(field_plot_path, scan, fields)
    plot_uncertainty_scaling(
        uncertainty_plot_path,
        scan,
        fields,
    )
    if waves:
        plot_wave_scan(wave_plot_path, scan, waves)

    print("particle-count/Knudsen scan analysis")
    print(f"  analyzed cells: {len(analyses)} / {len(cells)}")
    print(f"  missing cells: {len(missing)}")
    if wave_config is not None:
        print(
            "  wave extraction failures: "
            f"{sum(item.wave_error is not None for item in analyses)}"
        )
        marker_warnings = sum(
            not bool(metrics["full_ensemble_marker_check_passed"])
            for item in analyses
            if item.wave_metrics is not None
            for metrics in item.wave_metrics.values()
        )
        print(f"  wave marker warnings: {marker_warnings}")
    else:
        print("  wave extraction: not requested")
    print(f"  cell summary: {cells_path}")
    print(f"  field table: {fields_path}")
    if waves:
        print(f"  wave table: {waves_path}")
    print(f"  trend table: {trends_path}")
    print(f"  summary JSON: {summary_path}")
    print(f"  field plot: {field_plot_path}")
    print(f"  uncertainty plot: {uncertainty_plot_path}")
    if waves:
        print(f"  wave plot: {wave_plot_path}")
    diagnostic_warning = any(
        item.wave_error is not None
        or (
            item.wave_metrics is not None
            and any(
                not bool(
                    metrics["full_ensemble_marker_check_passed"]
                )
                for metrics in item.wave_metrics.values()
            )
        )
        for item in analyses
    )
    status = (
        "partial"
        if missing
        else "completed with wave warnings"
        if diagnostic_warning
        else "completed"
    )
    print(f"  status: {status}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, csv.Error) as exception:
        raise SystemExit(f"N/Kn scan analysis: {exception}") from exception
