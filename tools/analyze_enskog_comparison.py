#!/usr/bin/env python3
"""Compare particle shock tubes with ideal- and Enskog-Euler references.

The Enskog-side pressure uses the Carnahan--Starling hard-sphere equilibrium
closure.  This analyzer deliberately keeps it separate from the frozen ideal
Euler analysis and reports velocity errors with several normalizations so that
a small reference-field RMS cannot masquerade as a large physical mismatch.
"""

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

from analyze_multiseed_fields import (  # noqa: E402
    FIELDS,
    FIELD_LABELS,
    confidence_statistics,
    euler_at_particle_grid,
    load_ensemble,
    portable_path,
    sha256_file,
    student_t_critical_95,
)
from analyze_n_kn_scan import cell_is_complete  # noqa: E402
from enskog_reference import (  # noqa: E402
    EnskogEulerParameters,
    carnahan_starling_compressibility,
    enskog_pressure,
    packing_fraction,
    solve_enskog_euler_reference,
)
from run_multiseed_shock_tube import load_batch_config  # noqa: E402
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
PARTICLE_COLOR = "#38bdf8"
IDEAL_COLOR = "#f59e0b"
ENSKOG_COLOR = "#22c55e"
KINETIC_COLOR = "#a78bfa"
COUNT_COLORS = ("#38bdf8", "#a78bfa", "#22c55e", "#f97316")


@dataclass(frozen=True)
class ComparisonCell:
    cell: ScanCell
    x: np.ndarray
    particle_mean: np.ndarray
    particle_half_width: np.ndarray
    particle_enskog_mean: np.ndarray
    particle_enskog_half_width: np.ndarray
    ideal_reference: np.ndarray
    enskog_reference: np.ndarray
    field_rows: tuple[dict[str, Any], ...]
    velocity_rows: tuple[dict[str, Any], ...]
    enskog_steps: int
    enskog_mass_drift: float
    enskog_energy_drift: float
    enskog_maximum_packing_fraction: float
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
        help="output directory; defaults to SCAN_OUTPUT/enskog-comparison",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate closure and velocity-diagnostic arithmetic",
    )
    return parser.parse_args()


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=FOREGROUND, which="both")
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.grid(color=GRID, alpha=0.4, linewidth=0.8)
    axis.xaxis.label.set_color(FOREGROUND)
    axis.yaxis.label.set_color(FOREGROUND)
    axis.title.set_color(FOREGROUND)


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def field_metric(
    particle: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    tiny = np.finfo(np.float64).tiny
    difference = particle - reference
    reference_rms = rms(reference)
    reference_peak = float(np.max(np.abs(reference)))
    difference_rms = rms(difference)
    centered_particle = particle - np.mean(particle)
    centered_reference = reference - np.mean(reference)
    correlation_scale = float(
        np.linalg.norm(centered_particle) * np.linalg.norm(centered_reference)
    )
    correlation = (
        float(np.dot(centered_particle, centered_reference) / correlation_scale)
        if correlation_scale > tiny
        else math.nan
    )
    return {
        "rms_difference": difference_rms,
        "reference_rms": reference_rms,
        "reference_peak_absolute": reference_peak,
        "relative_l2": difference_rms / max(reference_rms, tiny),
        "rms_difference_over_reference_peak": (
            difference_rms / max(reference_peak, tiny)
        ),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "correlation": correlation,
    }


def shifted_reference(
    x: np.ndarray,
    reference: np.ndarray,
    shift: float,
) -> np.ndarray:
    return np.interp(
        x - shift,
        x,
        reference,
        left=float(reference[0]),
        right=float(reference[-1]),
    )


def velocity_diagnostics(
    x: np.ndarray,
    particle: np.ndarray,
    reference: np.ndarray,
    *,
    active_threshold_fraction: float = 0.05,
    maximum_shift_fraction: float = 0.0625,
) -> dict[str, float]:
    metrics = field_metric(particle, reference)
    tiny = np.finfo(np.float64).tiny
    peak = metrics["reference_peak_absolute"]
    active = np.abs(reference) >= active_threshold_fraction * peak
    if not np.any(active):
        active = np.ones_like(reference, dtype=bool)
    difference = particle - reference
    squared_error = np.square(difference)
    total_squared_error = float(np.sum(squared_error))
    active_reference_rms = rms(reference[active])
    active_difference_rms = rms(difference[active])
    quiet_error_fraction = (
        float(np.sum(squared_error[~active]) / total_squared_error)
        if total_squared_error > tiny
        else 0.0
    )

    dx = float(np.median(np.diff(x)))
    length = float(x[-1] - x[0] + dx)
    maximum_lag = max(1, int(round(maximum_shift_fraction * length / dx)))
    candidates = range(-maximum_lag, maximum_lag + 1)
    best_lag = min(
        candidates,
        key=lambda lag: rms(
            particle - shifted_reference(x, reference, lag * dx)
        ),
    )
    best_shift = best_lag * dx
    aligned = shifted_reference(x, reference, best_shift)
    aligned_difference_rms = rms(particle - aligned)
    aligned_norm = float(np.dot(aligned, aligned))
    amplitude = (
        float(np.dot(particle, aligned) / aligned_norm)
        if aligned_norm > tiny
        else math.nan
    )
    aligned_scaled = amplitude * aligned
    metrics.update(
        {
            "active_threshold_fraction_of_peak": active_threshold_fraction,
            "active_slab_fraction": float(np.mean(active)),
            "active_relative_l2": (
                active_difference_rms / max(active_reference_rms, tiny)
            ),
            "quiet_fraction_of_squared_error": quiet_error_fraction,
            "optimal_shift_slabs": best_lag,
            "optimal_shift_x": best_shift,
            "shift_aligned_relative_l2": (
                aligned_difference_rms / max(rms(aligned), tiny)
            ),
            "optimal_amplitude_scale": amplitude,
            "shift_and_amplitude_aligned_relative_l2": (
                rms(particle - aligned_scaled) / max(rms(aligned_scaled), tiny)
            ),
        }
    )
    return metrics


def interpolate_solution(
    x: np.ndarray,
    frames: list[Any],
) -> np.ndarray:
    values = np.empty((len(frames), len(x), len(FIELDS)), dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        reference_x = frame.fields["x_center"]
        for field_index, field in enumerate(FIELDS):
            values[frame_index, :, field_index] = np.interp(
                x,
                reference_x,
                frame.fields[field],
            )
    return values


def enskog_particle_statistics(
    ensemble: Any,
    diameter: float,
    boltzmann_constant: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = ensemble.values.copy()
    values[..., 3] = enskog_pressure(
        values[..., 0],
        values[..., 2],
        diameter,
        boltzmann_constant,
    )
    mean = np.mean(values, axis=0)
    standard_deviation = np.std(values, axis=0, ddof=1)
    critical = student_t_critical_95(len(ensemble.seeds) - 1)
    half_width = critical * standard_deviation / math.sqrt(len(ensemble.seeds))
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(half_width)):
        raise ValueError("Enskog-closed particle statistics are non-finite")
    return mean, half_width


def analyze_cell(
    scan: ScanConfig,
    cell: ScanCell,
    cell_output: Path,
) -> ComparisonCell:
    batch = load_batch_config(cell_output / "batch-input.json")
    effective_config = cell_output / "effective-config.json"
    with effective_config.open(encoding="utf-8") as stream:
        effective = json.load(stream)
    boltzmann_constant = float(effective["units"]["boltzmann_constant"])
    ensemble = load_ensemble(batch, cell_output)
    particle_mean, _, particle_half_width, _ = confidence_statistics(ensemble)
    ideal_reference, _ = euler_at_particle_grid(
        effective_config,
        batch,
        ensemble,
    )
    enskog_solution = solve_enskog_euler_reference(
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
        particle_diameter=cell.particle_diameter,
        parameters=EnskogEulerParameters(cells=batch.euler_cells),
    )
    enskog_reference = interpolate_solution(ensemble.x, enskog_solution.frames)
    particle_enskog_mean, particle_enskog_half_width = (
        enskog_particle_statistics(
            ensemble,
            cell.particle_diameter,
            boltzmann_constant,
        )
    )
    final = len(ensemble.frames) - 1
    field_rows: list[dict[str, Any]] = []
    velocity_rows: list[dict[str, Any]] = []
    for (
        model,
        particle_values,
        particle_bands,
        reference_values,
        pressure_definition,
    ) in (
        (
            "ideal_euler",
            particle_mean,
            particle_half_width,
            ideal_reference,
            "kinetic_n_kT",
        ),
        (
            "enskog_euler_carnahan_starling",
            particle_enskog_mean,
            particle_enskog_half_width,
            enskog_reference,
            "carnahan_starling_n_kT_Z",
        ),
    ):
        for field_index, field in enumerate(FIELDS):
            metrics = field_metric(
                particle_values[final, :, field_index],
                reference_values[final, :, field_index],
            )
            band_rms = rms(particle_bands[final, :, field_index])
            field_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "particle_count": cell.particle_count,
                    "target_knudsen": cell.target_knudsen,
                    "maximum_initial_packing_fraction": (
                        cell.maximum_initial_packing_fraction
                    ),
                    "reference_model": model,
                    "particle_pressure_definition": pressure_definition,
                    "field": field,
                    **metrics,
                    "ci95_half_width_rms": band_rms,
                    "relative_ci95_half_width_rms": (
                        band_rms
                        / max(
                            metrics["reference_rms"],
                            np.finfo(np.float64).tiny,
                        )
                    ),
                }
            )
        velocity_metrics = velocity_diagnostics(
            ensemble.x,
            particle_mean[final, :, 1],
            reference_values[final, :, 1],
        )
        velocity_band_rms = rms(particle_half_width[final, :, 1])
        velocity_rows.append(
            {
                "cell_id": cell.cell_id,
                "particle_count": cell.particle_count,
                "target_knudsen": cell.target_knudsen,
                "maximum_initial_packing_fraction": (
                    cell.maximum_initial_packing_fraction
                ),
                "reference_model": model,
                **velocity_metrics,
                "ci95_half_width_rms": velocity_band_rms,
                "relative_ci95_half_width_rms": (
                    velocity_band_rms
                    / max(
                        velocity_metrics["reference_rms"],
                        np.finfo(np.float64).tiny,
                    )
                ),
            }
        )
    source_hashes = tuple(
        {
            "seed": seed,
            "macroscopic_sha256": sha256_file(
                cell_output / f"seed-{seed}" / "macroscopic.csv"
            ),
        }
        for seed in ensemble.seeds
    )
    return ComparisonCell(
        cell=cell,
        x=ensemble.x,
        particle_mean=particle_mean,
        particle_half_width=particle_half_width,
        particle_enskog_mean=particle_enskog_mean,
        particle_enskog_half_width=particle_enskog_half_width,
        ideal_reference=ideal_reference,
        enskog_reference=enskog_reference,
        field_rows=tuple(field_rows),
        velocity_rows=tuple(velocity_rows),
        enskog_steps=enskog_solution.steps,
        enskog_mass_drift=enskog_solution.relative_mass_drift,
        enskog_energy_drift=enskog_solution.relative_energy_drift,
        enskog_maximum_packing_fraction=(
            enskog_solution.maximum_packing_fraction
        ),
        source_hashes=source_hashes,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_field_comparison(
    path: Path,
    scan: ScanConfig,
    analyses: list[ComparisonCell],
) -> None:
    largest = max(scan.particle_counts)
    selected = sorted(
        (item for item in analyses if item.cell.particle_count == largest),
        key=lambda item: item.cell.target_knudsen,
    )
    figure, axes = plt.subplots(
        len(selected),
        len(FIELDS),
        figsize=(16.0, 10.0),
        facecolor=BACKGROUND,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.06,
        right=0.985,
        top=0.91,
        bottom=0.075,
        hspace=0.34,
        wspace=0.25,
    )
    final = -1
    for row_index, analysis in enumerate(selected):
        for field_index, field in enumerate(FIELDS):
            axis = axes[row_index, field_index]
            style_axis(axis)
            if field == "pressure":
                particle = analysis.particle_enskog_mean[final, :, field_index]
                half = analysis.particle_enskog_half_width[final, :, field_index]
                axis.plot(
                    analysis.x,
                    analysis.particle_mean[final, :, field_index],
                    color=KINETIC_COLOR,
                    linewidth=1.0,
                    alpha=0.85,
                    label="particle kinetic n kT",
                )
                particle_label = "particle + CS closure"
            else:
                particle = analysis.particle_mean[final, :, field_index]
                half = analysis.particle_half_width[final, :, field_index]
                particle_label = "particle mean"
            axis.fill_between(
                analysis.x,
                particle - half,
                particle + half,
                color=PARTICLE_COLOR,
                alpha=0.18,
                linewidth=0.0,
            )
            axis.plot(
                analysis.x,
                particle,
                color=PARTICLE_COLOR,
                linewidth=1.5,
                label=particle_label,
            )
            axis.plot(
                analysis.x,
                analysis.ideal_reference[final, :, field_index],
                color=IDEAL_COLOR,
                linestyle="--",
                linewidth=1.4,
                label="ideal Euler",
            )
            axis.plot(
                analysis.x,
                analysis.enskog_reference[final, :, field_index],
                color=ENSKOG_COLOR,
                linestyle="-.",
                linewidth=1.5,
                label="Enskog-Euler (CS)",
            )
            if row_index == 0:
                axis.set_title(FIELD_LABELS[field], fontsize=12)
            if field_index == 0:
                axis.set_ylabel(
                    f"Kn={analysis.cell.target_knudsen:g}\nfield value"
                )
            if row_index == len(selected) - 1:
                axis.set_xlabel("x")
            if row_index == 0 and field_index == 0:
                axis.legend(
                    facecolor=PANEL,
                    edgecolor=GRID,
                    labelcolor=FOREGROUND,
                    fontsize=7,
                )
    figure.suptitle(
        f"Particle, ideal Euler, and Enskog-Euler at t=0.20; N={largest:,}",
        color=FOREGROUND,
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def plot_velocity_diagnostics(
    path: Path,
    scan: ScanConfig,
    rows: list[dict[str, Any]],
) -> None:
    metrics = (
        ("relative_l2", "domain relative L2"),
        (
            "rms_difference_over_reference_peak",
            "RMS difference / reference peak",
        ),
        ("active_relative_l2", "active-region relative L2"),
        ("shift_aligned_relative_l2", "shift-aligned relative L2"),
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 8.0),
        facecolor=BACKGROUND,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.88,
        bottom=0.09,
        hspace=0.3,
        wspace=0.24,
    )
    for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
        style_axis(axis)
        for count_index, particle_count in enumerate(scan.particle_counts):
            color = COUNT_COLORS[count_index % len(COUNT_COLORS)]
            for model, linestyle, marker in (
                ("ideal_euler", "--", "o"),
                ("enskog_euler_carnahan_starling", "-", "s"),
            ):
                selected = sorted(
                    (
                        row
                        for row in rows
                        if int(row["particle_count"]) == particle_count
                        and row["reference_model"] == model
                    ),
                    key=lambda row: float(row["target_knudsen"]),
                )
                axis.plot(
                    [float(row["target_knudsen"]) for row in selected],
                    [float(row[metric]) for row in selected],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.5,
                    label=(
                        f"N={particle_count:,}, "
                        f"{'ideal' if model == 'ideal_euler' else 'Enskog'}"
                    ),
                )
        axis.set_xscale("log")
        axis.set_xlabel("target Knudsen number")
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.legend(
            facecolor=PANEL,
            edgecolor=GRID,
            labelcolor=FOREGROUND,
            fontsize=7,
            ncol=2,
        )
    figure.suptitle(
        "Why the velocity relative error looks large",
        color=FOREGROUND,
        fontsize=17,
        fontweight="bold",
        y=0.96,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def median_metric(
    rows: list[dict[str, Any]], model: str, metric: str
) -> float:
    return float(
        np.median(
            [float(row[metric]) for row in rows if row["reference_model"] == model]
        )
    )


def reference_grid_check(
    scan: ScanConfig,
    scan_output: Path,
    analyses: list[ComparisonCell],
) -> list[dict[str, Any]]:
    cell = next(
        item.cell
        for item in analyses
        if item.cell.particle_count == max(scan.particle_counts)
        and math.isclose(
            item.cell.target_knudsen,
            min(scan.knudsen_numbers),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
    )
    analysis = next(item for item in analyses if item.cell == cell)
    cell_output = scan_output / "cells" / cell.cell_id
    batch = load_batch_config(cell_output / "batch-input.json")
    effective_config = cell_output / "effective-config.json"
    with effective_config.open(encoding="utf-8") as stream:
        effective = json.load(stream)
    end_time = float(effective["simulation"]["end_time"])
    frame_interval = float(effective["simulation"]["frame_interval"])
    frame_count = int(round(end_time / frame_interval))
    frame_times = [
        (index, min(index * frame_interval, end_time))
        for index in range(frame_count + 1)
    ]
    refined = solve_enskog_euler_reference(
        effective_config,
        frame_times,
        particle_count=batch.particle_count,
        particle_diameter=cell.particle_diameter,
        parameters=EnskogEulerParameters(cells=2 * batch.euler_cells),
    )
    rows = []
    for field_index, field in enumerate(FIELDS):
        refined_values = np.interp(
            analysis.x,
            refined.frames[-1].fields["x_center"],
            refined.frames[-1].fields[field],
        )
        coarse_values = analysis.enskog_reference[-1, :, field_index]
        rows.append(
            {
                "cell_id": cell.cell_id,
                "particle_count": cell.particle_count,
                "target_knudsen": cell.target_knudsen,
                "field": field,
                "coarse_cells": batch.euler_cells,
                "refined_cells": 2 * batch.euler_cells,
                "relative_l2_coarse_vs_refined": (
                    rms(coarse_values - refined_values)
                    / max(rms(refined_values), np.finfo(np.float64).tiny)
                ),
                "refined_relative_mass_drift": refined.relative_mass_drift,
                "refined_relative_energy_drift": refined.relative_energy_drift,
            }
        )
    return rows


def build_summary(
    scan: ScanConfig,
    scan_output: Path,
    analyses: list[ComparisonCell],
    field_rows: list[dict[str, Any]],
    velocity_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    largest = max(scan.particle_counts)
    largest_rows = [
        row for row in field_rows if int(row["particle_count"]) == largest
    ]
    largest_table = []
    for knudsen in scan.knudsen_numbers:
        selected = [
            row
            for row in largest_rows
            if math.isclose(
                float(row["target_knudsen"]),
                knudsen,
                rel_tol=0.0,
                abs_tol=1.0e-14,
            )
        ]
        record: dict[str, Any] = {"target_knudsen": knudsen}
        for row in selected:
            prefix = (
                "ideal"
                if row["reference_model"] == "ideal_euler"
                else "enskog"
            )
            record[f"{row['field']}_{prefix}_relative_l2"] = row[
                "relative_l2"
            ]
        largest_table.append(record)
    ideal_relative = median_metric(velocity_rows, "ideal_euler", "relative_l2")
    ideal_peak = median_metric(
        velocity_rows,
        "ideal_euler",
        "rms_difference_over_reference_peak",
    )
    return {
        "schema_version": 1,
        "experiment": "shock_tube_enskog_euler_comparison",
        "status": "completed",
        "scope": (
            "inviscid hard-sphere hydrodynamic reference using the "
            "Carnahan--Starling equilibrium EOS; not a full Enskog kinetic "
            "or dissipative transport solver"
        ),
        "method": {
            "packing_fraction": "eta = pi * n * d^3 / 6",
            "compressibility_factor": (
                "Z = (1 + eta + eta^2 - eta^3) / (1 - eta)^3"
            ),
            "pressure": "p = n * k_B * T * Z",
            "particle_enskog_pressure": (
                "Carnahan--Starling closure applied to each seed before "
                "forming the independent-seed mean"
            ),
            "velocity_active_region": (
                "slabs with |u_reference| >= 0.05 max |u_reference|"
            ),
            "velocity_shift_search": (
                "integer slab shifts within 6.25% of the longitudinal domain"
            ),
            "analysis_tool_sha256": sha256_file(Path(__file__)),
            "reference_tool_sha256": sha256_file(
                Path(__file__).with_name("enskog_reference.py")
            ),
        },
        "inputs": {
            "scan_config": {
                "path": portable_path(scan.path),
                "sha256": sha256_file(scan.path),
            },
            "scan_output": portable_path(scan_output),
        },
        "analyzed_cells": len(analyses),
        "largest_particle_count": largest,
        "largest_particle_count_field_comparison": largest_table,
        "velocity_diagnosis": {
            "median_ideal_domain_relative_l2": ideal_relative,
            "median_ideal_rms_difference_over_reference_peak": ideal_peak,
            "normalization_ratio": ideal_peak / ideal_relative,
            "median_ideal_active_region_relative_l2": median_metric(
                velocity_rows, "ideal_euler", "active_relative_l2"
            ),
            "median_ideal_quiet_fraction_of_squared_error": median_metric(
                velocity_rows,
                "ideal_euler",
                "quiet_fraction_of_squared_error",
            ),
            "median_ideal_shift_aligned_relative_l2": median_metric(
                velocity_rows,
                "ideal_euler",
                "shift_aligned_relative_l2",
            ),
            "median_ideal_relative_ci95_half_width_rms": median_metric(
                velocity_rows,
                "ideal_euler",
                "relative_ci95_half_width_rms",
            ),
            "median_enskog_domain_relative_l2": median_metric(
                velocity_rows,
                "enskog_euler_carnahan_starling",
                "relative_l2",
            ),
            "median_enskog_rms_difference_over_reference_peak": median_metric(
                velocity_rows,
                "enskog_euler_carnahan_starling",
                "rms_difference_over_reference_peak",
            ),
            "median_enskog_active_region_relative_l2": median_metric(
                velocity_rows,
                "enskog_euler_carnahan_starling",
                "active_relative_l2",
            ),
            "median_enskog_shift_aligned_relative_l2": median_metric(
                velocity_rows,
                "enskog_euler_carnahan_starling",
                "shift_aligned_relative_l2",
            ),
            "median_enskog_relative_ci95_half_width_rms": median_metric(
                velocity_rows,
                "enskog_euler_carnahan_starling",
                "relative_ci95_half_width_rms",
            ),
        },
        "reference_grid_check": grid_rows,
        "cells": [
            {
                "cell_id": item.cell.cell_id,
                "particle_count": item.cell.particle_count,
                "target_knudsen": item.cell.target_knudsen,
                "particle_diameter": item.cell.particle_diameter,
                "maximum_initial_packing_fraction": (
                    item.cell.maximum_initial_packing_fraction
                ),
                "initial_maximum_compressibility_factor": float(
                    carnahan_starling_compressibility(
                        item.cell.maximum_initial_packing_fraction
                    )
                ),
                "enskog_reference_steps": item.enskog_steps,
                "enskog_reference_relative_mass_drift": item.enskog_mass_drift,
                "enskog_reference_relative_energy_drift": (
                    item.enskog_energy_drift
                ),
                "enskog_reference_maximum_packing_fraction": (
                    item.enskog_maximum_packing_fraction
                ),
                "sources": list(item.source_hashes),
            }
            for item in analyses
        ],
    }


def run_self_test() -> int:
    number_density = np.array([1.0, 2.0])
    temperature = np.array([1.0, 0.5])
    pressure = enskog_pressure(number_density, temperature, 0.2)
    kinetic = number_density * temperature
    if not np.all(pressure > kinetic):
        raise AssertionError("Enskog pressure must exceed kinetic pressure")
    x = np.linspace(0.0, 4.0, 256, endpoint=False) + 4.0 / 512.0
    reference = np.exp(-((x - 2.0) / 0.2) ** 2)
    dx = float(np.median(np.diff(x)))
    particle = shifted_reference(x, reference, 3.0 * dx)
    metrics = velocity_diagnostics(x, particle, reference)
    if int(metrics["optimal_shift_slabs"]) != 3:
        raise AssertionError("velocity shift diagnostic failed")
    if float(metrics["shift_aligned_relative_l2"]) > 1.0e-12:
        raise AssertionError("velocity shift alignment left a residual")
    eta = packing_fraction(number_density, 0.2)
    if not np.all(eta > 0.0):
        raise AssertionError("packing-fraction diagnostic failed")
    print("Enskog comparison analysis self-test")
    print(f"  recovered synthetic shift: {metrics['optimal_shift_slabs']} slabs")
    print(f"  aligned residual: {metrics['shift_aligned_relative_l2']:.3e}")
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
    scan_output = arguments.scan_output.resolve()
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else scan_output / "enskog-comparison"
    )
    output.mkdir(parents=True, exist_ok=True)
    analyses = []
    for cell in derive_cells(scan):
        cell_output = scan_output / "cells" / cell.cell_id
        if not cell_is_complete(scan, cell, cell_output):
            raise ValueError(f"incomplete scan cell: {cell.cell_id}")
        print(
            f"comparing {cell.cell_id} "
            f"(N={cell.particle_count}, Kn={cell.target_knudsen:g})"
        )
        analyses.append(analyze_cell(scan, cell, cell_output))
    field_rows = [row for item in analyses for row in item.field_rows]
    velocity_rows = [row for item in analyses for row in item.velocity_rows]
    grid_rows = reference_grid_check(scan, scan_output, analyses)
    field_path = output / "field-comparison.csv"
    velocity_path = output / "velocity-diagnostics.csv"
    grid_path = output / "reference-grid-check.csv"
    summary_path = output / "summary.json"
    field_plot_path = output / "enskog-field-comparison.png"
    velocity_plot_path = output / "velocity-error-diagnostics.png"
    write_csv(field_path, field_rows)
    write_csv(velocity_path, velocity_rows)
    write_csv(grid_path, grid_rows)
    summary = build_summary(
        scan,
        scan_output,
        analyses,
        field_rows,
        velocity_rows,
        grid_rows,
    )
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    plot_field_comparison(field_plot_path, scan, analyses)
    plot_velocity_diagnostics(velocity_plot_path, scan, velocity_rows)
    diagnosis = summary["velocity_diagnosis"]
    print("Enskog-Euler comparison")
    print(f"  analyzed cells: {len(analyses)}")
    print(
        "  median ideal velocity relative L2 / peak-normalized RMS: "
        f"{diagnosis['median_ideal_domain_relative_l2']:.6g} / "
        f"{diagnosis['median_ideal_rms_difference_over_reference_peak']:.6g}"
    )
    print(
        "  median Enskog velocity relative L2 / peak-normalized RMS: "
        f"{diagnosis['median_enskog_domain_relative_l2']:.6g} / "
        f"{diagnosis['median_enskog_rms_difference_over_reference_peak']:.6g}"
    )
    print(
        "  median ideal quiet-region squared-error fraction: "
        f"{diagnosis['median_ideal_quiet_fraction_of_squared_error']:.6g}"
    )
    print(f"  field table: {field_path}")
    print(f"  velocity diagnostics: {velocity_path}")
    print(f"  reference grid check: {grid_path}")
    print(f"  summary JSON: {summary_path}")
    print(f"  field plot: {field_plot_path}")
    print(f"  velocity plot: {velocity_plot_path}")
    print("  status: completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, csv.Error) as exception:
        raise SystemExit(f"Enskog comparison: {exception}") from exception
