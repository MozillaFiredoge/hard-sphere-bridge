#!/usr/bin/env python3
"""Compare Enskog transport ablations over every saved shock-tube frame.

This analysis reuses completed particle trajectories and solves only the
one-dimensional CPU references.  It tracks field error, velocity diagnostics,
and robust positive-pulse peak/width measurements from t>0 through the final
saved frame.
"""

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

from analyze_enskog_comparison import (  # noqa: E402
    BACKGROUND,
    FOREGROUND,
    GRID,
    PANEL,
    PARTICLE_COLOR,
    enskog_particle_statistics,
    field_metric,
    interpolate_solution,
    rms,
    style_axis,
    velocity_diagnostics,
)
from analyze_enskog_transport_ablation import (  # noqa: E402
    ABLATIONS,
    SPEC_BY_MODEL,
    annotate_baseline_change,
    parameters_for,
    solver_diagnostics,
)
from analyze_multiseed_fields import (  # noqa: E402
    FIELDS,
    FIELD_LABELS,
    confidence_statistics,
    load_ensemble,
    portable_path,
    sha256_file,
)
from analyze_n_kn_scan import cell_is_complete  # noqa: E402
from enskog_nsf_reference import (  # noqa: E402
    EnskogNsfSolution,
    solve_enskog_nsf_reference,
)
from run_multiseed_shock_tube import load_batch_config  # noqa: E402
from run_n_kn_scan import (  # noqa: E402
    ScanCell,
    ScanConfig,
    derive_cells,
    load_scan_config,
)


@dataclass(frozen=True)
class ParticleCell:
    cell: ScanCell
    x: np.ndarray
    frames: np.ndarray
    times: np.ndarray
    particle_mean: np.ndarray
    particle_half_width: np.ndarray
    particle_enskog_mean: np.ndarray
    particle_enskog_half_width: np.ndarray
    source_hashes: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EvolutionCell:
    particle: ParticleCell
    references: dict[str, np.ndarray]
    field_rows: tuple[dict[str, Any], ...]
    velocity_rows: tuple[dict[str, Any], ...]
    shape_rows: tuple[dict[str, Any], ...]


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
        help=(
            "output directory; defaults to "
            "SCAN_OUTPUT/enskog-transport-evolution"
        ),
    )
    parser.add_argument(
        "--nsf-cells",
        type=int,
        default=512,
        help="finite-volume cells shared by every reference (default: 512)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate pulse measurement and time-series arithmetic",
    )
    return parser.parse_args()


def load_particle_cell(
    cell: ScanCell,
    cell_output: Path,
) -> tuple[ParticleCell, Any, Any]:
    batch = load_batch_config(cell_output / "batch-input.json")
    ensemble = load_ensemble(batch, cell_output)
    mean, _, half_width, _ = confidence_statistics(ensemble)
    with (cell_output / "effective-config.json").open(
        encoding="utf-8"
    ) as stream:
        effective = json.load(stream)
    boltzmann_constant = float(effective["units"]["boltzmann_constant"])
    enskog_mean, enskog_half_width = enskog_particle_statistics(
        ensemble,
        cell.particle_diameter,
        boltzmann_constant,
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
    return (
        ParticleCell(
            cell=cell,
            x=ensemble.x,
            frames=ensemble.frames,
            times=ensemble.times,
            particle_mean=mean,
            particle_half_width=half_width,
            particle_enskog_mean=enskog_mean,
            particle_enskog_half_width=enskog_half_width,
            source_hashes=source_hashes,
        ),
        batch,
        ensemble,
    )


def frame_times(ensemble: Any) -> list[tuple[int, float]]:
    return [
        (int(frame), float(time))
        for frame, time in zip(ensemble.frames, ensemble.times, strict=True)
    ]


def positive_pulse_metrics(
    x: np.ndarray,
    velocity: np.ndarray,
    *,
    threshold_fraction: float = 0.05,
) -> dict[str, float]:
    if (
        x.ndim != 1
        or velocity.ndim != 1
        or len(x) != len(velocity)
        or len(x) < 3
    ):
        raise ValueError("pulse inputs must be matching one-dimensional arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(velocity)):
        raise ValueError("pulse inputs must be finite")
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("pulse threshold fraction must lie in (0, 1)")
    peak_index = int(np.argmax(velocity))
    peak = float(velocity[peak_index])
    if peak <= np.finfo(np.float64).tiny:
        raise ValueError("positive velocity pulse has no positive peak")
    threshold = threshold_fraction * peak
    start = peak_index
    while start > 0 and velocity[start - 1] >= threshold:
        start -= 1
    stop = peak_index + 1
    while stop < len(velocity) and velocity[stop] >= threshold:
        stop += 1
    selected_x = x[start:stop]
    weights = np.maximum(velocity[start:stop], 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= np.finfo(np.float64).tiny:
        raise ValueError("positive velocity pulse has zero weight")
    dx = float(np.median(np.diff(x)))
    centroid = float(np.sum(selected_x * weights) / weight_sum)
    variance = float(
        np.sum(np.square(selected_x - centroid) * weights) / weight_sum
    )
    return {
        "threshold_fraction_of_peak": threshold_fraction,
        "peak_velocity": peak,
        "peak_x": float(x[peak_index]),
        "centroid_x": centroid,
        "rms_width": math.sqrt(max(variance, 0.0)),
        "support_width": float((stop - start) * dx),
        "equivalent_width": float(weight_sum * dx / peak),
    }


def safe_fractional_difference(value: float, reference: float) -> float:
    scale = max(abs(reference), 1.0e-12)
    return (value - reference) / scale


def analyze_cell(
    cell: ScanCell,
    cell_output: Path,
    nsf_cells: int,
    cache: dict[tuple[float, str], EnskogNsfSolution],
) -> EvolutionCell:
    particle, batch, ensemble = load_particle_cell(cell, cell_output)
    references: dict[str, np.ndarray] = {}
    for spec in ABLATIONS:
        key = (cell.target_knudsen, spec.model)
        solution = cache.get(key)
        if solution is None:
            solution = solve_enskog_nsf_reference(
                cell_output / "effective-config.json",
                frame_times(ensemble),
                particle_count=batch.particle_count,
                particle_diameter=cell.particle_diameter,
                parameters=parameters_for(spec, nsf_cells),
            )
            cache[key] = solution
        references[spec.model] = interpolate_solution(
            particle.x,
            solution.frames,
        )

    field_rows: list[dict[str, Any]] = []
    velocity_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    for frame_index in range(1, len(particle.frames)):
        frame = int(particle.frames[frame_index])
        time = float(particle.times[frame_index])
        particle_shape = positive_pulse_metrics(
            particle.x,
            particle.particle_mean[frame_index, :, 1],
        )
        shape_rows.append(
            {
                "cell_id": cell.cell_id,
                "particle_count": cell.particle_count,
                "target_knudsen": cell.target_knudsen,
                "frame": frame,
                "time": time,
                "model": "particle_mean",
                "model_label": "particle mean",
                "shear_viscosity_scale": None,
                "bulk_viscosity_scale": None,
                "thermal_conductivity_scale": None,
                **particle_shape,
                "peak_fractional_difference_vs_particle": 0.0,
                "rms_width_fractional_difference_vs_particle": 0.0,
            }
        )
        for spec in ABLATIONS:
            reference = references[spec.model]
            for field_index, field in enumerate(FIELDS):
                metrics = field_metric(
                    particle.particle_enskog_mean[
                        frame_index, :, field_index
                    ],
                    reference[frame_index, :, field_index],
                )
                band_rms = rms(
                    particle.particle_enskog_half_width[
                        frame_index, :, field_index
                    ]
                )
                field_rows.append(
                    {
                        "cell_id": cell.cell_id,
                        "particle_count": cell.particle_count,
                        "target_knudsen": cell.target_knudsen,
                        "frame": frame,
                        "time": time,
                        "model": spec.model,
                        "model_label": spec.label,
                        "shear_viscosity_scale": spec.shear_scale,
                        "bulk_viscosity_scale": spec.bulk_scale,
                        "thermal_conductivity_scale": spec.thermal_scale,
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
            velocity = velocity_diagnostics(
                particle.x,
                particle.particle_mean[frame_index, :, 1],
                reference[frame_index, :, 1],
            )
            velocity_band_rms = rms(
                particle.particle_half_width[frame_index, :, 1]
            )
            velocity_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "particle_count": cell.particle_count,
                    "target_knudsen": cell.target_knudsen,
                    "frame": frame,
                    "time": time,
                    "model": spec.model,
                    "model_label": spec.label,
                    "shear_viscosity_scale": spec.shear_scale,
                    "bulk_viscosity_scale": spec.bulk_scale,
                    "thermal_conductivity_scale": spec.thermal_scale,
                    **velocity,
                    "ci95_half_width_rms": velocity_band_rms,
                    "relative_ci95_half_width_rms": (
                        velocity_band_rms
                        / max(
                            velocity["reference_rms"],
                            np.finfo(np.float64).tiny,
                        )
                    ),
                }
            )
            shape = positive_pulse_metrics(
                particle.x,
                reference[frame_index, :, 1],
            )
            shape_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "particle_count": cell.particle_count,
                    "target_knudsen": cell.target_knudsen,
                    "frame": frame,
                    "time": time,
                    "model": spec.model,
                    "model_label": spec.label,
                    "shear_viscosity_scale": spec.shear_scale,
                    "bulk_viscosity_scale": spec.bulk_scale,
                    "thermal_conductivity_scale": spec.thermal_scale,
                    **shape,
                    "peak_fractional_difference_vs_particle": (
                        safe_fractional_difference(
                            shape["peak_velocity"],
                            particle_shape["peak_velocity"],
                        )
                    ),
                    "rms_width_fractional_difference_vs_particle": (
                        safe_fractional_difference(
                            shape["rms_width"],
                            particle_shape["rms_width"],
                        )
                    ),
                }
            )

    annotate_baseline_change(
        field_rows,
        ("cell_id", "frame", "field"),
    )
    annotate_baseline_change(
        velocity_rows,
        ("cell_id", "frame"),
    )
    return EvolutionCell(
        particle=particle,
        references=references,
        field_rows=tuple(field_rows),
        velocity_rows=tuple(velocity_rows),
        shape_rows=tuple(shape_rows),
    )


def temporal_summary_rows(
    velocity_rows: list[dict[str, Any]],
    shape_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in velocity_rows:
        groups.setdefault(
            (str(row["cell_id"]), str(row["model"])),
            [],
        ).append(row)
    shapes = {
        (str(row["cell_id"]), int(row["frame"]), str(row["model"])): row
        for row in shape_rows
    }
    result = []
    for (cell_id, model), rows in sorted(groups.items()):
        rows.sort(key=lambda row: float(row["time"]))
        changes = np.array(
            [
                float(
                    row[
                        "fractional_relative_l2_change_vs_zero_transport"
                    ]
                )
                for row in rows
            ]
        )
        model_peaks = []
        particle_peaks = []
        model_widths = []
        particle_widths = []
        for row in rows:
            frame = int(row["frame"])
            model_shape = shapes[(cell_id, frame, model)]
            particle_shape = shapes[(cell_id, frame, "particle_mean")]
            model_peaks.append(float(model_shape["peak_velocity"]))
            particle_peaks.append(float(particle_shape["peak_velocity"]))
            model_widths.append(float(model_shape["rms_width"]))
            particle_widths.append(float(particle_shape["rms_width"]))
        model_peaks_array = np.array(model_peaks)
        particle_peaks_array = np.array(particle_peaks)
        model_widths_array = np.array(model_widths)
        particle_widths_array = np.array(particle_widths)
        first = rows[0]
        spec = SPEC_BY_MODEL[model]
        result.append(
            {
                "cell_id": cell_id,
                "particle_count": first["particle_count"],
                "target_knudsen": first["target_knudsen"],
                "model": model,
                "model_label": spec.label,
                "postinitial_frames": len(rows),
                "fraction_frames_better_than_zero_transport": float(
                    np.mean(changes < 0.0)
                ),
                "all_frames_better_than_zero_transport": bool(
                    np.all(changes < 0.0)
                ),
                "mean_velocity_error_fractional_change": float(
                    np.mean(changes)
                ),
                "minimum_velocity_error_fractional_change": float(
                    np.min(changes)
                ),
                "maximum_velocity_error_fractional_change": float(
                    np.max(changes)
                ),
                "final_velocity_error_fractional_change": float(changes[-1]),
                "peak_velocity_relative_rmse_vs_particle": (
                    rms(model_peaks_array - particle_peaks_array)
                    / max(
                        rms(particle_peaks_array),
                        1.0e-12,
                    )
                ),
                "rms_width_relative_rmse_vs_particle": (
                    rms(model_widths_array - particle_widths_array)
                    / max(
                        rms(particle_widths_array),
                        1.0e-12,
                    )
                ),
                "mean_peak_fractional_bias_vs_particle": float(
                    np.mean(
                        [
                            safe_fractional_difference(value, reference)
                            for value, reference in zip(
                                model_peaks_array,
                                particle_peaks_array,
                                strict=True,
                            )
                        ]
                    )
                ),
                "mean_width_fractional_bias_vs_particle": float(
                    np.mean(
                        [
                            safe_fractional_difference(value, reference)
                            for value, reference in zip(
                                model_widths_array,
                                particle_widths_array,
                                strict=True,
                            )
                        ]
                    )
                ),
            }
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_velocity_error_evolution(
    path: Path,
    scan: ScanConfig,
    rows: list[dict[str, Any]],
) -> None:
    largest = max(scan.particle_counts)
    figure, axes = plt.subplots(
        1,
        len(scan.knudsen_numbers),
        figsize=(13.8, 4.7),
        facecolor=BACKGROUND,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.82,
        bottom=0.15,
        wspace=0.24,
    )
    for axis, knudsen in zip(
        axes[0],
        sorted(scan.knudsen_numbers),
        strict=True,
    ):
        style_axis(axis)
        for spec in ABLATIONS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if int(row["particle_count"]) == largest
                    and math.isclose(
                        float(row["target_knudsen"]),
                        knudsen,
                        rel_tol=0.0,
                        abs_tol=1.0e-14,
                    )
                    and row["model"] == spec.model
                ),
                key=lambda row: float(row["time"]),
            )
            axis.plot(
                [float(row["time"]) for row in selected],
                [float(row["relative_l2"]) for row in selected],
                color=spec.color,
                linestyle=spec.linestyle,
                marker=spec.marker,
                markersize=3.5,
                linewidth=1.5,
                label=spec.label,
            )
        axis.set_title(f"Kn={knudsen:g}")
        axis.set_xlabel("time")
        axis.set_ylabel(r"velocity relative $L_2$")
        axis.legend(
            facecolor=PANEL,
            edgecolor=GRID,
            labelcolor=FOREGROUND,
            fontsize=6.7,
        )
    figure.suptitle(
        f"Does transport improvement persist in time? N={largest:,}",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def plot_field_error_evolution(
    path: Path,
    scan: ScanConfig,
    rows: list[dict[str, Any]],
) -> None:
    largest = max(scan.particle_counts)
    knudsen_numbers = sorted(scan.knudsen_numbers)
    figure, axes = plt.subplots(
        len(knudsen_numbers),
        len(FIELDS),
        figsize=(16.0, 10.2),
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
    for row_index, knudsen in enumerate(knudsen_numbers):
        for field_index, field in enumerate(FIELDS):
            axis = axes[row_index, field_index]
            style_axis(axis)
            for spec in ABLATIONS:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if int(row["particle_count"]) == largest
                        and math.isclose(
                            float(row["target_knudsen"]),
                            knudsen,
                            rel_tol=0.0,
                            abs_tol=1.0e-14,
                        )
                        and row["field"] == field
                        and row["model"] == spec.model
                    ),
                    key=lambda row: float(row["time"]),
                )
                axis.plot(
                    [float(row["time"]) for row in selected],
                    [float(row["relative_l2"]) for row in selected],
                    color=spec.color,
                    linestyle=spec.linestyle,
                    linewidth=1.35,
                    label=spec.label,
                )
            axis.set_yscale("log")
            if row_index == 0:
                axis.set_title(FIELD_LABELS[field])
            if field_index == 0:
                axis.set_ylabel(f"Kn={knudsen:g}\nrelative L2")
            if row_index == len(knudsen_numbers) - 1:
                axis.set_xlabel("time")
            if row_index == 0 and field_index == 0:
                axis.legend(
                    facecolor=PANEL,
                    edgecolor=GRID,
                    labelcolor=FOREGROUND,
                    fontsize=6.5,
                )
    figure.suptitle(
        f"Same-grid Enskog transport errors over time; N={largest:,}",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def plot_velocity_shape_evolution(
    path: Path,
    scan: ScanConfig,
    rows: list[dict[str, Any]],
) -> None:
    largest = max(scan.particle_counts)
    knudsen_numbers = sorted(scan.knudsen_numbers)
    figure, axes = plt.subplots(
        2,
        len(knudsen_numbers),
        figsize=(14.0, 7.5),
        facecolor=BACKGROUND,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.88,
        bottom=0.09,
        hspace=0.30,
        wspace=0.23,
    )
    curves = (
        ("particle_mean", "particle mean", PARTICLE_COLOR, "-", "o"),
        *(
            (
                spec.model,
                spec.label,
                spec.color,
                spec.linestyle,
                spec.marker,
            )
            for spec in ABLATIONS
        ),
    )
    for column, knudsen in enumerate(knudsen_numbers):
        for row_index, (metric, label) in enumerate(
            (
                ("peak_velocity", "peak velocity"),
                ("rms_width", "positive-pulse RMS width"),
            )
        ):
            axis = axes[row_index, column]
            style_axis(axis)
            for model, model_label, color, linestyle, marker in curves:
                selected = sorted(
                    (
                        row
                        for row in rows
                        if int(row["particle_count"]) == largest
                        and math.isclose(
                            float(row["target_knudsen"]),
                            knudsen,
                            rel_tol=0.0,
                            abs_tol=1.0e-14,
                        )
                        and row["model"] == model
                    ),
                    key=lambda row: float(row["time"]),
                )
                axis.plot(
                    [float(row["time"]) for row in selected],
                    [float(row[metric]) for row in selected],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=3.0,
                    linewidth=1.35,
                    label=model_label,
                )
            if row_index == 0:
                axis.set_title(f"Kn={knudsen:g}")
            axis.set_xlabel("time")
            axis.set_ylabel(label)
            if row_index == 0 and column == 0:
                axis.legend(
                    facecolor=PANEL,
                    edgecolor=GRID,
                    labelcolor=FOREGROUND,
                    fontsize=6.2,
                    ncol=2,
                )
    figure.suptitle(
        f"Velocity-pulse amplitude and width over time; N={largest:,}",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def build_summary(
    scan: ScanConfig,
    scan_output: Path,
    analyses: list[EvolutionCell],
    temporal_rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    nsf_cells: int,
) -> dict[str, Any]:
    largest = max(scan.particle_counts)
    return {
        "schema_version": 1,
        "experiment": "shock_tube_enskog_transport_time_evolution",
        "status": "completed",
        "scope": (
            "postinitial saved-frame comparison using existing particle "
            "trajectories and same-grid Enskog transport ablations"
        ),
        "method": {
            "nsf_cells": nsf_cells,
            "excluded_initial_time": 0.0,
            "pulse_definition": (
                "contiguous component containing the positive velocity peak "
                "above five percent of that peak"
            ),
            "analysis_tool_sha256": sha256_file(Path(__file__)),
            "ablation_tool_sha256": sha256_file(
                Path(__file__).with_name(
                    "analyze_enskog_transport_ablation.py"
                )
            ),
            "nsf_reference_tool_sha256": sha256_file(
                Path(__file__).with_name("enskog_nsf_reference.py")
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
        "postinitial_frames_per_cell": int(
            len(analyses[0].particle.frames) - 1
        ),
        "largest_particle_count": largest,
        "largest_particle_count_temporal_summary": [
            row
            for row in temporal_rows
            if int(row["particle_count"]) == largest
        ],
        "solver_diagnostics": diagnostics,
        "sources": [
            {
                "cell_id": item.particle.cell.cell_id,
                "particle_count": item.particle.cell.particle_count,
                "target_knudsen": item.particle.cell.target_knudsen,
                "macroscopic_sources": list(item.particle.source_hashes),
            }
            for item in analyses
        ],
    }


def run_self_test() -> int:
    x = np.linspace(-1.0, 1.0, 2001)
    center = 0.15
    sigma = 0.20
    velocity = np.exp(-0.5 * np.square((x - center) / sigma))
    pulse = positive_pulse_metrics(x, velocity)
    if abs(pulse["centroid_x"] - center) > 1.0e-3:
        raise AssertionError("pulse centroid self-test failed")
    if not 0.18 < pulse["rms_width"] < 0.21:
        raise AssertionError("pulse width self-test failed")
    rows = [
        {
            "cell_id": "synthetic",
            "frame": 1,
            "model": model,
            "relative_l2": value,
        }
        for model, value in (
            ("zero_transport", 0.20),
            ("shear_only", 0.10),
        )
    ]
    annotate_baseline_change(rows, ("cell_id", "frame"))
    shear = next(row for row in rows if row["model"] == "shear_only")
    if not math.isclose(
        float(shear["fractional_relative_l2_change_vs_zero_transport"]),
        -0.5,
    ):
        raise AssertionError("time-series baseline arithmetic failed")
    print("Enskog transport time-evolution self-test")
    print("  positive-pulse centroid and width: passed")
    print("  time-series baseline arithmetic: passed")
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
    if arguments.nsf_cells < 64:
        raise ValueError("--nsf-cells must be at least 64")
    scan = load_scan_config(arguments.config)
    scan_output = arguments.scan_output.resolve()
    output = (
        arguments.output.resolve()
        if arguments.output is not None
        else scan_output / "enskog-transport-evolution"
    )
    output.mkdir(parents=True, exist_ok=True)

    cache: dict[tuple[float, str], EnskogNsfSolution] = {}
    analyses = []
    for cell in derive_cells(scan):
        cell_output = scan_output / "cells" / cell.cell_id
        if not cell_is_complete(scan, cell, cell_output):
            raise ValueError(f"incomplete scan cell: {cell.cell_id}")
        print(
            f"tracking {cell.cell_id} "
            f"(N={cell.particle_count}, Kn={cell.target_knudsen:g})"
        )
        analyses.append(
            analyze_cell(
                cell,
                cell_output,
                arguments.nsf_cells,
                cache,
            )
        )

    field_rows = [row for item in analyses for row in item.field_rows]
    velocity_rows = [row for item in analyses for row in item.velocity_rows]
    shape_rows = [row for item in analyses for row in item.shape_rows]
    temporal_rows = temporal_summary_rows(velocity_rows, shape_rows)
    diagnostic_rows = solver_diagnostics(cache)

    field_path = output / "field-evolution.csv"
    velocity_path = output / "velocity-evolution.csv"
    shape_path = output / "velocity-shape-evolution.csv"
    temporal_path = output / "temporal-summary.csv"
    diagnostics_path = output / "solver-diagnostics.csv"
    summary_path = output / "summary.json"
    velocity_plot_path = output / "velocity-error-evolution.png"
    field_plot_path = output / "field-error-evolution.png"
    shape_plot_path = output / "velocity-shape-evolution.png"
    write_csv(field_path, field_rows)
    write_csv(velocity_path, velocity_rows)
    write_csv(shape_path, shape_rows)
    write_csv(temporal_path, temporal_rows)
    write_csv(diagnostics_path, diagnostic_rows)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            build_summary(
                scan,
                scan_output,
                analyses,
                temporal_rows,
                diagnostic_rows,
                arguments.nsf_cells,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    plot_velocity_error_evolution(
        velocity_plot_path,
        scan,
        velocity_rows,
    )
    plot_field_error_evolution(field_plot_path, scan, field_rows)
    plot_velocity_shape_evolution(shape_plot_path, scan, shape_rows)

    print("Enskog transport time evolution")
    print(f"  analyzed cells: {len(analyses)}")
    print(f"  postinitial frames per cell: {len(analyses[0].particle.frames) - 1}")
    print(f"  unique reference solves: {len(cache)}")
    print(f"  NSF cells: {arguments.nsf_cells}")
    print(f"  field evolution: {field_path}")
    print(f"  velocity evolution: {velocity_path}")
    print(f"  velocity shape: {shape_path}")
    print(f"  temporal summary: {temporal_path}")
    print(f"  solver diagnostics: {diagnostics_path}")
    print(f"  summary JSON: {summary_path}")
    print(f"  velocity-error plot: {velocity_plot_path}")
    print(f"  field-error plot: {field_plot_path}")
    print(f"  velocity-shape plot: {shape_plot_path}")
    print("  status: completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, csv.Error) as exception:
        raise SystemExit(
            f"Enskog transport time evolution: {exception}"
        ) from exception
