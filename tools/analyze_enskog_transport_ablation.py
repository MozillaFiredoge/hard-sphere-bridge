#!/usr/bin/env python3
"""Ablate Enskog NSF shear, bulk, and heat-transport contributions.

The analyzer reuses completed particle-count/Knudsen trajectories.  All five
continuum variants use the same grid and Enskog equilibrium closure, so their
differences isolate constitutive transport terms rather than reference-grid
changes.  No particle trajectory is rerun.
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
    ComparisonCell,
    analyze_cell as analyze_particle_cell,
    field_metric,
    interpolate_solution,
    rms,
    style_axis,
    velocity_diagnostics,
)
from analyze_multiseed_fields import (  # noqa: E402
    FIELDS,
    FIELD_LABELS,
    load_ensemble,
    portable_path,
    sha256_file,
)
from analyze_n_kn_scan import cell_is_complete  # noqa: E402
from enskog_nsf_reference import (  # noqa: E402
    EnskogNsfParameters,
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
class AblationSpec:
    model: str
    label: str
    shear_scale: float
    bulk_scale: float
    thermal_scale: float
    color: str
    linestyle: str
    marker: str


ABLATIONS = (
    AblationSpec(
        "zero_transport",
        "zero transport",
        0.0,
        0.0,
        0.0,
        "#22c55e",
        "--",
        "o",
    ),
    AblationSpec(
        "shear_only",
        "shear only",
        1.0,
        0.0,
        0.0,
        "#a78bfa",
        "-.",
        "s",
    ),
    AblationSpec(
        "momentum_transport",
        "shear + bulk",
        1.0,
        1.0,
        0.0,
        "#3b82f6",
        ":",
        "D",
    ),
    AblationSpec(
        "heat_only",
        "heat only",
        0.0,
        0.0,
        1.0,
        "#f59e0b",
        (0, (3, 1, 1, 1)),
        "v",
    ),
    AblationSpec(
        "full_transport",
        "full NSF",
        1.0,
        1.0,
        1.0,
        "#f43f5e",
        "-",
        "^",
    ),
)
SPEC_BY_MODEL = {spec.model: spec for spec in ABLATIONS}


@dataclass(frozen=True)
class AblationCell:
    base: ComparisonCell
    references: dict[str, np.ndarray]
    field_rows: tuple[dict[str, Any], ...]
    velocity_rows: tuple[dict[str, Any], ...]


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
            "SCAN_OUTPUT/enskog-transport-ablation"
        ),
    )
    parser.add_argument(
        "--nsf-cells",
        type=int,
        default=512,
        help="finite-volume cells shared by every ablation (default: 512)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate ablation definitions and error-change arithmetic",
    )
    return parser.parse_args()


def frame_times(ensemble: Any) -> list[tuple[int, float]]:
    return [
        (int(frame), float(time))
        for frame, time in zip(ensemble.frames, ensemble.times, strict=True)
    ]


def parameters_for(spec: AblationSpec, cells: int) -> EnskogNsfParameters:
    return EnskogNsfParameters(
        cells=cells,
        shear_viscosity_scale=spec.shear_scale,
        bulk_viscosity_scale=spec.bulk_scale,
        thermal_conductivity_scale=spec.thermal_scale,
    )


def analyze_cell(
    scan: ScanConfig,
    cell: ScanCell,
    cell_output: Path,
    nsf_cells: int,
    cache: dict[tuple[float, str], EnskogNsfSolution],
) -> AblationCell:
    base = analyze_particle_cell(scan, cell, cell_output)
    batch = load_batch_config(cell_output / "batch-input.json")
    ensemble = load_ensemble(batch, cell_output)
    effective_config = cell_output / "effective-config.json"
    references: dict[str, np.ndarray] = {}
    field_rows: list[dict[str, Any]] = []
    velocity_rows: list[dict[str, Any]] = []
    final = len(ensemble.frames) - 1

    for spec in ABLATIONS:
        key = (cell.target_knudsen, spec.model)
        solution = cache.get(key)
        if solution is None:
            solution = solve_enskog_nsf_reference(
                effective_config,
                frame_times(ensemble),
                particle_count=batch.particle_count,
                particle_diameter=cell.particle_diameter,
                parameters=parameters_for(spec, nsf_cells),
            )
            cache[key] = solution
        reference = interpolate_solution(base.x, solution.frames)
        references[spec.model] = reference

        for field_index, field in enumerate(FIELDS):
            metrics = field_metric(
                base.particle_enskog_mean[final, :, field_index],
                reference[final, :, field_index],
            )
            band_rms = rms(
                base.particle_enskog_half_width[final, :, field_index]
            )
            field_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "particle_count": cell.particle_count,
                    "target_knudsen": cell.target_knudsen,
                    "maximum_initial_packing_fraction": (
                        cell.maximum_initial_packing_fraction
                    ),
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
            base.x,
            base.particle_mean[final, :, 1],
            reference[final, :, 1],
        )
        velocity_band_rms = rms(base.particle_half_width[final, :, 1])
        velocity_rows.append(
            {
                "cell_id": cell.cell_id,
                "particle_count": cell.particle_count,
                "target_knudsen": cell.target_knudsen,
                "maximum_initial_packing_fraction": (
                    cell.maximum_initial_packing_fraction
                ),
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

    annotate_baseline_change(field_rows, ("cell_id", "field"))
    annotate_baseline_change(velocity_rows, ("cell_id",))
    return AblationCell(
        base=base,
        references=references,
        field_rows=tuple(field_rows),
        velocity_rows=tuple(velocity_rows),
    )


def annotate_baseline_change(
    rows: list[dict[str, Any]],
    group_keys: tuple[str, ...],
) -> None:
    baselines = {
        tuple(row[key] for key in group_keys): float(row["relative_l2"])
        for row in rows
        if row["model"] == "zero_transport"
    }
    tiny = np.finfo(np.float64).tiny
    for row in rows:
        baseline = baselines[tuple(row[key] for key in group_keys)]
        value = float(row["relative_l2"])
        row["absolute_relative_l2_change_vs_zero_transport"] = value - baseline
        row["fractional_relative_l2_change_vs_zero_transport"] = (
            value / max(baseline, tiny) - 1.0
        )


def solver_diagnostics(
    cache: dict[tuple[float, str], EnskogNsfSolution],
) -> list[dict[str, Any]]:
    rows = []
    for (knudsen, model), solution in sorted(cache.items()):
        spec = SPEC_BY_MODEL[model]
        rows.append(
            {
                "target_knudsen": knudsen,
                "model": model,
                "model_label": spec.label,
                "shear_viscosity_scale": spec.shear_scale,
                "bulk_viscosity_scale": spec.bulk_scale,
                "thermal_conductivity_scale": spec.thermal_scale,
                "cells": solution.parameters.cells,
                "steps": solution.steps,
                "hyperbolic_limited_steps": (
                    solution.hyperbolic_limited_steps
                ),
                "diffusion_limited_steps": solution.diffusion_limited_steps,
                "positivity_retries": solution.positivity_retries,
                "minimum_time_step": solution.minimum_time_step,
                "maximum_momentum_diffusivity": (
                    solution.maximum_momentum_diffusivity
                ),
                "maximum_thermal_diffusivity": (
                    solution.maximum_thermal_diffusivity
                ),
                "relative_mass_drift": solution.relative_mass_drift,
                "relative_energy_drift": solution.relative_energy_drift,
                "minimum_density": solution.minimum_density,
                "minimum_temperature": solution.minimum_temperature,
            }
        )
    return rows


def velocity_attribution_rows(
    velocity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, float]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in velocity_rows:
        cell_id = str(row["cell_id"])
        grouped.setdefault(cell_id, {})[str(row["model"])] = float(
            row["relative_l2"]
        )
        metadata[cell_id] = {
            "cell_id": cell_id,
            "particle_count": row["particle_count"],
            "target_knudsen": row["target_knudsen"],
        }
    result = []
    for cell_id in sorted(grouped):
        errors = grouped[cell_id]
        missing = [spec.model for spec in ABLATIONS if spec.model not in errors]
        if missing:
            raise ValueError(
                f"{cell_id}: missing ablation models: {', '.join(missing)}"
            )
        zero = errors["zero_transport"]
        shear = errors["shear_only"]
        momentum = errors["momentum_transport"]
        heat = errors["heat_only"]
        full = errors["full_transport"]
        result.append(
            {
                **metadata[cell_id],
                "zero_transport_relative_l2": zero,
                "shear_only_relative_l2": shear,
                "momentum_transport_relative_l2": momentum,
                "heat_only_relative_l2": heat,
                "full_transport_relative_l2": full,
                "shear_step_fractional_change": shear / zero - 1.0,
                "bulk_after_shear_fractional_change": (
                    momentum / shear - 1.0
                ),
                "heat_only_fractional_change_vs_zero": heat / zero - 1.0,
                "heat_after_momentum_fractional_change": (
                    full / momentum - 1.0
                ),
                "full_fractional_change_vs_zero": full / zero - 1.0,
            }
        )
    return result


def bulk_increment_grid_check(
    scan: ScanConfig,
    scan_output: Path,
    analyses: list[AblationCell],
    nsf_cells: int,
) -> list[dict[str, Any]]:
    item = next(
        candidate
        for candidate in analyses
        if candidate.base.cell.particle_count == max(scan.particle_counts)
        and math.isclose(
            candidate.base.cell.target_knudsen,
            min(scan.knudsen_numbers),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        )
    )
    cell = item.base.cell
    cell_output = scan_output / "cells" / cell.cell_id
    batch = load_batch_config(cell_output / "batch-input.json")
    ensemble = load_ensemble(batch, cell_output)
    refined: dict[str, tuple[EnskogNsfSolution, np.ndarray]] = {}
    for model in ("shear_only", "momentum_transport"):
        spec = SPEC_BY_MODEL[model]
        solution = solve_enskog_nsf_reference(
            cell_output / "effective-config.json",
            frame_times(ensemble),
            particle_count=batch.particle_count,
            particle_diameter=cell.particle_diameter,
            parameters=parameters_for(spec, 2 * nsf_cells),
        )
        refined[model] = (
            solution,
            interpolate_solution(item.base.x, solution.frames),
        )

    coarse_shear = item.references["shear_only"][-1]
    coarse_momentum = item.references["momentum_transport"][-1]
    refined_shear = refined["shear_only"][1][-1]
    refined_momentum = refined["momentum_transport"][1][-1]
    rows = []
    tiny = np.finfo(np.float64).tiny
    for field_index, field in enumerate(FIELDS):
        coarse_increment = (
            rms(
                coarse_momentum[:, field_index]
                - coarse_shear[:, field_index]
            )
            / max(rms(coarse_shear[:, field_index]), tiny)
        )
        refined_increment = (
            rms(
                refined_momentum[:, field_index]
                - refined_shear[:, field_index]
            )
            / max(rms(refined_shear[:, field_index]), tiny)
        )
        rows.append(
            {
                "cell_id": cell.cell_id,
                "particle_count": cell.particle_count,
                "target_knudsen": cell.target_knudsen,
                "field": field,
                "coarse_cells": nsf_cells,
                "refined_cells": 2 * nsf_cells,
                "shear_coarse_refined_relative_l2": (
                    rms(
                        coarse_shear[:, field_index]
                        - refined_shear[:, field_index]
                    )
                    / max(rms(refined_shear[:, field_index]), tiny)
                ),
                "momentum_coarse_refined_relative_l2": (
                    rms(
                        coarse_momentum[:, field_index]
                        - refined_momentum[:, field_index]
                    )
                    / max(rms(refined_momentum[:, field_index]), tiny)
                ),
                "bulk_increment_coarse_relative_l2": coarse_increment,
                "bulk_increment_refined_relative_l2": refined_increment,
                "bulk_increment_refined_over_coarse": (
                    refined_increment / max(coarse_increment, tiny)
                ),
                "refined_shear_steps": refined["shear_only"][0].steps,
                "refined_momentum_steps": (
                    refined["momentum_transport"][0].steps
                ),
                "maximum_refined_absolute_conservation_drift": max(
                    abs(refined["shear_only"][0].relative_mass_drift),
                    abs(refined["shear_only"][0].relative_energy_drift),
                    abs(
                        refined["momentum_transport"][0].relative_mass_drift
                    ),
                    abs(
                        refined["momentum_transport"][0].relative_energy_drift
                    ),
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_error_ablation(
    path: Path,
    scan: ScanConfig,
    rows: list[dict[str, Any]],
) -> None:
    largest = max(scan.particle_counts)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 7.8),
        facecolor=BACKGROUND,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.87,
        bottom=0.09,
        hspace=0.30,
        wspace=0.23,
    )
    for axis, field in zip(axes.flat, FIELDS, strict=True):
        style_axis(axis)
        for spec in ABLATIONS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if int(row["particle_count"]) == largest
                    and row["field"] == field
                    and row["model"] == spec.model
                ),
                key=lambda row: float(row["target_knudsen"]),
            )
            axis.plot(
                [float(row["target_knudsen"]) for row in selected],
                [float(row["relative_l2"]) for row in selected],
                color=spec.color,
                linestyle=spec.linestyle,
                marker=spec.marker,
                linewidth=1.7,
                label=spec.label,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("target Knudsen number")
        axis.set_ylabel("relative L2")
        axis.set_title(FIELD_LABELS[field])
        axis.legend(
            facecolor=PANEL,
            edgecolor=GRID,
            labelcolor=FOREGROUND,
            fontsize=7,
        )
    figure.suptitle(
        f"Which Enskog transport term reduces the error? N={largest:,}",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def plot_velocity_profiles(
    path: Path,
    scan: ScanConfig,
    analyses: list[AblationCell],
) -> None:
    largest = max(scan.particle_counts)
    selected = sorted(
        (
            item
            for item in analyses
            if item.base.cell.particle_count == largest
        ),
        key=lambda item: item.base.cell.target_knudsen,
    )
    figure, axes = plt.subplots(
        1,
        len(selected),
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
    for axis, item in zip(axes[0], selected, strict=True):
        style_axis(axis)
        base = item.base
        axis.fill_between(
            base.x,
            base.particle_mean[-1, :, 1] - base.particle_half_width[-1, :, 1],
            base.particle_mean[-1, :, 1] + base.particle_half_width[-1, :, 1],
            color=PARTICLE_COLOR,
            alpha=0.18,
            linewidth=0.0,
        )
        axis.plot(
            base.x,
            base.particle_mean[-1, :, 1],
            color=PARTICLE_COLOR,
            linewidth=1.7,
            label="particle mean",
        )
        for spec in ABLATIONS:
            axis.plot(
                base.x,
                item.references[spec.model][-1, :, 1],
                color=spec.color,
                linestyle=spec.linestyle,
                linewidth=1.25,
                label=spec.label,
            )
        axis.set_title(f"Kn={base.cell.target_knudsen:g}")
        axis.set_xlabel("x")
        axis.set_ylabel(r"$u_x$")
        axis.legend(
            facecolor=PANEL,
            edgecolor=GRID,
            labelcolor=FOREGROUND,
            fontsize=6.6,
        )
    figure.suptitle(
        f"Enskog transport ablation at t=0.20; N={largest:,}",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def plot_velocity_attribution(
    path: Path,
    scan: ScanConfig,
    rows: list[dict[str, Any]],
) -> None:
    largest = max(scan.particle_counts)
    selected = sorted(
        (row for row in rows if int(row["particle_count"]) == largest),
        key=lambda row: float(row["target_knudsen"]),
    )
    models = (
        ("shear_only", "shear only", "#a78bfa"),
        ("momentum_transport", "shear + bulk", "#3b82f6"),
        ("heat_only", "heat only", "#f59e0b"),
        ("full_transport", "full NSF", "#f43f5e"),
    )
    x = np.arange(len(selected), dtype=np.float64)
    width = 0.19
    figure, axis = plt.subplots(figsize=(10.5, 5.2), facecolor=BACKGROUND)
    figure.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.15)
    style_axis(axis)
    for model_index, (model, label, color) in enumerate(models):
        values = []
        for row in selected:
            baseline = float(row["zero_transport_relative_l2"])
            values.append(
                100.0
                * (
                    float(row[f"{model}_relative_l2"]) / baseline
                    - 1.0
                )
            )
        offset = (model_index - (len(models) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            values,
            width=width,
            color=color,
            label=label,
        )
    axis.axhline(0.0, color=FOREGROUND, linewidth=1.0, alpha=0.75)
    axis.set_xticks(
        x,
        [f"Kn={float(row['target_knudsen']):g}" for row in selected],
    )
    axis.set_ylabel("velocity relative-L2 change vs zero transport (%)")
    axis.legend(
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=FOREGROUND,
        fontsize=8,
        ncol=2,
    )
    figure.suptitle(
        f"Path-independent model errors; N={largest:,}, t=0.20",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def build_summary(
    scan: ScanConfig,
    scan_output: Path,
    analyses: list[AblationCell],
    field_rows: list[dict[str, Any]],
    velocity_rows: list[dict[str, Any]],
    attribution_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    nsf_cells: int,
) -> dict[str, Any]:
    largest = max(scan.particle_counts)
    return {
        "schema_version": 1,
        "experiment": "shock_tube_enskog_transport_ablation",
        "status": "completed",
        "scope": (
            "constitutive-flux ablation at fixed Carnahan--Starling "
            "thermodynamics and a shared finite-volume grid"
        ),
        "method": {
            "nsf_cells": nsf_cells,
            "models": [
                {
                    "model": spec.model,
                    "label": spec.label,
                    "shear_viscosity_scale": spec.shear_scale,
                    "bulk_viscosity_scale": spec.bulk_scale,
                    "thermal_conductivity_scale": spec.thermal_scale,
                }
                for spec in ABLATIONS
            ],
            "sequential_attribution_warning": (
                "bulk-after-shear and heat-after-momentum changes are "
                "path-dependent because the PDE response is nonlinear"
            ),
            "analysis_tool_sha256": sha256_file(Path(__file__)),
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
        "unique_reference_solves": len(diagnostic_rows),
        "largest_particle_count": largest,
        "largest_particle_count_field_rows": [
            row
            for row in field_rows
            if int(row["particle_count"]) == largest
        ],
        "largest_particle_count_velocity_rows": [
            row
            for row in velocity_rows
            if int(row["particle_count"]) == largest
        ],
        "largest_particle_count_velocity_attribution": [
            row
            for row in attribution_rows
            if int(row["particle_count"]) == largest
        ],
        "solver_diagnostics": diagnostic_rows,
        "bulk_increment_grid_check": grid_rows,
        "sources": [
            {
                "cell_id": item.base.cell.cell_id,
                "particle_count": item.base.cell.particle_count,
                "target_knudsen": item.base.cell.target_knudsen,
                "macroscopic_sources": list(item.base.source_hashes),
            }
            for item in analyses
        ],
    }


def run_self_test() -> int:
    if len(SPEC_BY_MODEL) != len(ABLATIONS):
        raise AssertionError("ablation model names are not unique")
    if (
        SPEC_BY_MODEL["zero_transport"].shear_scale != 0.0
        or SPEC_BY_MODEL["zero_transport"].bulk_scale != 0.0
        or SPEC_BY_MODEL["zero_transport"].thermal_scale != 0.0
    ):
        raise AssertionError("zero-transport model is not zero")
    if (
        SPEC_BY_MODEL["full_transport"].shear_scale != 1.0
        or SPEC_BY_MODEL["full_transport"].bulk_scale != 1.0
        or SPEC_BY_MODEL["full_transport"].thermal_scale != 1.0
    ):
        raise AssertionError("full-transport model is not complete")
    rows = [
        {
            "cell_id": "synthetic",
            "model": model,
            "relative_l2": value,
        }
        for model, value in (
            ("zero_transport", 0.20),
            ("shear_only", 0.12),
            ("momentum_transport", 0.10),
            ("heat_only", 0.18),
            ("full_transport", 0.08),
        )
    ]
    annotate_baseline_change(rows, ("cell_id",))
    full = next(row for row in rows if row["model"] == "full_transport")
    if not math.isclose(
        float(full["fractional_relative_l2_change_vs_zero_transport"]),
        -0.60,
    ):
        raise AssertionError("baseline change arithmetic failed")
    attributed = velocity_attribution_rows(
        [
            {
                **row,
                "particle_count": 8,
                "target_knudsen": 0.1,
            }
            for row in rows
        ]
    )
    if not math.isclose(
        float(attributed[0]["bulk_after_shear_fractional_change"]),
        -1.0 / 6.0,
    ):
        raise AssertionError("sequential attribution arithmetic failed")
    print("Enskog transport ablation self-test")
    print("  model definitions: passed")
    print("  baseline and sequential attribution: passed")
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
        else scan_output / "enskog-transport-ablation"
    )
    output.mkdir(parents=True, exist_ok=True)

    cache: dict[tuple[float, str], EnskogNsfSolution] = {}
    analyses = []
    for cell in derive_cells(scan):
        cell_output = scan_output / "cells" / cell.cell_id
        if not cell_is_complete(scan, cell, cell_output):
            raise ValueError(f"incomplete scan cell: {cell.cell_id}")
        print(
            f"ablating {cell.cell_id} "
            f"(N={cell.particle_count}, Kn={cell.target_knudsen:g})"
        )
        analyses.append(
            analyze_cell(
                scan,
                cell,
                cell_output,
                arguments.nsf_cells,
                cache,
            )
        )

    field_rows = [row for item in analyses for row in item.field_rows]
    velocity_rows = [row for item in analyses for row in item.velocity_rows]
    attribution_rows = velocity_attribution_rows(velocity_rows)
    diagnostic_rows = solver_diagnostics(cache)
    grid_rows = bulk_increment_grid_check(
        scan,
        scan_output,
        analyses,
        arguments.nsf_cells,
    )

    field_path = output / "field-ablation.csv"
    velocity_path = output / "velocity-ablation.csv"
    attribution_path = output / "velocity-attribution.csv"
    diagnostics_path = output / "solver-diagnostics.csv"
    grid_path = output / "bulk-increment-grid-check.csv"
    summary_path = output / "summary.json"
    error_plot_path = output / "transport-ablation-errors.png"
    velocity_plot_path = output / "transport-ablation-velocity.png"
    attribution_plot_path = output / "velocity-error-attribution.png"
    write_csv(field_path, field_rows)
    write_csv(velocity_path, velocity_rows)
    write_csv(attribution_path, attribution_rows)
    write_csv(diagnostics_path, diagnostic_rows)
    write_csv(grid_path, grid_rows)
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(
            build_summary(
                scan,
                scan_output,
                analyses,
                field_rows,
                velocity_rows,
                attribution_rows,
                diagnostic_rows,
                grid_rows,
                arguments.nsf_cells,
            ),
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    plot_error_ablation(error_plot_path, scan, field_rows)
    plot_velocity_profiles(velocity_plot_path, scan, analyses)
    plot_velocity_attribution(attribution_plot_path, scan, attribution_rows)

    print("Enskog transport ablation")
    print(f"  analyzed cells: {len(analyses)}")
    print(f"  ablation models: {len(ABLATIONS)}")
    print(f"  unique reference solves: {len(cache)}")
    print(f"  NSF cells: {arguments.nsf_cells}")
    print(f"  field table: {field_path}")
    print(f"  velocity table: {velocity_path}")
    print(f"  attribution table: {attribution_path}")
    print(f"  solver diagnostics: {diagnostics_path}")
    print(f"  bulk-increment grid check: {grid_path}")
    print(f"  summary JSON: {summary_path}")
    print(f"  error plot: {error_plot_path}")
    print(f"  velocity plot: {velocity_plot_path}")
    print(f"  attribution plot: {attribution_plot_path}")
    print("  status: completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, csv.Error) as exception:
        raise SystemExit(f"Enskog transport ablation: {exception}") from exception
