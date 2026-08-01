#!/usr/bin/env python3
"""Compare particle fields with ideal Euler, Enskog-Euler, and Enskog NSF."""

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
    COUNT_COLORS,
    ENSKOG_COLOR,
    FOREGROUND,
    GRID,
    IDEAL_COLOR,
    KINETIC_COLOR,
    PANEL,
    PARTICLE_COLOR,
    ComparisonCell,
    analyze_cell as analyze_inviscid_cell,
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
    enskog_transport_coefficients,
    solve_enskog_nsf_reference,
)
from run_multiseed_shock_tube import load_batch_config  # noqa: E402
from run_n_kn_scan import (  # noqa: E402
    ScanCell,
    ScanConfig,
    derive_cells,
    load_scan_config,
)


NSF_COLOR = "#f43f5e"


@dataclass(frozen=True)
class NsfComparisonCell:
    base: ComparisonCell
    nsf_reference: np.ndarray
    nsf_solution: EnskogNsfSolution
    field_rows: tuple[dict[str, Any], ...]
    velocity_row: dict[str, Any]
    transport_row: dict[str, Any]


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
        help="output directory; defaults to SCAN_OUTPUT/enskog-nsf-comparison",
    )
    parser.add_argument(
        "--nsf-cells",
        type=int,
        default=512,
        help="Enskog NSF finite-volume cells (default: 512)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="validate three-model table and improvement arithmetic",
    )
    return parser.parse_args()


def frame_times(ensemble: Any) -> list[tuple[int, float]]:
    return [
        (int(frame), float(time))
        for frame, time in zip(ensemble.frames, ensemble.times, strict=True)
    ]


def analyze_cell(
    scan: ScanConfig,
    cell: ScanCell,
    cell_output: Path,
    nsf_cells: int,
    cached_solution: EnskogNsfSolution | None,
) -> NsfComparisonCell:
    base = analyze_inviscid_cell(scan, cell, cell_output)
    batch = load_batch_config(cell_output / "batch-input.json")
    effective_config = cell_output / "effective-config.json"
    with effective_config.open(encoding="utf-8") as stream:
        effective = json.load(stream)
    particle_mass = float(effective["units"]["particle_mass"])
    boltzmann_constant = float(effective["units"]["boltzmann_constant"])
    ensemble = load_ensemble(batch, cell_output)
    solution = cached_solution
    if solution is None:
        solution = solve_enskog_nsf_reference(
            effective_config,
            frame_times(ensemble),
            particle_count=batch.particle_count,
            particle_diameter=cell.particle_diameter,
            parameters=EnskogNsfParameters(cells=nsf_cells),
        )
    nsf_reference = interpolate_solution(ensemble.x, solution.frames)
    final = len(ensemble.frames) - 1
    rows = []
    for field_index, field in enumerate(FIELDS):
        metrics = field_metric(
            base.particle_enskog_mean[final, :, field_index],
            nsf_reference[final, :, field_index],
        )
        band_rms = rms(base.particle_enskog_half_width[final, :, field_index])
        rows.append(
            {
                "cell_id": cell.cell_id,
                "particle_count": cell.particle_count,
                "target_knudsen": cell.target_knudsen,
                "maximum_initial_packing_fraction": (
                    cell.maximum_initial_packing_fraction
                ),
                "reference_model": "enskog_navier_stokes_fourier",
                "particle_pressure_definition": (
                    "carnahan_starling_n_kT_Z"
                ),
                "field": field,
                **metrics,
                "ci95_half_width_rms": band_rms,
                "relative_ci95_half_width_rms": (
                    band_rms
                    / max(metrics["reference_rms"], np.finfo(np.float64).tiny)
                ),
            }
        )
    velocity = velocity_diagnostics(
        ensemble.x,
        base.particle_mean[final, :, 1],
        nsf_reference[final, :, 1],
    )
    velocity_band_rms = rms(base.particle_half_width[final, :, 1])
    velocity_row = {
        "cell_id": cell.cell_id,
        "particle_count": cell.particle_count,
        "target_knudsen": cell.target_knudsen,
        "maximum_initial_packing_fraction": (
            cell.maximum_initial_packing_fraction
        ),
        "reference_model": "enskog_navier_stokes_fourier",
        **velocity,
        "ci95_half_width_rms": velocity_band_rms,
        "relative_ci95_half_width_rms": (
            velocity_band_rms
            / max(velocity["reference_rms"], np.finfo(np.float64).tiny)
        ),
    }

    initial = solution.frames[0].fields
    final_fields = solution.frames[-1].fields
    initial_transport = enskog_transport_coefficients(
        initial["number_density"],
        initial["temperature"],
        cell.particle_diameter,
        particle_mass,
        boltzmann_constant,
    )
    final_transport = enskog_transport_coefficients(
        final_fields["number_density"],
        final_fields["temperature"],
        cell.particle_diameter,
        particle_mass,
        boltzmann_constant,
    )
    transport_row = {
        "cell_id": cell.cell_id,
        "particle_count": cell.particle_count,
        "target_knudsen": cell.target_knudsen,
        "particle_diameter": cell.particle_diameter,
        "nsf_cells": nsf_cells,
        "transport_scale": solution.parameters.transport_scale,
        "shear_viscosity_scale": (
            solution.parameters.shear_viscosity_scale
        ),
        "bulk_viscosity_scale": solution.parameters.bulk_viscosity_scale,
        "thermal_conductivity_scale": (
            solution.parameters.thermal_conductivity_scale
        ),
        "steps": solution.steps,
        "hyperbolic_limited_steps": solution.hyperbolic_limited_steps,
        "diffusion_limited_steps": solution.diffusion_limited_steps,
        "positivity_retries": solution.positivity_retries,
        "minimum_time_step": solution.minimum_time_step,
        "maximum_absolute_relative_conservation_drift": max(
            abs(solution.relative_mass_drift),
            abs(solution.relative_energy_drift),
        ),
        "initial_shear_viscosity_minimum": float(
            np.min(initial_transport.shear_viscosity)
        ),
        "initial_shear_viscosity_maximum": float(
            np.max(initial_transport.shear_viscosity)
        ),
        "initial_bulk_viscosity_maximum": float(
            np.max(initial_transport.bulk_viscosity)
        ),
        "initial_thermal_conductivity_minimum": float(
            np.min(initial_transport.thermal_conductivity)
        ),
        "initial_thermal_conductivity_maximum": float(
            np.max(initial_transport.thermal_conductivity)
        ),
        "final_momentum_diffusivity_maximum": float(
            np.max(final_transport.momentum_diffusivity)
        ),
        "final_thermal_diffusivity_maximum": float(
            np.max(final_transport.thermal_diffusivity)
        ),
        "maximum_reference_packing_fraction": (
            solution.maximum_packing_fraction
        ),
    }
    return NsfComparisonCell(
        base=base,
        nsf_reference=nsf_reference,
        nsf_solution=solution,
        field_rows=tuple(rows),
        velocity_row=velocity_row,
        transport_row=transport_row,
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
    analyses: list[NsfComparisonCell],
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
    for row_index, item in enumerate(selected):
        base = item.base
        for field_index, field in enumerate(FIELDS):
            axis = axes[row_index, field_index]
            style_axis(axis)
            if field == "pressure":
                particle = base.particle_enskog_mean[-1, :, field_index]
                half = base.particle_enskog_half_width[-1, :, field_index]
                axis.plot(
                    base.x,
                    base.particle_mean[-1, :, field_index],
                    color=KINETIC_COLOR,
                    linewidth=0.9,
                    alpha=0.8,
                    label="particle kinetic n kT",
                )
                particle_label = "particle + CS closure"
            else:
                particle = base.particle_mean[-1, :, field_index]
                half = base.particle_half_width[-1, :, field_index]
                particle_label = "particle mean"
            axis.fill_between(
                base.x,
                particle - half,
                particle + half,
                color=PARTICLE_COLOR,
                alpha=0.16,
                linewidth=0.0,
            )
            axis.plot(
                base.x,
                particle,
                color=PARTICLE_COLOR,
                linewidth=1.45,
                label=particle_label,
            )
            axis.plot(
                base.x,
                base.ideal_reference[-1, :, field_index],
                color=IDEAL_COLOR,
                linestyle="--",
                linewidth=1.25,
                label="ideal Euler",
            )
            axis.plot(
                base.x,
                base.enskog_reference[-1, :, field_index],
                color=ENSKOG_COLOR,
                linestyle="-.",
                linewidth=1.25,
                label="Enskog-Euler",
            )
            axis.plot(
                base.x,
                item.nsf_reference[-1, :, field_index],
                color=NSF_COLOR,
                linewidth=1.5,
                label="Enskog NSF",
            )
            if row_index == 0:
                axis.set_title(FIELD_LABELS[field], fontsize=12)
            if field_index == 0:
                axis.set_ylabel(
                    f"Kn={base.cell.target_knudsen:g}\nfield value"
                )
            if row_index == len(selected) - 1:
                axis.set_xlabel("x")
            if row_index == 0 and field_index == 0:
                axis.legend(
                    facecolor=PANEL,
                    edgecolor=GRID,
                    labelcolor=FOREGROUND,
                    fontsize=6.8,
                )
    figure.suptitle(
        f"Particle, Euler, and Enskog NSF at t=0.20; N={largest:,}",
        color=FOREGROUND,
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def plot_error_comparison(
    path: Path,
    scan: ScanConfig,
    field_rows: list[dict[str, Any]],
) -> None:
    largest = max(scan.particle_counts)
    models = (
        ("ideal_euler", "ideal Euler", IDEAL_COLOR, "--", "o"),
        (
            "enskog_euler_carnahan_starling",
            "Enskog-Euler",
            ENSKOG_COLOR,
            "-.",
            "s",
        ),
        (
            "enskog_navier_stokes_fourier",
            "Enskog NSF",
            NSF_COLOR,
            "-",
            "^",
        ),
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 7.6),
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
        for model, label, color, linestyle, marker in models:
            selected = sorted(
                (
                    row
                    for row in field_rows
                    if int(row["particle_count"]) == largest
                    and row["field"] == field
                    and row["reference_model"] == model
                ),
                key=lambda row: float(row["target_knudsen"]),
            )
            axis.plot(
                [float(row["target_knudsen"]) for row in selected],
                [float(row["relative_l2"]) for row in selected],
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.7,
                label=label,
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
            fontsize=8,
        )
    figure.suptitle(
        f"Does Enskog transport reduce particle/reference error? N={largest:,}",
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
    analyses: list[NsfComparisonCell],
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
        figsize=(13.5, 4.6),
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
            linewidth=1.5,
            label="particle mean",
        )
        axis.plot(
            base.x,
            base.ideal_reference[-1, :, 1],
            color=IDEAL_COLOR,
            linestyle="--",
            linewidth=1.3,
            label="ideal Euler",
        )
        axis.plot(
            base.x,
            base.enskog_reference[-1, :, 1],
            color=ENSKOG_COLOR,
            linestyle="-.",
            linewidth=1.3,
            label="Enskog-Euler",
        )
        axis.plot(
            base.x,
            item.nsf_reference[-1, :, 1],
            color=NSF_COLOR,
            linewidth=1.6,
            label="Enskog NSF",
        )
        axis.set_title(f"Kn={base.cell.target_knudsen:g}")
        axis.set_xlabel("x")
        axis.set_ylabel(r"$u_x$")
        axis.legend(
            facecolor=PANEL,
            edgecolor=GRID,
            labelcolor=FOREGROUND,
            fontsize=7,
        )
    figure.suptitle(
        f"Velocity pulse with Enskog transport; N={largest:,}, t=0.20",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def reference_grid_check(
    scan: ScanConfig,
    scan_output: Path,
    analyses: list[NsfComparisonCell],
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
    refined = solve_enskog_nsf_reference(
        cell_output / "effective-config.json",
        frame_times(ensemble),
        particle_count=batch.particle_count,
        particle_diameter=cell.particle_diameter,
        parameters=EnskogNsfParameters(cells=2 * nsf_cells),
    )
    refined_values = interpolate_solution(item.base.x, refined.frames)
    rows = []
    for field_index, field in enumerate(FIELDS):
        rows.append(
            {
                "cell_id": cell.cell_id,
                "particle_count": cell.particle_count,
                "target_knudsen": cell.target_knudsen,
                "field": field,
                "coarse_cells": nsf_cells,
                "refined_cells": 2 * nsf_cells,
                "relative_l2_coarse_vs_refined": (
                    rms(
                        item.nsf_reference[-1, :, field_index]
                        - refined_values[-1, :, field_index]
                    )
                    / max(
                        rms(refined_values[-1, :, field_index]),
                        np.finfo(np.float64).tiny,
                    )
                ),
                "refined_steps": refined.steps,
                "refined_relative_mass_drift": refined.relative_mass_drift,
                "refined_relative_energy_drift": refined.relative_energy_drift,
            }
        )
    return rows


def largest_particle_table(
    scan: ScanConfig,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    largest = max(scan.particle_counts)
    table = []
    for knudsen in scan.knudsen_numbers:
        record: dict[str, Any] = {"target_knudsen": knudsen}
        for row in rows:
            if (
                int(row["particle_count"]) != largest
                or not math.isclose(
                    float(row["target_knudsen"]),
                    knudsen,
                    rel_tol=0.0,
                    abs_tol=1.0e-14,
                )
            ):
                continue
            model = {
                "ideal_euler": "ideal",
                "enskog_euler_carnahan_starling": "enskog_euler",
                "enskog_navier_stokes_fourier": "enskog_nsf",
            }[str(row["reference_model"])]
            record[f"{row['field']}_{model}_relative_l2"] = row["relative_l2"]
        table.append(record)
    return table


def build_summary(
    scan: ScanConfig,
    scan_output: Path,
    analyses: list[NsfComparisonCell],
    field_rows: list[dict[str, Any]],
    velocity_rows: list[dict[str, Any]],
    grid_rows: list[dict[str, Any]],
    nsf_cells: int,
) -> dict[str, Any]:
    largest = max(scan.particle_counts)
    largest_velocity = [
        row for row in velocity_rows if int(row["particle_count"]) == largest
    ]
    return {
        "schema_version": 1,
        "experiment": "shock_tube_enskog_navier_stokes_fourier_comparison",
        "status": "completed",
        "scope": (
            "first-Sonine Navier--Stokes-order Enskog transport with "
            "Carnahan--Starling equilibrium pressure; not a full nonlocal "
            "Enskog kinetic solver"
        ),
        "method": {
            "nsf_cells": nsf_cells,
            "hyperbolic_flux": "MUSCL Rusanov with minmod and SSP-RK2",
            "dissipative_flux": (
                "centered face gradients with tau_xx=(4 mu/3+zeta) du/dx "
                "and q_x=-kappa dT/dx"
            ),
            "time_step": (
                "minimum of hyperbolic CFL, explicit diffusion CFL, and "
                "next output time"
            ),
            "transport_closure": {
                "source": (
                    "Montanero and Santos, Physical Review E 54, 438 "
                    "(1996), equations 2.15--2.21"
                ),
                "doi": "10.1103/PhysRevE.54.438",
                "contact_pair_correlation": (
                    "chi=(1-eta/2)/(1-eta)^3"
                ),
                "normal_stress": (
                    "tau_xx=(4*mu/3+zeta)*du/dx"
                ),
                "heat_flux": "q_x=-kappa*dT/dx",
                "transport_scale": 1.0,
                "shear_viscosity_scale": 1.0,
                "bulk_viscosity_scale": 1.0,
                "thermal_conductivity_scale": 1.0,
            },
            "analysis_tool_sha256": sha256_file(Path(__file__)),
            "nsf_reference_tool_sha256": sha256_file(
                Path(__file__).with_name("enskog_nsf_reference.py")
            ),
            "enskog_euler_tool_sha256": sha256_file(
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
        "largest_particle_count_field_comparison": largest_particle_table(
            scan, field_rows
        ),
        "largest_particle_count_velocity_diagnostics": largest_velocity,
        "reference_grid_check": grid_rows,
        "transport": [item.transport_row for item in analyses],
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
    rows = [
        {
            "particle_count": 8,
            "target_knudsen": 0.1,
            "field": "bulk_velocity_x",
            "reference_model": model,
            "relative_l2": value,
        }
        for model, value in (
            ("ideal_euler", 0.3),
            ("enskog_euler_carnahan_starling", 0.25),
            ("enskog_navier_stokes_fourier", 0.1),
        )
    ]
    fake = type("Scan", (), {"particle_counts": (8,), "knudsen_numbers": (0.1,)})
    table = largest_particle_table(fake, rows)
    if not math.isclose(
        float(table[0]["bulk_velocity_x_enskog_nsf_relative_l2"]),
        0.1,
    ):
        raise AssertionError("three-model comparison table self-test failed")
    print("Enskog NSF comparison analysis self-test")
    print("  three-model table: passed")
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
        else scan_output / "enskog-nsf-comparison"
    )
    output.mkdir(parents=True, exist_ok=True)
    analyses = []
    cached: dict[float, EnskogNsfSolution] = {}
    for cell in derive_cells(scan):
        cell_output = scan_output / "cells" / cell.cell_id
        if not cell_is_complete(scan, cell, cell_output):
            raise ValueError(f"incomplete scan cell: {cell.cell_id}")
        print(
            f"comparing {cell.cell_id} "
            f"(N={cell.particle_count}, Kn={cell.target_knudsen:g})"
        )
        item = analyze_cell(
            scan,
            cell,
            cell_output,
            arguments.nsf_cells,
            cached.get(cell.target_knudsen),
        )
        cached.setdefault(cell.target_knudsen, item.nsf_solution)
        analyses.append(item)

    field_rows = [
        row
        for item in analyses
        for row in (*item.base.field_rows, *item.field_rows)
    ]
    velocity_rows = [
        row
        for item in analyses
        for row in (*item.base.velocity_rows, item.velocity_row)
    ]
    transport_rows = [item.transport_row for item in analyses]
    grid_rows = reference_grid_check(
        scan,
        scan_output,
        analyses,
        arguments.nsf_cells,
    )
    field_path = output / "field-comparison.csv"
    velocity_path = output / "velocity-diagnostics.csv"
    transport_path = output / "transport-coefficients.csv"
    grid_path = output / "reference-grid-check.csv"
    summary_path = output / "summary.json"
    fields_plot_path = output / "enskog-nsf-field-comparison.png"
    errors_plot_path = output / "model-error-comparison.png"
    velocity_plot_path = output / "velocity-profile-comparison.png"
    write_csv(field_path, field_rows)
    write_csv(velocity_path, velocity_rows)
    write_csv(transport_path, transport_rows)
    write_csv(grid_path, grid_rows)
    summary = build_summary(
        scan,
        scan_output,
        analyses,
        field_rows,
        velocity_rows,
        grid_rows,
        arguments.nsf_cells,
    )
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    plot_field_comparison(fields_plot_path, scan, analyses)
    plot_error_comparison(errors_plot_path, scan, field_rows)
    plot_velocity_profiles(velocity_plot_path, scan, analyses)

    print("Enskog Navier--Stokes--Fourier comparison")
    print(f"  analyzed cells: {len(analyses)}")
    print(f"  unique NSF solves: {len(cached)}")
    print(f"  NSF cells: {arguments.nsf_cells}")
    print(f"  field table: {field_path}")
    print(f"  velocity diagnostics: {velocity_path}")
    print(f"  transport coefficients: {transport_path}")
    print(f"  reference grid check: {grid_path}")
    print(f"  summary JSON: {summary_path}")
    print(f"  field plot: {fields_plot_path}")
    print(f"  error plot: {errors_plot_path}")
    print(f"  velocity plot: {velocity_plot_path}")
    print("  status: completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, csv.Error) as exception:
        raise SystemExit(f"Enskog NSF comparison: {exception}") from exception
