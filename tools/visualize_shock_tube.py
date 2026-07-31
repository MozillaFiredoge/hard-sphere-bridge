#!/usr/bin/env python3
"""Render synchronized micro/meso/macro shock-tube diagnostics.

The renderer consumes the CSV files written by ``particles_cli shock-tube``.
It never interpolates particle states: one video frame is produced for each
recorded simulation frame.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "particles-matplotlib")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from euler_reference import (  # noqa: E402
    EulerFrame,
    EulerParameters,
    EulerSolution,
    solve_euler_reference,
    write_euler_csv,
)


BACKGROUND = "#090d18"
PANEL = "#111827"
FOREGROUND = "#e5e7eb"
MUTED = "#94a3b8"
GRID = "#334155"
CURRENT = "#38bdf8"
REFERENCE = "#a1a1aa"
EULER = "#fbbf24"
INTERFACE = "#f59e0b"

REQUIRED_FILES = (
    "evolution.csv",
    "summary.csv",
    "macroscopic.csv",
    "phase_space.csv",
    "microscopic.csv",
)


@dataclass
class FrameData:
    frame: int
    time: float
    microscopic: dict[str, np.ndarray]
    phase_counts: np.ndarray
    phase_extent: tuple[float, float, float, float]
    macroscopic: dict[str, np.ndarray]
    evolution: dict[str, str]
    summary: dict[str, str]


@dataclass
class VisualizationData:
    frames: list[FrameData]
    particle_count: int
    microscopic_samples: int
    phase_x_bins: int
    phase_vx_bins: int
    macroscopic_slabs: int
    domain_extent: tuple[float, float, float]
    interface_x: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render synchronized microscopic particles, x-vx phase space, "
            "and macroscopic shock-tube fields from simulator CSV output."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="directory containing the five shock-tube CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output directory (default: INPUT/visualization)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/shock_tube.json"),
        help="simulation JSON used for domain and interface annotations",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="recorded frames per video second; no temporal interpolation",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="frame resolution scale; 120 produces 1920x1080",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="render PNG files but do not invoke ffmpeg",
    )
    parser.add_argument(
        "--euler-cells",
        type=int,
        default=2048,
        help="finite-volume cells in the ideal-gas Euler reference",
    )
    parser.add_argument(
        "--euler-cfl",
        type=float,
        default=0.45,
        help="CFL number for the ideal-gas Euler reference",
    )
    parser.add_argument(
        "--no-euler",
        action="store_true",
        help="omit the Euler reference and reproduce particle-only views",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    return rows


def group_by_frame(rows: list[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(int(row["frame"]), []).append(row)
    return grouped


def load_config(config_path: Path) -> tuple[tuple[float, float, float], float]:
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    extent = tuple(float(value) for value in config["domain"]["extent"])
    if len(extent) != 3:
        raise ValueError("domain.extent must contain exactly three values")
    interface_x = float(config["initial_condition"]["interface_x"])
    return (extent[0], extent[1], extent[2]), interface_x


def as_array(rows: list[dict[str, str]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows], dtype=np.float64)


def build_frame(
    frame: int,
    microscopic_rows: list[dict[str, str]],
    phase_rows: list[dict[str, str]],
    macro_rows: list[dict[str, str]],
    evolution: dict[str, str],
    summary: dict[str, str],
) -> FrameData:
    microscopic_rows.sort(key=lambda row: int(row["particle_index"]))
    macro_rows.sort(key=lambda row: int(row["slab"]))

    microscopic = {
        field: as_array(microscopic_rows, field)
        for field in (
            "position_x",
            "position_y",
            "position_z",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "speed",
        )
    }
    macroscopic = {
        field: as_array(macro_rows, field)
        for field in (
            "x_center",
            "particle_count",
            "number_density",
            "bulk_velocity_x",
            "temperature",
            "pressure",
        )
    }

    phase_x_bins = max(int(row["x_bin"]) for row in phase_rows) + 1
    phase_vx_bins = max(int(row["velocity_x_bin"]) for row in phase_rows) + 1
    if len(phase_rows) != phase_x_bins * phase_vx_bins:
        raise ValueError(
            f"frame {frame}: phase-space grid is incomplete "
            f"({len(phase_rows)} rows for {phase_x_bins}x{phase_vx_bins})"
        )
    phase_counts = np.zeros((phase_vx_bins, phase_x_bins), dtype=np.float64)
    for row in phase_rows:
        phase_counts[int(row["velocity_x_bin"]), int(row["x_bin"])] = int(
            row["count"]
        )
    phase_extent = (
        min(float(row["x_lower"]) for row in phase_rows),
        max(float(row["x_upper"]) for row in phase_rows),
        min(float(row["velocity_x_lower"]) for row in phase_rows),
        max(float(row["velocity_x_upper"]) for row in phase_rows),
    )

    times = {
        float(microscopic_rows[0]["time"]),
        float(phase_rows[0]["time"]),
        float(macro_rows[0]["time"]),
        float(evolution["time"]),
        float(summary["time"]),
    }
    if max(times) - min(times) > 1.0e-12:
        raise ValueError(f"frame {frame}: diagnostic times are not synchronized: {times}")

    return FrameData(
        frame=frame,
        time=float(summary["time"]),
        microscopic=microscopic,
        phase_counts=phase_counts,
        phase_extent=phase_extent,
        macroscopic=macroscopic,
        evolution=evolution,
        summary=summary,
    )


def load_data(input_dir: Path, config_path: Path) -> VisualizationData:
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{input_dir} is missing required files: {', '.join(missing)}"
        )

    domain_extent, interface_x = load_config(config_path)
    microscopic = group_by_frame(read_csv(input_dir / "microscopic.csv"))
    phase = group_by_frame(read_csv(input_dir / "phase_space.csv"))
    macro = group_by_frame(read_csv(input_dir / "macroscopic.csv"))
    evolution_rows = read_csv(input_dir / "evolution.csv")
    summary_rows = read_csv(input_dir / "summary.csv")
    evolution = {int(row["frame"]): row for row in evolution_rows}
    summary = {int(row["frame"]): row for row in summary_rows}

    frame_sets = {
        "microscopic": set(microscopic),
        "phase_space": set(phase),
        "macroscopic": set(macro),
        "evolution": set(evolution),
        "summary": set(summary),
    }
    expected = set(next(iter(frame_sets.values())))
    mismatches = {
        name: sorted(frames)
        for name, frames in frame_sets.items()
        if frames != expected
    }
    if mismatches:
        raise ValueError(f"CSV frame sets do not match: {mismatches}")

    frames = [
        build_frame(
            frame,
            microscopic[frame],
            phase[frame],
            macro[frame],
            evolution[frame],
            summary[frame],
        )
        for frame in sorted(expected)
    ]
    if not frames:
        raise ValueError("no synchronized frames found")

    first = frames[0]
    particle_count = int(first.summary["particle_count"])
    return VisualizationData(
        frames=frames,
        particle_count=particle_count,
        microscopic_samples=len(first.microscopic["position_x"]),
        phase_x_bins=first.phase_counts.shape[1],
        phase_vx_bins=first.phase_counts.shape[0],
        macroscopic_slabs=len(first.macroscopic["x_center"]),
        domain_extent=domain_extent,
        interface_x=interface_x,
    )


def style_figure(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(BACKGROUND)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.xaxis.label.set_color(FOREGROUND)
    ax.yaxis.label.set_color(FOREGROUND)
    ax.title.set_color(FOREGROUND)
    ax.grid(color=GRID, linewidth=0.55, alpha=0.45)


def padded_limits(values: np.ndarray, fraction: float = 0.07) -> tuple[float, float]:
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    span = high - low
    if span <= 0.0:
        span = max(abs(low), 1.0)
    return low - fraction * span, high + fraction * span


def macro_limits(
    data: VisualizationData, euler: EulerSolution | None
) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for field in (
        "number_density",
        "bulk_velocity_x",
        "temperature",
        "pressure",
    ):
        values = [frame.macroscopic[field] for frame in data.frames]
        if euler is not None:
            values.extend(frame.fields[field] for frame in euler.frames)
        limits[field] = padded_limits(np.concatenate(values))
    return limits


def add_interface(ax: plt.Axes, interface_x: float) -> None:
    ax.axvline(
        interface_x,
        color=INTERFACE,
        linestyle=(0, (3, 3)),
        linewidth=1.0,
        alpha=0.75,
    )


def render_macroscopic_evolution(
    data: VisualizationData,
    euler: EulerSolution | None,
    output_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), dpi=dpi)
    style_figure(fig)
    fields = (
        ("number_density", r"number density  $n$"),
        ("bulk_velocity_x", r"bulk velocity  $u_x$"),
        ("temperature", r"temperature  $T$"),
        ("pressure", r"pressure  $p$"),
    )
    selected_indices = np.unique(
        np.linspace(0, len(data.frames) - 1, min(6, len(data.frames)), dtype=int)
    )
    color_map = plt.get_cmap("turbo")
    time_max = max(frame.time for frame in data.frames)
    time_norm = Normalize(vmin=0.0, vmax=max(time_max, 1.0e-12))
    euler_by_frame = (
        {frame.frame: frame for frame in euler.frames}
        if euler is not None
        else {}
    )

    for ax, (field, label) in zip(axes.flat, fields):
        style_axis(ax)
        for index in selected_indices:
            frame = data.frames[int(index)]
            color = color_map(time_norm(frame.time))
            ax.plot(
                frame.macroscopic["x_center"],
                frame.macroscopic[field],
                color=color,
                linewidth=1.55,
                label=f"t = {frame.time:.2f}",
            )
            euler_frame = euler_by_frame.get(frame.frame)
            if euler_frame is not None:
                ax.plot(
                    euler_frame.fields["x_center"],
                    euler_frame.fields[field],
                    color=color,
                    linewidth=1.15,
                    linestyle=(0, (5, 3)),
                    alpha=0.95,
                )
        add_interface(ax, data.interface_x)
        ax.set_xlim(0.0, data.domain_extent[0])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(label)
        ax.yaxis.set_major_locator(MaxNLocator(5))

    axes[0, 0].legend(
        frameon=False,
        fontsize=8,
        labelcolor=FOREGROUND,
        ncol=2,
        loc="best",
    )
    if euler is not None:
        axes[0, 1].legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=FOREGROUND,
                    linewidth=1.6,
                    label="hard-sphere moments",
                ),
                Line2D(
                    [0],
                    [0],
                    color=FOREGROUND,
                    linewidth=1.2,
                    linestyle=(0, (5, 3)),
                    label=r"ideal Euler, $\gamma=5/3$",
                ),
            ],
            frameon=False,
            fontsize=8,
            labelcolor=FOREGROUND,
            loc="upper left",
        )
    fig.suptitle(
        (
            "Hard-sphere fluid moments compared with an ideal-gas Euler reference"
            if euler is not None
            else "Macroscopic fields derived from Newtonian hard-sphere states"
        ),
        color=FOREGROUND,
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        (
            f"N = {data.particle_count:,}   |   {data.macroscopic_slabs} slabs"
            f"   |   orange dashed line: initial interface x = {data.interface_x:g}"
            "   |   no temporal smoothing"
            + (
                f"   |   Euler: {euler.parameters.cells} cells, "
                f"CFL {euler.parameters.cfl:g}"
                if euler is not None
                else ""
            )
        ),
        color=MUTED,
        fontsize=9,
        ha="center",
    )
    fig.tight_layout(rect=(0.025, 0.04, 0.98, 0.95))
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_frame(
    data: VisualizationData,
    frame: FrameData,
    output_path: Path,
    dpi: int,
    phase_max: float,
    velocity_limit: float,
    limits: dict[str, tuple[float, float]],
    cumulative_collisions: int,
    euler_frame: EulerFrame | None,
    euler_parameters: EulerParameters | None,
) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    style_figure(fig)
    grid = fig.add_gridspec(
        4,
        3,
        width_ratios=(1.1, 1.1, 1.0),
        left=0.055,
        right=0.975,
        bottom=0.12,
        top=0.82,
        wspace=0.25,
        hspace=0.28,
    )
    micro_ax = fig.add_subplot(grid[:, 0])
    phase_ax = fig.add_subplot(grid[:, 1])
    macro_axes = [fig.add_subplot(grid[row, 2]) for row in range(4)]

    style_axis(micro_ax)
    micro = frame.microscopic
    scatter = micro_ax.scatter(
        micro["position_x"],
        micro["position_y"],
        c=micro["velocity_x"],
        cmap="coolwarm",
        norm=Normalize(vmin=-velocity_limit, vmax=velocity_limit),
        s=8,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
    )
    add_interface(micro_ax, data.interface_x)
    micro_ax.set_xlim(0.0, data.domain_extent[0])
    micro_ax.set_ylim(0.0, data.domain_extent[1])
    micro_ax.set_aspect("auto")
    micro_ax.set_xlabel(r"$x$")
    micro_ax.set_ylabel(r"$y$  (periodic)")
    micro_ax.text(
        0.98,
        0.018,
        "display aspect expanded; coordinate limits are physical",
        transform=micro_ax.transAxes,
        color=MUTED,
        fontsize=7.5,
        ha="right",
        va="bottom",
    )
    micro_ax.set_title(
        "MICROSCOPIC\n"
        f"{data.microscopic_samples:,} stable samples from N = {data.particle_count:,}",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    colorbar = fig.colorbar(
        scatter, ax=micro_ax, orientation="horizontal", pad=0.12, fraction=0.035
    )
    colorbar.set_label(r"particle velocity $v_x$", color=FOREGROUND, fontsize=9)
    colorbar.ax.tick_params(colors=MUTED, labelsize=8)
    colorbar.outline.set_edgecolor(GRID)

    style_axis(phase_ax)
    positive_counts = np.ma.masked_less(frame.phase_counts, 1.0)
    image = phase_ax.imshow(
        positive_counts,
        origin="lower",
        extent=frame.phase_extent,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        norm=LogNorm(vmin=1.0, vmax=max(phase_max, 1.0)),
        rasterized=True,
    )
    add_interface(phase_ax, data.interface_x)
    phase_ax.set_xlim(0.0, data.domain_extent[0])
    phase_ax.set_ylim(frame.phase_extent[2], frame.phase_extent[3])
    phase_ax.set_xlabel(r"$x$")
    phase_ax.set_ylabel(r"$v_x$")
    clipped_velocity_samples = int(frame.summary["phase_space_underflow"]) + int(
        frame.summary["phase_space_overflow"]
    )
    phase_ax.set_title(
        "MESOSCOPIC\n"
        rf"empirical $\widehat{{f}}_N(x,v_x,t)$; "
        f"N = {data.particle_count:,}, clipped velocity tails = "
        f"{clipped_velocity_samples:,}",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    colorbar = fig.colorbar(
        image, ax=phase_ax, orientation="horizontal", pad=0.12, fraction=0.035
    )
    colorbar.set_label("particles per phase-space bin (log scale)", color=FOREGROUND)
    colorbar.ax.tick_params(colors=MUTED, labelsize=8)
    colorbar.outline.set_edgecolor(GRID)

    macro_fields = (
        ("number_density", r"$n(x)$"),
        ("bulk_velocity_x", r"$u_x(x)$"),
        ("temperature", r"$T(x)$"),
        ("pressure", r"$p(x)$"),
    )
    for index, (ax, (field, label)) in enumerate(zip(macro_axes, macro_fields)):
        style_axis(ax)
        ax.plot(
            frame.macroscopic["x_center"],
            frame.macroscopic[field],
            color=CURRENT,
            linewidth=1.55,
            label=f"hard spheres, t = {frame.time:.2f}",
        )
        if euler_frame is not None:
            ax.plot(
                euler_frame.fields["x_center"],
                euler_frame.fields[field],
                color=EULER,
                linewidth=1.25,
                linestyle=(0, (5, 3)),
                alpha=0.95,
                label=r"ideal Euler, $\gamma=5/3$",
            )
        else:
            initial = data.frames[0]
            ax.plot(
                initial.macroscopic["x_center"],
                initial.macroscopic[field],
                color=REFERENCE,
                linewidth=1.0,
                linestyle=(0, (3, 3)),
                alpha=0.8,
                label="t = 0",
            )
        add_interface(ax, data.interface_x)
        ax.set_xlim(0.0, data.domain_extent[0])
        ax.set_ylim(*limits[field])
        ax.set_ylabel(label, fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if index == 0:
            ax.set_title(
                "MACROSCOPIC\n"
                + (
                    f"{data.macroscopic_slabs}-slab particle moments vs "
                    "ideal Euler"
                    if euler_frame is not None
                    else (
                        "moments over all particles in "
                        f"{data.macroscopic_slabs} slabs"
                    )
                ),
                fontsize=11,
                fontweight="bold",
                pad=8,
            )
            ax.legend(
                frameon=False,
                fontsize=7.5,
                labelcolor=FOREGROUND,
                loc="best",
            )
        if index == len(macro_axes) - 1:
            ax.set_xlabel(r"$x$")
        else:
            ax.tick_params(labelbottom=False)

    status = frame.evolution["status"]
    relative_energy_drift = float(frame.evolution["relative_energy_drift"])
    minimum_gap = float(frame.evolution["minimum_surface_gap"])
    gap_text = "n/a" if not math.isfinite(minimum_gap) else f"{minimum_gap:.2e}"
    fig.suptitle(
        r"Newtonian hard spheres  $\longrightarrow$  empirical phase space"
        r"  $\longrightarrow$  fluid moments",
        color=FOREGROUND,
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5,
        0.905,
        f"recorded frame {frame.frame}   |   t = {frame.time:.2f}",
        color=FOREGROUND,
        fontsize=13,
        fontweight="bold",
        ha="center",
    )
    fig.text(
        0.5,
        0.055,
        (
            f"status: {status}    |    cumulative accepted collisions: "
            f"{cumulative_collisions:,}    |    interval rollbacks: "
            f"{int(frame.evolution['rollbacks']):,}    |    exact recoveries: "
            f"{int(frame.evolution['exact_recoveries']):,}"
        ),
        color=FOREGROUND,
        fontsize=10,
        ha="center",
    )
    fig.text(
        0.5,
        0.027,
        (
            f"committed minimum surface gap: {gap_text}    |    "
            f"relative kinetic-energy drift: {relative_energy_drift:.2e}    |    "
            "orange dashed line: initial interface"
            + (
                f"    |    Euler numerical reference: "
                f"{euler_parameters.cells} cells, CFL {euler_parameters.cfl:g}"
                if euler_parameters is not None
                else ""
            )
        ),
        color=MUTED,
        fontsize=9,
        ha="center",
    )
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def encode_video(
    frames_dir: Path, output_path: Path, fps: float, frame_count: int
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found; rerun with --no-video")
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:g}",
        "-i",
        str(frames_dir / "frame-%03d.png"),
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def write_euler_comparison_csv(
    data: VisualizationData,
    euler: EulerSolution,
    output_path: Path,
) -> dict[str, float]:
    euler_by_frame = {frame.frame: frame for frame in euler.frames}
    fields = (
        "number_density",
        "bulk_velocity_x",
        "temperature",
        "pressure",
    )
    columns = (
        "frame",
        "time",
        "field",
        "rms_error",
        "relative_l2_error",
        "mean_absolute_error",
        "maximum_absolute_error",
        "particle_minimum",
        "particle_maximum",
        "euler_minimum",
        "euler_maximum",
    )
    final_relative_errors: dict[str, float] = {}
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for particle_frame in data.frames:
            euler_frame = euler_by_frame[particle_frame.frame]
            particle_x = particle_frame.macroscopic["x_center"]
            euler_x = euler_frame.fields["x_center"]
            for field in fields:
                particle_values = particle_frame.macroscopic[field]
                euler_values = np.interp(
                    particle_x,
                    euler_x,
                    euler_frame.fields[field],
                )
                difference = particle_values - euler_values
                rms_error = float(np.sqrt(np.mean(difference**2)))
                euler_l2 = float(np.sqrt(np.mean(euler_values**2)))
                relative_l2_error = rms_error / max(
                    euler_l2, np.finfo(np.float64).tiny
                )
                writer.writerow(
                    {
                        "frame": particle_frame.frame,
                        "time": f"{particle_frame.time:.17g}",
                        "field": field,
                        "rms_error": f"{rms_error:.17g}",
                        "relative_l2_error": f"{relative_l2_error:.17g}",
                        "mean_absolute_error": (
                            f"{float(np.mean(np.abs(difference))):.17g}"
                        ),
                        "maximum_absolute_error": (
                            f"{float(np.max(np.abs(difference))):.17g}"
                        ),
                        "particle_minimum": (
                            f"{float(np.min(particle_values)):.17g}"
                        ),
                        "particle_maximum": (
                            f"{float(np.max(particle_values)):.17g}"
                        ),
                        "euler_minimum": (
                            f"{float(np.min(euler_values)):.17g}"
                        ),
                        "euler_maximum": (
                            f"{float(np.max(euler_values)):.17g}"
                        ),
                    }
                )
                if particle_frame is data.frames[-1]:
                    final_relative_errors[field] = relative_l2_error
    return final_relative_errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(arguments: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    output = completed.stdout.strip()
    return output if output else None


def portable_path(path: Path, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repository_root))
    except ValueError:
        return str(resolved)


def sanitized_repository_remote(remote: str | None) -> str | None:
    if remote is None:
        return None
    if "://" in remote:
        parsed = urlsplit(remote)
        if (
            parsed.scheme not in {"git", "http", "https", "ssh"}
            or parsed.hostname is None
        ):
            return None
        host = parsed.hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    if ":" in remote and not remote.startswith(("/", ".")):
        authority, path = remote.split(":", 1)
        host = authority.rsplit("@", 1)[-1]
        if host and path:
            return f"{host}:{path}"
    return None


def reproducibility_metadata(
    input_dir: Path,
    config_path: Path,
    video_written: bool,
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[1]
    git_commit = command_output(
        ["git", "rev-parse", "HEAD"],
        repository_root,
    )
    git_status = command_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        repository_root,
    )
    remote = command_output(
        ["git", "remote", "get-url", "origin"],
        repository_root,
    )
    ffmpeg_version: str | None = None
    if video_written:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is not None:
            version_output = command_output([ffmpeg, "-version"], repository_root)
            if version_output is not None:
                ffmpeg_version = version_output.splitlines()[0]

    source_files = {
        name: {
            "bytes": (input_dir / name).stat().st_size,
            "sha256": sha256_file(input_dir / name),
        }
        for name in REQUIRED_FILES
    }
    simulation_metadata_path = input_dir / "run-metadata.json"
    simulation_metadata: dict[str, Any] | None = None
    if simulation_metadata_path.is_file():
        with simulation_metadata_path.open(encoding="utf-8") as stream:
            simulation_metadata = json.load(stream)
    return {
        "repository": sanitized_repository_remote(remote),
        "renderer_git_commit": git_commit,
        "renderer_git_worktree_dirty": git_status is not None,
        "simulation_run_metadata": simulation_metadata,
        "source_directory": portable_path(input_dir, repository_root),
        "config": {
            "path": portable_path(config_path, repository_root),
            "bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
        },
        "source_csv": source_files,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "ffmpeg": ffmpeg_version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "renderer": {
            "path": portable_path(Path(__file__), repository_root),
            "sha256": sha256_file(Path(__file__)),
        },
    }


def write_manifest(
    output_path: Path,
    input_dir: Path,
    config_path: Path,
    data: VisualizationData,
    fps: float,
    video_written: bool,
    euler: EulerSolution | None,
    euler_final_relative_l2_errors: dict[str, float] | None,
) -> None:
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "reproducibility": reproducibility_metadata(
            input_dir,
            config_path,
            video_written,
        ),
        "recorded_frames": len(data.frames),
        "frame_numbers": [frame.frame for frame in data.frames],
        "simulation_times": [frame.time for frame in data.frames],
        "temporal_interpolation": False,
        "video_recorded_frames_per_second": fps,
        "particle_count": data.particle_count,
        "microscopic_samples_per_frame": data.microscopic_samples,
        "phase_space_particles_per_frame": data.particle_count,
        "phase_space_bins": [data.phase_x_bins, data.phase_vx_bins],
        "macroscopic_particles_per_frame": data.particle_count,
        "macroscopic_slabs": data.macroscopic_slabs,
        "euler_reference": (
            {
                "enabled": True,
                "model": "one-dimensional monatomic ideal-gas Euler",
                "gamma": euler.parameters.gamma,
                "cells": euler.parameters.cells,
                "cfl": euler.parameters.cfl,
                "limiter_theta": euler.parameters.limiter_theta,
                "scheme": "MUSCL + minmod + Rusanov + SSP-RK2",
                "longitudinal_boundary": "reflecting",
                "initial_condition": "analytic smooth profile from config",
                "participates_in_particle_advancement": False,
                "steps": euler.steps,
                "relative_mass_drift": euler.relative_mass_drift,
                "relative_energy_drift": euler.relative_energy_drift,
                "minimum_density": euler.minimum_density,
                "minimum_pressure": euler.minimum_pressure,
                "final_frame_relative_l2_errors": (
                    euler_final_relative_l2_errors
                ),
            }
            if euler is not None
            else {"enabled": False}
        ),
        "artifacts": {
            "macroscopic_evolution": "macroscopic-evolution.png",
            "final_overview": "final-overview.png",
            "frames": "frames/frame-000.png ...",
            "video": "shock-tube.mp4" if video_written else None,
            "euler_reference": "euler-reference.csv" if euler is not None else None,
            "euler_comparison": (
                "euler-comparison.csv" if euler is not None else None
            ),
        },
    }
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def main() -> int:
    args = parse_args()
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if not args.no_euler and args.euler_cells < 8:
        raise ValueError("--euler-cells must be at least eight")
    if not args.no_euler and not 0.0 < args.euler_cfl < 1.0:
        raise ValueError("--euler-cfl must lie strictly between zero and one")

    input_dir = args.input.resolve()
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else (input_dir / "visualization")
    )
    config_path = args.config.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(input_dir, config_path)
    print(
        f"loaded {len(data.frames)} synchronized frames, "
        f"N={data.particle_count:,}"
    )

    euler: EulerSolution | None = None
    euler_final_relative_l2_errors: dict[str, float] | None = None
    if not args.no_euler:
        euler = solve_euler_reference(
            config_path,
            [(frame.frame, frame.time) for frame in data.frames],
            particle_count=data.particle_count,
            parameters=EulerParameters(
                cells=args.euler_cells,
                cfl=args.euler_cfl,
            ),
        )
        euler_path = output_dir / "euler-reference.csv"
        write_euler_csv(euler, euler_path)
        euler_comparison_path = output_dir / "euler-comparison.csv"
        euler_final_relative_l2_errors = write_euler_comparison_csv(
            data,
            euler,
            euler_comparison_path,
        )
        print(
            f"Euler reference: cells={euler.parameters.cells}, "
            f"steps={euler.steps}, "
            f"mass drift={euler.relative_mass_drift:.3e}, "
            f"energy drift={euler.relative_energy_drift:.3e}"
        )
        print(f"wrote {euler_path}")
        print(f"wrote {euler_comparison_path}")
        print(
            "final relative L2 errors: "
            + ", ".join(
                f"{field}={value:.4g}"
                for field, value in euler_final_relative_l2_errors.items()
            )
        )
    euler_by_frame = (
        {frame.frame: frame for frame in euler.frames}
        if euler is not None
        else {}
    )

    macro_path = output_dir / "macroscopic-evolution.png"
    render_macroscopic_evolution(data, euler, macro_path, args.dpi)
    print(f"wrote {macro_path}")

    phase_max = max(float(np.max(frame.phase_counts)) for frame in data.frames)
    sampled_velocity_max = max(
        float(np.max(np.abs(frame.microscopic["velocity_x"])))
        for frame in data.frames
    )
    velocity_limit = max(
        abs(data.frames[0].phase_extent[2]),
        abs(data.frames[0].phase_extent[3]),
        sampled_velocity_max,
    )
    limits = macro_limits(data, euler)
    cumulative_collisions = 0
    final_frame_path: Path | None = None
    for sequence, frame in enumerate(data.frames):
        cumulative_collisions += int(frame.evolution["accepted_collisions"])
        frame_path = frames_dir / f"frame-{sequence:03d}.png"
        render_frame(
            data,
            frame,
            frame_path,
            args.dpi,
            phase_max,
            velocity_limit,
            limits,
            cumulative_collisions,
            euler_by_frame.get(frame.frame),
            euler.parameters if euler is not None else None,
        )
        final_frame_path = frame_path
        print(
            f"wrote {frame_path.name} "
            f"(simulation frame {frame.frame}, t={frame.time:g})"
        )

    if final_frame_path is None:
        raise RuntimeError("no frames were rendered")
    final_overview = output_dir / "final-overview.png"
    shutil.copyfile(final_frame_path, final_overview)
    print(f"wrote {final_overview}")

    video_written = False
    video_path = output_dir / "shock-tube.mp4"
    if not args.no_video:
        encode_video(frames_dir, video_path, args.fps, len(data.frames))
        video_written = True
        print(f"wrote {video_path}")

    manifest_path = output_dir / "manifest.json"
    write_manifest(
        manifest_path,
        input_dir,
        config_path,
        data,
        args.fps,
        video_written,
        euler,
        euler_final_relative_l2_errors,
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"visualization failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
