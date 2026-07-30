#!/usr/bin/env python3
"""Render synchronized micro/meso/macro shock-tube diagnostics.

The renderer consumes the CSV files written by ``particles_cli shock-tube``.
It never interpolates particle states: one video frame is produced for each
recorded simulation frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
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
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


BACKGROUND = "#090d18"
PANEL = "#111827"
FOREGROUND = "#e5e7eb"
MUTED = "#94a3b8"
GRID = "#334155"
CURRENT = "#38bdf8"
REFERENCE = "#a1a1aa"
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


def macro_limits(data: VisualizationData) -> dict[str, tuple[float, float]]:
    return {
        field: padded_limits(
            np.concatenate([frame.macroscopic[field] for frame in data.frames])
        )
        for field in (
            "number_density",
            "bulk_velocity_x",
            "temperature",
            "pressure",
        )
    }


def add_interface(ax: plt.Axes, interface_x: float) -> None:
    ax.axvline(
        interface_x,
        color=INTERFACE,
        linestyle=(0, (3, 3)),
        linewidth=1.0,
        alpha=0.75,
    )


def render_macroscopic_evolution(
    data: VisualizationData, output_path: Path, dpi: int
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
    fig.suptitle(
        "Macroscopic fields derived from the same Newtonian hard-sphere states",
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
    initial = data.frames[0]
    for index, (ax, (field, label)) in enumerate(zip(macro_axes, macro_fields)):
        style_axis(ax)
        ax.plot(
            initial.macroscopic["x_center"],
            initial.macroscopic[field],
            color=REFERENCE,
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            alpha=0.8,
            label="t = 0",
        )
        ax.plot(
            frame.macroscopic["x_center"],
            frame.macroscopic[field],
            color=CURRENT,
            linewidth=1.55,
            label=f"t = {frame.time:.2f}",
        )
        add_interface(ax, data.interface_x)
        ax.set_xlim(0.0, data.domain_extent[0])
        ax.set_ylim(*limits[field])
        ax.set_ylabel(label, fontsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if index == 0:
            ax.set_title(
                "MACROSCOPIC\n"
                f"moments over all particles in {data.macroscopic_slabs} slabs",
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
        ),
        color=MUTED,
        fontsize=9,
        ha="center",
    )
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def encode_video(frames_dir: Path, output_path: Path, fps: float) -> None:
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


def write_manifest(
    output_path: Path,
    input_dir: Path,
    config_path: Path,
    data: VisualizationData,
    fps: float,
    video_written: bool,
) -> None:
    manifest: dict[str, Any] = {
        "source_directory": str(input_dir.resolve()),
        "config": str(config_path.resolve()),
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
        "artifacts": {
            "macroscopic_evolution": "macroscopic-evolution.png",
            "final_overview": "final-overview.png",
            "frames": "frames/frame-000.png ...",
            "video": "shock-tube.mp4" if video_written else None,
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

    macro_path = output_dir / "macroscopic-evolution.png"
    render_macroscopic_evolution(data, macro_path, args.dpi)
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
    limits = macro_limits(data)
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
        encode_video(frames_dir, video_path, args.fps)
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
    )
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"visualization failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
