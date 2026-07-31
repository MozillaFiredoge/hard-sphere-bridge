#!/usr/bin/env python3
"""Extract shock-tube wave markers and compare their propagation speeds.

The particle and Euler profiles are smoothed over the same physical length and
processed independently. The reported locations are reproducible numerical
markers, not exact discontinuity locations:

* rarefaction: centroid around the largest positive velocity gradient left of
  the initial interface;
* contact: centroid around the largest entropy-proxy gradient between the
  interface and the compression marker;
* compression: centroid around the largest negative log-pressure gradient
  right of the initial interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "particles-matplotlib")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


BACKGROUND = "#090d18"
PANEL = "#111827"
FOREGROUND = "#e5e7eb"
MUTED = "#94a3b8"
GRID = "#334155"
PARTICLE_COLOR = "#38bdf8"
EULER_COLOR = "#f59e0b"
FEATURE_COLORS = {
    "rarefaction": "#a78bfa",
    "contact": "#22c55e",
    "compression": "#f97316",
}
FEATURE_LABELS = {
    "rarefaction": "rarefaction gradient",
    "contact": "contact",
    "compression": "compression",
}


@dataclass(frozen=True)
class FieldFrame:
    time: float
    x: np.ndarray
    number_density: np.ndarray
    bulk_velocity_x: np.ndarray
    pressure: np.ndarray


@dataclass(frozen=True)
class ExtractionConfig:
    gamma: float
    smoothing_length: float
    centroid_weight_power: float
    rarefaction_min_x: float
    rarefaction_interface_margin: float
    rarefaction_centroid_half_width: float
    shock_interface_margin: float
    right_max_x: float
    shock_centroid_half_width: float
    contact_interface_margin: float
    contact_shock_separation: float
    contact_centroid_half_width: float
    fit_start_time: float
    sensitivity_smoothing_lengths: tuple[float, ...]
    maximum_mean_absolute_position_error: float
    maximum_relative_speed_error: float
    maximum_particle_fit_rmse: float


@dataclass(frozen=True)
class FeaturePositions:
    rarefaction: float
    contact: float
    compression: float


@dataclass(frozen=True)
class SpeedFit:
    speed: float
    rmse: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract rarefaction/contact/compression markers from synchronized "
            "particle and Euler shock-tube fields."
        )
    )
    parser.add_argument(
        "particle_csv",
        nargs="?",
        type=Path,
        help="macroscopic.csv written by particles_cli shock-tube",
    )
    parser.add_argument(
        "euler_csv",
        nargs="?",
        type=Path,
        help="euler-reference.csv written by visualize_shock_tube.py",
    )
    parser.add_argument(
        "--simulation-config",
        type=Path,
        default=Path("configs/shock_tube.json"),
        help="shock-tube JSON containing initial_condition.interface_x",
    )
    parser.add_argument(
        "--analysis-config",
        type=Path,
        default=Path("configs/wave_kinematics.json"),
        help="wave-marker definitions, fit interval, and acceptance thresholds",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/wave-kinematics"),
        help="directory for CSV, JSON, and PNG results",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a deterministic synthetic kinematics check and exit",
    )
    return parser.parse_args()


def require_number(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if non_negative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_extraction_config(path: Path) -> ExtractionConfig:
    root = load_json_object(path)
    if root.get("experiment") != "wave_kinematics":
        raise ValueError(f"{path}: experiment must be wave_kinematics")
    if root.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    features = root.get("features")
    fit = root.get("fit")
    acceptance = root.get("acceptance")
    sensitivity = root.get("sensitivity")
    if not all(
        isinstance(value, dict)
        for value in (features, fit, sensitivity, acceptance)
    ):
        raise ValueError(
            f"{path}: features, fit, sensitivity, and acceptance must be objects"
        )
    assert isinstance(features, dict)
    assert isinstance(fit, dict)
    assert isinstance(sensitivity, dict)
    assert isinstance(acceptance, dict)
    sensitivity_values = sensitivity.get("smoothing_lengths")
    if (
        not isinstance(sensitivity_values, list)
        or len(sensitivity_values) < 2
    ):
        raise ValueError(
            f"{path}: sensitivity.smoothing_lengths requires at least two values"
        )
    config = ExtractionConfig(
        gamma=require_number(root.get("gamma"), "gamma", positive=True),
        smoothing_length=require_number(
            root.get("smoothing_length"),
            "smoothing_length",
            positive=True,
        ),
        centroid_weight_power=require_number(
            root.get("centroid_weight_power"),
            "centroid_weight_power",
            positive=True,
        ),
        rarefaction_min_x=require_number(
            features.get("rarefaction_min_x"),
            "features.rarefaction_min_x",
        ),
        rarefaction_interface_margin=require_number(
            features.get("rarefaction_interface_margin"),
            "features.rarefaction_interface_margin",
            non_negative=True,
        ),
        rarefaction_centroid_half_width=require_number(
            features.get("rarefaction_centroid_half_width"),
            "features.rarefaction_centroid_half_width",
            positive=True,
        ),
        shock_interface_margin=require_number(
            features.get("shock_interface_margin"),
            "features.shock_interface_margin",
            non_negative=True,
        ),
        right_max_x=require_number(
            features.get("right_max_x"),
            "features.right_max_x",
        ),
        shock_centroid_half_width=require_number(
            features.get("shock_centroid_half_width"),
            "features.shock_centroid_half_width",
            positive=True,
        ),
        contact_interface_margin=require_number(
            features.get("contact_interface_margin"),
            "features.contact_interface_margin",
            non_negative=True,
        ),
        contact_shock_separation=require_number(
            features.get("contact_shock_separation"),
            "features.contact_shock_separation",
            positive=True,
        ),
        contact_centroid_half_width=require_number(
            features.get("contact_centroid_half_width"),
            "features.contact_centroid_half_width",
            positive=True,
        ),
        fit_start_time=require_number(
            fit.get("start_time"),
            "fit.start_time",
            non_negative=True,
        ),
        sensitivity_smoothing_lengths=tuple(
            require_number(
                value,
                f"sensitivity.smoothing_lengths[{index}]",
                positive=True,
            )
            for index, value in enumerate(sensitivity_values)
        ),
        maximum_mean_absolute_position_error=require_number(
            acceptance.get("maximum_mean_absolute_position_error"),
            "acceptance.maximum_mean_absolute_position_error",
            positive=True,
        ),
        maximum_relative_speed_error=require_number(
            acceptance.get("maximum_relative_speed_error"),
            "acceptance.maximum_relative_speed_error",
            positive=True,
        ),
        maximum_particle_fit_rmse=require_number(
            acceptance.get("maximum_particle_fit_rmse"),
            "acceptance.maximum_particle_fit_rmse",
            positive=True,
        ),
    )
    if config.rarefaction_min_x >= config.right_max_x:
        raise ValueError("feature search interval is empty")
    if len(set(config.sensitivity_smoothing_lengths)) != len(
        config.sensitivity_smoothing_lengths
    ):
        raise ValueError("sensitivity smoothing lengths must be unique")
    return config


def load_interface_x(path: Path) -> float:
    root = load_json_object(path)
    initial = root.get("initial_condition")
    if not isinstance(initial, dict):
        raise ValueError(f"{path}: initial_condition must be an object")
    return require_number(
        initial.get("interface_x"),
        "initial_condition.interface_x",
    )


def load_frames(path: Path) -> dict[float, FieldFrame]:
    rows_by_time: dict[float, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "time",
            "x_center",
            "number_density",
            "bulk_velocity_x",
            "pressure",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        for row in reader:
            rows_by_time[float(row["time"])].append(row)
    if not rows_by_time:
        raise ValueError(f"{path}: no field rows")

    frames: dict[float, FieldFrame] = {}
    for time, rows in rows_by_time.items():
        rows.sort(key=lambda row: float(row["x_center"]))
        x = np.array([float(row["x_center"]) for row in rows])
        number_density = np.array(
            [float(row["number_density"]) for row in rows]
        )
        bulk_velocity_x = np.array(
            [float(row["bulk_velocity_x"]) for row in rows]
        )
        pressure = np.array([float(row["pressure"]) for row in rows])
        if x.size < 16:
            raise ValueError(f"{path}: time {time:g} has fewer than 16 cells")
        if not (
            np.all(np.isfinite(x))
            and np.all(np.isfinite(number_density))
            and np.all(np.isfinite(bulk_velocity_x))
            and np.all(np.isfinite(pressure))
        ):
            raise ValueError(f"{path}: time {time:g} contains non-finite values")
        if not np.all(np.diff(x) > 0.0):
            raise ValueError(f"{path}: x coordinates must increase")
        if np.any(number_density <= 0.0) or np.any(pressure <= 0.0):
            raise ValueError(
                f"{path}: time {time:g} requires positive density and pressure"
            )
        spacing = np.diff(x)
        if not np.allclose(spacing, np.median(spacing), rtol=1.0e-3):
            raise ValueError(f"{path}: time {time:g} grid must be uniform")
        frames[time] = FieldFrame(
            time=time,
            x=x,
            number_density=number_density,
            bulk_velocity_x=bulk_velocity_x,
            pressure=pressure,
        )
    return frames


def gaussian_smooth(
    values: np.ndarray,
    x: np.ndarray,
    physical_sigma: float,
) -> np.ndarray:
    spacing = float(np.median(np.diff(x)))
    sigma_cells = max(physical_sigma / spacing, 0.5)
    radius = max(2, int(math.ceil(4.0 * sigma_cells)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(offsets / sigma_cells))
    kernel /= np.sum(kernel)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def weighted_peak_centroid(
    x: np.ndarray,
    score: np.ndarray,
    mask: np.ndarray,
    half_width: float,
    power: float,
    feature: str,
) -> float:
    if not np.any(mask):
        raise ValueError(f"{feature}: empty search interval")
    candidate_x = x[mask]
    candidate_score = np.maximum(score[mask], 0.0)
    peak_index = int(np.argmax(candidate_score))
    if candidate_score[peak_index] <= 0.0:
        raise ValueError(f"{feature}: no positive feature score")
    peak_x = candidate_x[peak_index]
    local = np.abs(candidate_x - peak_x) <= half_width
    weights = np.power(candidate_score[local], power)
    weight_sum = float(np.sum(weights))
    if not math.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError(f"{feature}: invalid centroid weights")
    return float(np.sum(candidate_x[local] * weights) / weight_sum)


def extract_features(
    frame: FieldFrame,
    interface_x: float,
    config: ExtractionConfig,
) -> FeaturePositions:
    x = frame.x
    density = gaussian_smooth(
        frame.number_density,
        x,
        config.smoothing_length,
    )
    velocity = gaussian_smooth(
        frame.bulk_velocity_x,
        x,
        config.smoothing_length,
    )
    pressure = gaussian_smooth(
        frame.pressure,
        x,
        config.smoothing_length,
    )

    log_pressure = np.log(pressure)
    entropy_proxy = log_pressure - config.gamma * np.log(density)
    velocity_gradient = np.gradient(velocity, x)
    log_pressure_gradient = np.gradient(log_pressure, x)
    entropy_gradient = np.gradient(entropy_proxy, x)

    rarefaction_mask = (
        (x >= config.rarefaction_min_x)
        & (x <= interface_x - config.rarefaction_interface_margin)
    )
    rarefaction = weighted_peak_centroid(
        x,
        velocity_gradient,
        rarefaction_mask,
        config.rarefaction_centroid_half_width,
        config.centroid_weight_power,
        "rarefaction",
    )

    compression_mask = (
        (x >= interface_x + config.shock_interface_margin)
        & (x <= config.right_max_x)
    )
    compression = weighted_peak_centroid(
        x,
        -log_pressure_gradient,
        compression_mask,
        config.shock_centroid_half_width,
        config.centroid_weight_power,
        "compression",
    )

    contact_mask = (
        (x >= interface_x + config.contact_interface_margin)
        & (x <= compression - config.contact_shock_separation)
    )
    contact = weighted_peak_centroid(
        x,
        np.abs(entropy_gradient),
        contact_mask,
        config.contact_centroid_half_width,
        config.centroid_weight_power,
        "contact",
    )
    return FeaturePositions(
        rarefaction=rarefaction,
        contact=contact,
        compression=compression,
    )


def common_fit_times(
    particle_frames: dict[float, FieldFrame],
    euler_frames: dict[float, FieldFrame],
    fit_start_time: float,
) -> list[float]:
    times = sorted(
        time
        for time in particle_frames.keys() & euler_frames.keys()
        if time + 1.0e-12 >= fit_start_time
    )
    if len(times) < 3:
        raise ValueError(
            "at least three synchronized frames are required at or after "
            f"fit.start_time={fit_start_time:g}"
        )
    return times


def anchored_speed_fit(
    times: np.ndarray,
    positions: np.ndarray,
    interface_x: float,
) -> SpeedFit:
    denominator = float(np.dot(times, times))
    if denominator <= 0.0:
        raise ValueError("speed fit requires positive times")
    speed = float(np.dot(times, positions - interface_x) / denominator)
    residual = positions - (interface_x + speed * times)
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    return SpeedFit(speed=speed, rmse=rmse)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(
    particle_frames: dict[float, FieldFrame],
    euler_frames: dict[float, FieldFrame],
    interface_x: float,
    config: ExtractionConfig,
) -> tuple[
    list[float],
    dict[str, dict[float, float]],
    dict[str, dict[float, float]],
    dict[str, dict[str, float | bool]],
    bool,
]:
    times = common_fit_times(
        particle_frames,
        euler_frames,
        config.fit_start_time,
    )
    particle_positions = {
        feature: {} for feature in FEATURE_LABELS
    }
    euler_positions = {
        feature: {} for feature in FEATURE_LABELS
    }
    for time in times:
        particle = extract_features(
            particle_frames[time],
            interface_x,
            config,
        )
        euler = extract_features(
            euler_frames[time],
            interface_x,
            config,
        )
        for feature in FEATURE_LABELS:
            particle_positions[feature][time] = float(
                getattr(particle, feature)
            )
            euler_positions[feature][time] = float(getattr(euler, feature))

    time_array = np.array(times)
    summary: dict[str, dict[str, float | bool]] = {}
    passed = True
    for feature in FEATURE_LABELS:
        particle_values = np.array(
            [particle_positions[feature][time] for time in times]
        )
        euler_values = np.array(
            [euler_positions[feature][time] for time in times]
        )
        particle_fit = anchored_speed_fit(
            time_array,
            particle_values,
            interface_x,
        )
        euler_fit = anchored_speed_fit(
            time_array,
            euler_values,
            interface_x,
        )
        differences = particle_values - euler_values
        relative_speed_error = abs(
            particle_fit.speed - euler_fit.speed
        ) / max(abs(euler_fit.speed), 1.0e-15)
        mean_absolute_position_error = float(np.mean(np.abs(differences)))
        maximum_absolute_position_error = float(np.max(np.abs(differences)))
        feature_passed = (
            mean_absolute_position_error
            <= config.maximum_mean_absolute_position_error
            and relative_speed_error <= config.maximum_relative_speed_error
            and particle_fit.rmse <= config.maximum_particle_fit_rmse
        )
        passed = passed and feature_passed
        summary[feature] = {
            "particle_speed": particle_fit.speed,
            "euler_speed": euler_fit.speed,
            "relative_speed_error": relative_speed_error,
            "particle_fit_rmse": particle_fit.rmse,
            "euler_fit_rmse": euler_fit.rmse,
            "mean_absolute_position_error": mean_absolute_position_error,
            "maximum_absolute_position_error": maximum_absolute_position_error,
            "passed": feature_passed,
        }
    return (
        times,
        particle_positions,
        euler_positions,
        summary,
        passed,
    )


def write_positions_csv(
    path: Path,
    times: list[float],
    particle_positions: dict[str, dict[float, float]],
    euler_positions: dict[str, dict[float, float]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time",
                "feature",
                "particle_position",
                "euler_position",
                "position_error",
            ]
        )
        for time in times:
            for feature in FEATURE_LABELS:
                particle = particle_positions[feature][time]
                euler = euler_positions[feature][time]
                writer.writerow([time, feature, particle, euler, particle - euler])


def write_speeds_csv(
    path: Path,
    fit_start_time: float,
    fit_end_time: float,
    sample_count: int,
    summary: dict[str, dict[str, float | bool]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "feature",
                "fit_start_time",
                "fit_end_time",
                "sample_count",
                "particle_speed",
                "euler_speed",
                "relative_speed_error",
                "particle_fit_rmse",
                "euler_fit_rmse",
                "mean_absolute_position_error",
                "maximum_absolute_position_error",
                "passed",
            ]
        )
        for feature, metrics in summary.items():
            writer.writerow(
                [
                    feature,
                    fit_start_time,
                    fit_end_time,
                    sample_count,
                    metrics["particle_speed"],
                    metrics["euler_speed"],
                    metrics["relative_speed_error"],
                    metrics["particle_fit_rmse"],
                    metrics["euler_fit_rmse"],
                    metrics["mean_absolute_position_error"],
                    metrics["maximum_absolute_position_error"],
                    "yes" if metrics["passed"] else "no",
                ]
            )


def analyze_smoothing_sensitivity(
    particle_frames: dict[float, FieldFrame],
    euler_frames: dict[float, FieldFrame],
    interface_x: float,
    config: ExtractionConfig,
) -> list[dict[str, float | str | bool]]:
    rows: list[dict[str, float | str | bool]] = []
    for smoothing_length in config.sensitivity_smoothing_lengths:
        sensitivity_config = replace(
            config,
            smoothing_length=smoothing_length,
        )
        _, _, _, summary, passed = analyze(
            particle_frames,
            euler_frames,
            interface_x,
            sensitivity_config,
        )
        for feature, metrics in summary.items():
            rows.append(
                {
                    "smoothing_length": smoothing_length,
                    "feature": feature,
                    "particle_speed": float(metrics["particle_speed"]),
                    "euler_speed": float(metrics["euler_speed"]),
                    "relative_speed_error": float(
                        metrics["relative_speed_error"]
                    ),
                    "mean_absolute_position_error": float(
                        metrics["mean_absolute_position_error"]
                    ),
                    "particle_fit_rmse": float(
                        metrics["particle_fit_rmse"]
                    ),
                    "feature_passed": bool(metrics["passed"]),
                    "scale_passed": passed,
                }
            )
    return rows


def write_sensitivity_csv(
    path: Path,
    rows: list[dict[str, float | str | bool]],
) -> None:
    fieldnames = [
        "smoothing_length",
        "feature",
        "particle_speed",
        "euler_speed",
        "relative_speed_error",
        "mean_absolute_position_error",
        "particle_fit_rmse",
        "feature_passed",
        "scale_passed",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "feature_passed": (
                        "yes" if row["feature_passed"] else "no"
                    ),
                    "scale_passed": "yes" if row["scale_passed"] else "no",
                }
            )


def write_summary_json(
    path: Path,
    particle_csv: Path,
    euler_csv: Path,
    simulation_config: Path,
    analysis_config: Path,
    interface_x: float,
    times: list[float],
    config: ExtractionConfig,
    summary: dict[str, dict[str, float | bool]],
    sensitivity_rows: list[dict[str, float | str | bool]],
    passed: bool,
) -> None:
    document = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "method": {
            "particle_and_euler_processed_independently": True,
            "analysis_tool_sha256": sha256_file(Path(__file__)),
            "smoothing_length": config.smoothing_length,
            "centroid_weight_power": config.centroid_weight_power,
            "speed_fit": "x(t) = interface_x + speed * t",
            "interface_x": interface_x,
            "fit_start_time": times[0],
            "fit_end_time": times[-1],
            "sample_count": len(times),
        },
        "inputs": {
            "particle_csv": {
                "path": str(particle_csv),
                "sha256": sha256_file(particle_csv),
            },
            "euler_csv": {
                "path": str(euler_csv),
                "sha256": sha256_file(euler_csv),
            },
            "simulation_config": {
                "path": str(simulation_config),
                "sha256": sha256_file(simulation_config),
            },
            "analysis_config": {
                "path": str(analysis_config),
                "sha256": sha256_file(analysis_config),
            },
        },
        "acceptance": {
            "maximum_mean_absolute_position_error": (
                config.maximum_mean_absolute_position_error
            ),
            "maximum_relative_speed_error": (
                config.maximum_relative_speed_error
            ),
            "maximum_particle_fit_rmse": config.maximum_particle_fit_rmse,
        },
        "features": summary,
        "smoothing_sensitivity": sensitivity_rows,
    }
    with path.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")


def style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(PANEL)
    axis.tick_params(colors=FOREGROUND)
    for spine in axis.spines.values():
        spine.set_color(GRID)
    axis.grid(color=GRID, alpha=0.45, linewidth=0.8)
    axis.xaxis.label.set_color(FOREGROUND)
    axis.yaxis.label.set_color(FOREGROUND)
    axis.title.set_color(FOREGROUND)


def write_plot(
    path: Path,
    times: list[float],
    particle_positions: dict[str, dict[float, float]],
    euler_positions: dict[str, dict[float, float]],
    summary: dict[str, dict[str, float | bool]],
    interface_x: float,
    passed: bool,
) -> None:
    time_array = np.array(times)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 6.4),
        facecolor=BACKGROUND,
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.86,
        bottom=0.18,
        wspace=0.2,
    )
    for axis in axes:
        style_axis(axis)

    axes[0].axhline(
        interface_x,
        color=MUTED,
        linestyle=":",
        linewidth=1.0,
        label="initial interface",
    )
    for feature, color in FEATURE_COLORS.items():
        particle = np.array(
            [particle_positions[feature][time] for time in times]
        )
        euler = np.array([euler_positions[feature][time] for time in times])
        axes[0].plot(
            time_array,
            particle,
            marker="o",
            color=color,
            linewidth=2.0,
            label=f"particles — {FEATURE_LABELS[feature]}",
        )
        axes[0].plot(
            time_array,
            euler,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Euler — {FEATURE_LABELS[feature]}",
        )
        axes[1].plot(
            time_array,
            particle - euler,
            marker="o",
            color=color,
            linewidth=2.0,
            label=FEATURE_LABELS[feature],
        )

    axes[0].set_title("Wave-marker trajectories")
    axes[0].set_xlabel("simulation time")
    axes[0].set_ylabel("x position")
    axes[0].legend(
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=FOREGROUND,
        fontsize=8,
        ncol=2,
    )
    axes[1].axhline(0.0, color=MUTED, linewidth=1.0)
    axes[1].set_title("Particle − Euler position")
    axes[1].set_xlabel("simulation time")
    axes[1].set_ylabel("position difference")
    axes[1].legend(
        facecolor=PANEL,
        edgecolor=GRID,
        labelcolor=FOREGROUND,
        fontsize=8,
    )

    summary_lines = [
        (
            f"{FEATURE_LABELS[feature]}: "
            f"cₚ={float(metrics['particle_speed']):+.3f}, "
            f"cₑ={float(metrics['euler_speed']):+.3f}, "
            f"Δc={100.0 * float(metrics['relative_speed_error']):.1f}%"
        )
        for feature, metrics in summary.items()
    ]
    summary_lines.append(f"acceptance: {'PASS' if passed else 'FAIL'}")
    figure.text(
        0.5,
        0.035,
        "   |   ".join(summary_lines),
        ha="center",
        va="bottom",
        color=FOREGROUND,
        fontsize=9,
    )
    figure.suptitle(
        "Hard-sphere shock tube: wave kinematics",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def write_sensitivity_plot(
    path: Path,
    rows: list[dict[str, float | str | bool]],
    config: ExtractionConfig,
) -> None:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.4),
        facecolor=BACKGROUND,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.82,
        bottom=0.15,
        wspace=0.2,
    )
    for axis in axes:
        style_axis(axis)
    for feature, color in FEATURE_COLORS.items():
        feature_rows = [
            row for row in rows if row["feature"] == feature
        ]
        lengths = np.array(
            [float(row["smoothing_length"]) for row in feature_rows]
        )
        speed_errors = np.array(
            [
                100.0 * float(row["relative_speed_error"])
                for row in feature_rows
            ]
        )
        position_errors = np.array(
            [
                float(row["mean_absolute_position_error"])
                for row in feature_rows
            ]
        )
        axes[0].plot(
            lengths,
            speed_errors,
            marker="o",
            linewidth=2.0,
            color=color,
            label=FEATURE_LABELS[feature],
        )
        axes[1].plot(
            lengths,
            position_errors,
            marker="o",
            linewidth=2.0,
            color=color,
            label=FEATURE_LABELS[feature],
        )
    axes[0].axhline(
        100.0 * config.maximum_relative_speed_error,
        color=MUTED,
        linestyle="--",
        linewidth=1.2,
        label="acceptance limit",
    )
    axes[1].axhline(
        config.maximum_mean_absolute_position_error,
        color=MUTED,
        linestyle="--",
        linewidth=1.2,
        label="acceptance limit",
    )
    axes[0].set_title("Speed sensitivity")
    axes[0].set_xlabel("Gaussian smoothing length")
    axes[0].set_ylabel("relative speed difference (%)")
    axes[1].set_title("Position sensitivity")
    axes[1].set_xlabel("Gaussian smoothing length")
    axes[1].set_ylabel("mean absolute position difference")
    for axis in axes:
        axis.legend(
            facecolor=PANEL,
            edgecolor=GRID,
            labelcolor=FOREGROUND,
            fontsize=8,
        )
    figure.suptitle(
        "Wave-marker coarse-graining sensitivity",
        color=FOREGROUND,
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 0.5 * (1.0 + np.tanh(clipped))


def synthetic_frames(
    cells: int,
    times: list[float],
    interface_x: float,
    noisy: bool,
) -> dict[float, FieldFrame]:
    x = (np.arange(cells, dtype=np.float64) + 0.5) * (4.0 / cells)
    speeds = {
        "rarefaction": -0.8,
        "contact": 0.7,
        "compression": 1.7,
    }
    frames = {}
    for time in times:
        rarefaction = interface_x + speeds["rarefaction"] * time
        contact = interface_x + speeds["contact"] * time
        compression = interface_x + speeds["compression"] * time
        log_pressure = (
            -0.25 * sigmoid((x - rarefaction) / 0.08)
            - 1.10 * sigmoid((x - compression) / 0.025)
        )
        pressure = np.exp(log_pressure)
        entropy = (
            0.35 * sigmoid((x - contact) / 0.025)
            + 0.04 * sigmoid((x - compression) / 0.02)
        )
        number_density = np.exp(
            (np.log(pressure) - entropy) / (5.0 / 3.0)
        )
        velocity = 0.22 * sigmoid((x - rarefaction) / 0.07)
        if noisy:
            phase = 13.0 * x + 37.0 * time
            pressure *= 1.0 + 0.006 * np.sin(phase)
            number_density *= 1.0 + 0.008 * np.cos(1.7 * phase)
            velocity += 0.002 * np.sin(2.3 * phase)
        frames[time] = FieldFrame(
            time=time,
            x=x,
            number_density=number_density,
            bulk_velocity_x=velocity,
            pressure=pressure,
        )
    return frames


def run_self_test() -> int:
    times = [0.12, 0.14, 0.16, 0.18, 0.20]
    interface_x = 2.0
    config = ExtractionConfig(
        gamma=5.0 / 3.0,
        smoothing_length=0.04,
        centroid_weight_power=2.0,
        rarefaction_min_x=1.2,
        rarefaction_interface_margin=0.01,
        rarefaction_centroid_half_width=0.1,
        shock_interface_margin=0.02,
        right_max_x=2.8,
        shock_centroid_half_width=0.08,
        contact_interface_margin=0.0,
        contact_shock_separation=0.06,
        contact_centroid_half_width=0.06,
        fit_start_time=0.12,
        sensitivity_smoothing_lengths=(0.03, 0.04, 0.05),
        maximum_mean_absolute_position_error=0.02,
        maximum_relative_speed_error=0.04,
        maximum_particle_fit_rmse=0.01,
    )
    particle_frames = synthetic_frames(256, times, interface_x, noisy=True)
    euler_frames = synthetic_frames(2048, times, interface_x, noisy=False)
    _, _, _, summary, passed = analyze(
        particle_frames,
        euler_frames,
        interface_x,
        config,
    )
    expected_speeds = {
        "rarefaction": -0.8,
        "contact": 0.7,
        "compression": 1.7,
    }
    for feature, expected in expected_speeds.items():
        particle_speed = float(summary[feature]["particle_speed"])
        if abs(particle_speed - expected) / abs(expected) > 0.06:
            raise AssertionError(
                f"{feature} synthetic speed {particle_speed:g} "
                f"differs from {expected:g}"
            )
    if not passed:
        raise AssertionError("synthetic particle/Euler comparison did not pass")
    print("wave-kinematics self-test")
    for feature, metrics in summary.items():
        print(
            f"  {feature}: particle speed="
            f"{float(metrics['particle_speed']):.8g}, "
            f"Euler speed={float(metrics['euler_speed']):.8g}, "
            f"relative error="
            f"{float(metrics['relative_speed_error']):.3g}"
        )
    print("  status: passed")
    return 0


def main() -> int:
    arguments = parse_arguments()
    if arguments.self_test:
        return run_self_test()
    if arguments.particle_csv is None or arguments.euler_csv is None:
        raise ValueError(
            "particle_csv and euler_csv are required unless --self-test is used"
        )

    config = load_extraction_config(arguments.analysis_config)
    interface_x = load_interface_x(arguments.simulation_config)
    particle_frames = load_frames(arguments.particle_csv)
    euler_frames = load_frames(arguments.euler_csv)
    (
        times,
        particle_positions,
        euler_positions,
        summary,
        passed,
    ) = analyze(
        particle_frames,
        euler_frames,
        interface_x,
        config,
    )
    sensitivity_rows = analyze_smoothing_sensitivity(
        particle_frames,
        euler_frames,
        interface_x,
        config,
    )

    arguments.output.mkdir(parents=True, exist_ok=True)
    positions_path = arguments.output / "wave-positions.csv"
    speeds_path = arguments.output / "wave-speeds.csv"
    sensitivity_path = arguments.output / "smoothing-sensitivity.csv"
    summary_path = arguments.output / "summary.json"
    plot_path = arguments.output / "wave-kinematics.png"
    sensitivity_plot_path = (
        arguments.output / "smoothing-sensitivity.png"
    )
    write_positions_csv(
        positions_path,
        times,
        particle_positions,
        euler_positions,
    )
    write_speeds_csv(
        speeds_path,
        times[0],
        times[-1],
        len(times),
        summary,
    )
    write_sensitivity_csv(sensitivity_path, sensitivity_rows)
    write_summary_json(
        summary_path,
        arguments.particle_csv,
        arguments.euler_csv,
        arguments.simulation_config,
        arguments.analysis_config,
        interface_x,
        times,
        config,
        summary,
        sensitivity_rows,
        passed,
    )
    write_plot(
        plot_path,
        times,
        particle_positions,
        euler_positions,
        summary,
        interface_x,
        passed,
    )
    write_sensitivity_plot(
        sensitivity_plot_path,
        sensitivity_rows,
        config,
    )

    print("shock-tube wave-kinematics experiment")
    print(f"  fit frames: {len(times)} ({times[0]:g} to {times[-1]:g})")
    print(f"  smoothing length: {config.smoothing_length:g}")
    for feature, metrics in summary.items():
        print(
            f"  {feature}: particle speed="
            f"{float(metrics['particle_speed']):.8g}, "
            f"Euler speed={float(metrics['euler_speed']):.8g}, "
            f"relative speed error="
            f"{float(metrics['relative_speed_error']):.6g}, "
            f"mean |position error|="
            f"{float(metrics['mean_absolute_position_error']):.6g}, "
            f"fit rms={float(metrics['particle_fit_rmse']):.6g}, "
            f"status={'passed' if metrics['passed'] else 'failed'}"
        )
    print(f"  positions CSV: {positions_path}")
    print(f"  speeds CSV: {speeds_path}")
    passing_scales = sorted(
        {
            float(row["smoothing_length"])
            for row in sensitivity_rows
            if bool(row["scale_passed"])
        }
    )
    print(
        "  sensitivity passing scales: "
        + (
            ", ".join(f"{value:g}" for value in passing_scales)
            if passing_scales
            else "none"
        )
    )
    print(f"  sensitivity CSV: {sensitivity_path}")
    print(f"  summary JSON: {summary_path}")
    print(f"  plot: {plot_path}")
    print(f"  sensitivity plot: {sensitivity_plot_path}")
    print(f"  overall status: {'passed' if passed else 'failed'}")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, csv.Error) as exception:
        raise SystemExit(f"wave-kinematics: {exception}") from exception
