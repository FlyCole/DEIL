#!/usr/bin/env python3
"""Plot per-task success-rate sensitivity to additive depth noise."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASKS = (
    ("phone_on_base", "Phone on Base"),
    ("open_box", "Open Box"),
    ("close_box", "Close Box"),
    ("plate_out", "Plate Out"),
    ("basketball", "Basketball"),
    ("close_microwave", "Close Microwave"),
    ("toilet_seat_down", "Toilet Seat Down"),
    ("toilet_seat_up", "Toilet Seat Up"),
    ("toilet_roll_off", "Toilet Roll Off"),
    ("meat_off_grill", "Meat Off Grill"),
)
NOISE_LEVELS_MM = (0, 10, 20, 30, 40)
NOISY_ESTIMATOR = re.compile(r"gt_noisy_(\d+(?:p\d+)?)mm$")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Draw all ten task success-rate curves for GT and 10--40 mm "
            "additive depth noise."
        )
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=script_dir / "stats",
        help="Directory containing per-run best_stats.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "outputs" / "task_level" / "depth_noise_sensitivity",
        help="Directory for the PNG, PDF, and source-data CSV.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    return parser.parse_args()


def estimator_noise_mm(estimator: str) -> int | None:
    if estimator == "gt":
        return 0
    match = NOISY_ESTIMATOR.fullmatch(estimator)
    if match is None:
        return None
    value = float(match.group(1).replace("p", "."))
    if not value.is_integer():
        return None
    return int(value)


def load_results(stats_dir: Path) -> dict[tuple[str, int], dict]:
    expected_tasks = {task for task, _ in TASKS}
    results: dict[tuple[str, int], dict] = {}

    for stats_path in sorted(stats_dir.glob("*/*/best_stats.json")):
        with stats_path.open("r", encoding="utf-8") as stats_file:
            record = json.load(stats_file)

        task = str(record.get("task_name", ""))
        noise_mm = estimator_noise_mm(str(record.get("depth_estimator", "")))
        if task not in expected_tasks or noise_mm not in NOISE_LEVELS_MM:
            continue

        key = (task, noise_mm)
        if key in results:
            raise RuntimeError(
                f"Duplicate task/noise result for {task}, {noise_mm} mm: "
                f"{results[key]['source_json']} and {stats_path}"
            )

        rollout_count = int(record["rollout_count"])
        success_count = int(record["success_count"])
        success_rate = float(record["success_rate"])
        if rollout_count <= 0 or not 0 <= success_count <= rollout_count:
            raise RuntimeError(f"Invalid rollout counts in {stats_path}")
        count_rate = success_count / rollout_count
        if not np.isclose(success_rate, count_rate):
            raise RuntimeError(
                f"Success-rate/count mismatch in {stats_path}: "
                f"{success_rate} versus {count_rate}"
            )

        results[key] = {
            "task_name": task,
            "noise_std_mm": noise_mm,
            "depth_estimator": record["depth_estimator"],
            "success_count": success_count,
            "rollout_count": rollout_count,
            "success_rate": success_rate,
            "run_name": record["run_name"],
            "source_json": str(stats_path.resolve()),
        }

    missing = [
        (task, noise_mm)
        for task, _ in TASKS
        for noise_mm in NOISE_LEVELS_MM
        if (task, noise_mm) not in results
    ]
    if missing:
        formatted = ", ".join(f"{task}@{noise}mm" for task, noise in missing)
        raise RuntimeError(f"Missing task/noise results: {formatted}")
    return results


def write_source_data(results: dict[tuple[str, int], dict], csv_path: Path) -> None:
    fieldnames = (
        "task_name",
        "task_label",
        "noise_std_mm",
        "depth_estimator",
        "success_count",
        "rollout_count",
        "success_rate",
        "success_rate_percent",
        "run_name",
        "source_json",
    )
    labels = dict(TASKS)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for task, _ in TASKS:
            for noise_mm in NOISE_LEVELS_MM:
                row = results[(task, noise_mm)].copy()
                row["task_label"] = labels[task]
                row["success_rate_percent"] = 100.0 * row["success_rate"]
                writer.writerow(row)


def draw_figure(
    results: dict[tuple[str, int], dict],
    png_path: Path,
    pdf_path: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 15,
            "axes.labelsize": 12.5,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = plt.get_cmap("tab10").colors
    markers = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "*")
    figure, axis = plt.subplots(figsize=(12.8, 6.7))

    x = np.asarray(NOISE_LEVELS_MM, dtype=float)
    for index, (task, label) in enumerate(TASKS):
        y = 100.0 * np.asarray(
            [results[(task, level)]["success_rate"] for level in NOISE_LEVELS_MM]
        )
        axis.plot(
            x,
            y,
            color=colors[index],
            marker=markers[index],
            markersize=6.2,
            markeredgecolor="white",
            markeredgewidth=0.65,
            linewidth=2.1,
            alpha=0.9,
            zorder=2,
            label=label,
        )

    axis.set_xlim(-1.5, 41.5)
    axis.set_ylim(0, 104)
    axis.set_xticks(NOISE_LEVELS_MM, ("0\n(GT)", "10", "20", "30", "40"))
    axis.set_yticks(np.arange(0, 101, 20))
    axis.set_xlabel("Additive depth-noise standard deviation (mm)")
    axis.set_ylabel("Task success rate (%)")
    axis.set_title("Task-dependent sensitivity to additive depth noise", pad=13)
    axis.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.8)
    axis.grid(axis="x", color="#e5e7eb", linewidth=0.65, alpha=0.65)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    legend = axis.legend(
        title="Task",
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.8,
    )
    figure.subplots_adjust(left=0.085, right=0.77, bottom=0.12, top=0.89)
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if not args.stats_dir.is_dir():
        raise FileNotFoundError(f"Stats directory not found: {args.stats_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "task_depth_noise_sensitivity_curves"
    png_path = args.output_dir / f"{stem}.png"
    pdf_path = args.output_dir / f"{stem}.pdf"
    csv_path = args.output_dir / f"{stem}.csv"

    results = load_results(args.stats_dir)
    write_source_data(results, csv_path)
    draw_figure(results, png_path, pdf_path, args.dpi)

    print(f"Validated {len(results)} task/noise results.")
    print(f"PNG: {png_path.resolve()}")
    print(f"PDF: {pdf_path.resolve()}")
    print(f"CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
