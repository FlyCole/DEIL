#!/usr/bin/env python3
import os
import re
import json
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.optimize import curve_fit
from scipy.stats import spearmanr


# ---------------------------
# Helpers
# ---------------------------
# Display-name mapping (must match CSV column names as keys)
DISPLAY_NAMES = {
    'Pi3':          r'$\pi$3',
    'VGGT':         'VGGT',
    'MoGe':         'MoGe',
    'Mast3r':       'Mast3r',
    'MapAnything':  'MapAnything',
    'Depth':        'Depth',
}

def _disp(label: str) -> str:
    return DISPLAY_NAMES.get(label, label)


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def build_color_map(labels):
    labels = [str(l) for l in labels]
    unique = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab10")
    colors, idx = {}, 0
    for lab in unique:
        low = lab.lower()
        if "mapanything" in low or "map_anything" in low:
            colors[lab] = "red"
        else:
            colors[lab] = cmap(idx % 10)
            idx += 1
    return colors


# ---------------------------
# Load stats
# ---------------------------
def load_chamfer_mean_from_rlb_jsons(rlb_json_dir: str) -> dict:
    paths = sorted(glob(os.path.join(rlb_json_dir, "rlb_*.json")))
    if not paths:
        raise FileNotFoundError(f"No rlb_*.json files found in: {rlb_json_dir}")

    per_method_vals = {}
    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)
        metrics = data.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue

        for method_key, stats in metrics.items():
            if not isinstance(stats, dict):
                continue
            v = stats.get("mean", None)
            if v is None:
                continue
            try:
                v = float(v)
            except Exception:
                continue
            per_method_vals.setdefault(str(method_key), []).append(v)

    out = {k: float(np.mean(np.asarray(v, dtype=float))) for k, v in per_method_vals.items() if v}
    if not out:
        raise RuntimeError("No chamfer means found in rlb_*.json files.")
    return out


def load_success_mean_std_from_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)

    # Exclude derived summary rows so they are not counted as additional tasks.
    if "Task Name" in df.columns:
        task_names = df["Task Name"].astype(str).str.strip().str.casefold()
        df = df.loc[~task_names.isin({"avg", "average"})]

    # Rename RGB-D column to Depth if present (keeps the same data)
    if "RGB-D" in df.columns:
        df = df.rename(columns={"RGB-D": "Depth"})

    method_cols = [c for c in df.columns if c != "Task Name"] if "Task Name" in df.columns else list(df.columns)

    out = {}
    for c in method_cols:
        vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        out[str(c)] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "count": int(vals.size),
        }
    if not out:
        raise RuntimeError("No numeric success-rate columns found in the CSV.")
    return out


def match_methods_yerr(chamfer_mean: dict, success_stats: dict):
    """
    Match JSON method keys to CSV columns.
    Returns list:
      (label, json_key, x_mean, y_mean, y_std)
    """
    csv_lookup = {_canon(c): c for c in success_stats.keys()}
    pairs = []

    for json_key, x in chamfer_mean.items():
        jk = str(json_key)
        core = jk[len("point_cloud_"):] if jk.startswith("point_cloud_") else jk
        c_core = _canon(core)

        if c_core in csv_lookup:
            ccol = csv_lookup[c_core]
            pairs.append((str(ccol), jk, float(x),
                          float(success_stats[ccol]["mean"]),
                          float(success_stats[ccol]["std"])))
            continue

        c_full = _canon(jk)
        if c_full in csv_lookup:
            ccol = csv_lookup[c_full]
            pairs.append((str(ccol), jk, float(x),
                          float(success_stats[ccol]["mean"]),
                          float(success_stats[ccol]["std"])))

    # Add synthetic Depth point: CD=0, success from CSV "Depth"
    if "Depth" in success_stats:
        pairs.append((
            "Depth",                 # label
            "rgbd_depth_baseline",   # pseudo key (not from JSON)
            0.0,                     # Chamfer distance
            float(success_stats["Depth"]["mean"]),
            float(success_stats["Depth"]["std"]),
        ))
    else:
        print("⚠️ CSV does not contain 'RGB-D' or 'Depth' column; Depth point not added.")

    if not pairs:
        raise RuntimeError("No matched methods between chamfer means and CSV columns.")
    return pairs


# ---------------------------
# Plot (one continuous x-axis + y error bars)
# ---------------------------
def plot_cd_vs_success_yerr(
    pairs,  # (label, json_key, x_mean, y_mean, y_std)
    out_png,
    title=None,
    fit_type='linear',  # 'linear' or 'curve' (a/x + b)
    x_label="Average Chamfer Distance (across tasks)",
    x_divisor=10.0,
    x_as_percent=False,
    show_correlations=False,
    annotate_points=True,
    annotation_fontsize=10,
    annotation_offsets=None,
    annotation_offset_scale=1.0,
):
    pairs = [(str(a), str(b), float(c), float(d), float(e)) for (a, b, c, d, e) in pairs]

    labels = [p[0] for p in pairs]
    if x_divisor <= 0:
        raise ValueError("x_divisor must be positive")
    xmean  = np.array([p[2] for p in pairs], dtype=float) / x_divisor
    ymean  = np.array([p[3] for p in pairs], dtype=float)
    ystd   = np.array([p[4] for p in pairs], dtype=float)

    # Map labels to display names for plotting
    disp_labels = [_disp(l) for l in labels]
    colors = build_color_map(disp_labels)

    fig, axis = plt.subplots(figsize=(7, 5))
    point_annotation_offsets = {
        "Pi3": (6, 8),
        "VGGT": (6, -14),
        "Mast3r": (6, -14),
    }
    if annotation_offsets:
        point_annotation_offsets.update(annotation_offsets)
    for xi, yi, yei, lab in zip(xmean, ymean, ystd, labels):
        dname = _disp(str(lab))
        axis.errorbar(
            xi, yi,
            yerr=yei if yei > 0 else None,
            fmt="o",
            markersize=7,
            color=colors[dname],
            ecolor=colors[dname],
            elinewidth=1.2,
            capsize=3,
        )
        if annotate_points:
            base_offset = point_annotation_offsets.get(lab, (6, 6))
            offset = tuple(
                value * annotation_offset_scale for value in base_offset
            )
            axis.annotate(
                dname,
                (xi, yi),
                xytext=offset,
                textcoords="offset points",
                fontsize=annotation_fontsize,
                ha="right" if offset[0] < 0 else "left",
                va="top" if offset[1] < 0 else "bottom",
            )

    xmin = float(np.min(xmean))
    xmax = float(np.max(xmean))
    xspan = max(xmax - xmin, 1e-6)
    x_padding = 0.05 * xspan
    xmin_plot = xmin - x_padding
    xmax_plot = xmax + x_padding
    axis.set_xlim(xmin_plot, xmax_plot)

    # Fit all estimators together, including MapAnything and the Depth point.
    if fit_type == 'curve' and len(xmean) >= 2:
        def _inv_func(x, a, b, c):
            return a / (x + c) + b
        try:
            popt, _ = curve_fit(_inv_func, xmean, ymean, p0=[-0.01, 0.0, 0.001], maxfev=10000)
            x_fit = np.linspace(max(xmin_plot, 0.0), xmax_plot, 200)
            axis.plot(
                x_fit,
                _inv_func(x_fit, *popt),
                '--',
                color='gray',
                alpha=0.5,
                linewidth=1,
                label='a/(x+c)+b',
            )
        except (RuntimeError, ValueError):
            pass
    elif fit_type == 'linear' and len(xmean) >= 2:
        coeffs = np.polyfit(xmean, ymean, 1)
        r = np.corrcoef(xmean, ymean)[0, 1]
        x_fit = np.linspace(xmin_plot, xmax_plot, 100)
        axis.plot(
            x_fit,
            np.polyval(coeffs, x_fit),
            '--',
            color='gray',
            alpha=0.5,
            linewidth=1,
            label=f'r = {r:.2f}',
        )

    if (
        show_correlations
        and len(xmean) >= 2
        and float(np.ptp(xmean)) > 0.0
        and float(np.ptp(ymean)) > 0.0
    ):
        pearson = float(np.corrcoef(xmean, ymean)[0, 1])
        spearman = float(spearmanr(xmean, ymean).statistic)
        if np.isfinite(pearson) and np.isfinite(spearman):
            axis.text(
                0.015,
                0.015,
                f"Pearson r={pearson:.2f}\nSpearman ρ={spearman:.2f}",
                transform=axis.transAxes,
                va="bottom",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#BBBBBB",
                    "alpha": 0.85,
                },
                zorder=10,
            )

    axis.set_xlabel(x_label)
    if x_as_percent:
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("Average Success Rate (across tasks)")
    axis.set_title(title or "Chamfer Distance vs Success rate (y = mean ± std)")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=colors[lab], markersize=8, label=lab)
        for lab in colors.keys()
    ]
    axis.legend(handles=handles, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"✅ Saved plot: {out_png}")


# ---------------------------
# CLI
# ---------------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rlb_json_dir", type=str, default="results/rlbench/chamfer_distance/")
    p.add_argument("--csv_path", type=str, default="results/dp3_sim.csv")
    p.add_argument("--out_png", type=str, default="results/rlbench/chamfer_distance/cd_vs_success.png")
    p.add_argument("--fit_type", type=str, default="linear", choices=["linear", "curve"],
                   help="Trendline type: 'linear' for y=ax+b, 'curve' for y=a/x+b")
    args = p.parse_args()

    chamfer_mean = load_chamfer_mean_from_rlb_jsons(args.rlb_json_dir)
    success_stats = load_success_mean_std_from_csv(args.csv_path)
    pairs = match_methods_yerr(chamfer_mean, success_stats)

    print("Matched methods (label <- json_key):")
    for lab, jk, xm, ym, ys in pairs:
        print(f"  {lab:<10} <- {jk:<24} | x={xm:.6f}  y={ym:.3f}±{ys:.3f}")

    plot_cd_vs_success_yerr(
        pairs,
        out_png=args.out_png,
        title="Average Chamfer Distance vs Success Rate (BC)",
        fit_type=args.fit_type,
    )

if __name__ == "__main__":
    main()
