import os
import re
import json
from glob import glob
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_rlb_files(input_dir: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Returns list of (exp_name, filename, json_data) for files matching rlb_*.json
    """
    paths = sorted(glob(os.path.join(input_dir, "rlb_*.json")))
    if not paths:
        raise RuntimeError(f"No files found: {os.path.join(input_dir, 'rlb_*.json')}")

    out = []
    for p in paths:
        fname = os.path.basename(p)
        m = re.match(r"rlb_(.+)\.json$", fname)
        if not m:
            continue
        exp_name = m.group(1)
        with open(p, "r") as f:
            data = json.load(f)
        out.append((exp_name, fname, data))
    if not out:
        raise RuntimeError("No valid rlb_*.json matched the expected naming pattern.")
    return out


def build_matrix(exps: List[Tuple[str, str, Dict[str, Any]]]):
    """
    Builds:
      - methods: sorted list of all method keys seen in any experiment
      - exp_names: sorted list of experiment names
      - mat_mean: shape (E, M) with NaN for missing
      - mat_median, mat_p95, mat_std similarly
      - meta per experiment (zarr_path, normalization, etc.) (optional)
    """
    exp_names = [e[0] for e in exps]

    # union of methods
    method_set = set()
    for _, _, data in exps:
        for k in (data.get("metrics") or {}).keys():
            if k == "point_cloud_map_anything":
                continue
            method_set.add(k)
    methods = sorted(method_set)

    E, M = len(exps), len(methods)

    def make_mat():
        return np.full((E, M), np.nan, dtype=np.float32)

    mat_mean = make_mat()
    mat_median = make_mat()
    mat_p95 = make_mat()
    mat_std = make_mat()

    exp_meta = []

    for ei, (exp_name, fname, data) in enumerate(exps):
        metrics = data.get("metrics") or {}
        for mi, method in enumerate(methods):
            v = metrics.get(method)
            if not v:
                continue
            mat_mean[ei, mi] = np.float32(v.get("mean")) if v.get("mean") is not None else np.nan
            mat_median[ei, mi] = np.float32(v.get("median")) if v.get("median") is not None else np.nan
            mat_p95[ei, mi] = np.float32(v.get("p95")) if v.get("p95") is not None else np.nan
            mat_std[ei, mi] = np.float32(v.get("std")) if v.get("std") is not None else np.nan

        exp_meta.append({
            "name": exp_name,
            "source_file": fname,
            "zarr_path": data.get("zarr_path"),
            "baseline": data.get("baseline"),
            "normalization": data.get("normalization"),
        })

    return methods, exp_names, mat_mean, mat_median, mat_p95, mat_std, exp_meta


def nan_stats(vec: np.ndarray) -> Dict[str, float]:
    """
    vec: (E,) with NaNs
    """
    valid = vec[~np.isnan(vec)]
    if valid.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(valid.size),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "median": float(np.median(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }


def aggregate_across_experiments(methods, mat_mean, mat_median, mat_p95, mat_std):
    """
    Returns dict keyed by method with stats across experiments.
    """
    agg = {}
    for mi, method in enumerate(methods):
        agg[method] = {
            "mean_of_means": nan_stats(mat_mean[:, mi]),
            "mean_of_medians": nan_stats(mat_median[:, mi]),
            "mean_of_p95": nan_stats(mat_p95[:, mi]),
            "mean_of_stds": nan_stats(mat_std[:, mi]),
        }

    print(agg)
    return agg


def save_aggregate_json(output_path: str, exp_meta, methods, exp_names,
                        mat_mean, mat_median, mat_p95, mat_std, agg):
    """
    Saves everything needed for later analysis.
    """
    # per-experiment summary table
    per_exp = []
    for ei, exp_name in enumerate(exp_names):
        row = {"name": exp_name}
        row["methods"] = {}
        for mi, method in enumerate(methods):
            row["methods"][method] = {
                "mean": None if np.isnan(mat_mean[ei, mi]) else float(mat_mean[ei, mi]),
                "median": None if np.isnan(mat_median[ei, mi]) else float(mat_median[ei, mi]),
                "p95": None if np.isnan(mat_p95[ei, mi]) else float(mat_p95[ei, mi]),
                "std": None if np.isnan(mat_std[ei, mi]) else float(mat_std[ei, mi]),
            }
        per_exp.append(row)

    payload = {
        "experiments_meta": exp_meta,
        "methods": methods,
        "per_experiment": per_exp,
        "aggregate_across_experiments": agg,
        "notes": {
            "nan_policy": "Methods missing in an experiment are ignored (NaN) in aggregates."
        }
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✅ Wrote aggregate JSON: {output_path}")


# -------------------------
# Plotting (matplotlib only)
# -------------------------
def plot_bar_avg_mean(methods, mat_mean, out_png):
    """
    Bar plot: average mean Chamfer per method, with error bars = std across experiments.
    """
    means = []
    stds = []
    labels = []
    for mi, method in enumerate(methods):
        v = mat_mean[:, mi]
        valid = v[~np.isnan(v)]
        if valid.size == 0:
            continue
        labels.append(method)
        means.append(np.mean(valid))
        stds.append(np.std(valid))

    x = np.arange(len(labels))

    plt.figure()
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Chamfer (avg of per-experiment means)")
    plt.title("Average Chamfer per method (error = std across experiments)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"🖼️ Saved plot: {out_png}")


def plot_box_mean(methods, mat_mean, out_png):
    """
    Box plot: distribution of per-experiment mean Chamfer per method.
    """
    data = []
    labels = []
    for mi, method in enumerate(methods):
        v = mat_mean[:, mi]
        valid = v[~np.isnan(v)]
        if valid.size == 0:
            continue
        data.append(valid)
        labels.append(method)

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Chamfer (per-experiment mean)")
    plt.title("Distribution of mean Chamfer across experiments")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"🖼️ Saved plot: {out_png}")


def plot_experiment_lines(methods, exp_names, mat_mean, out_png):
    """
    Line plot: each experiment as a line over methods (mean Chamfer).
    Helpful for spotting which experiments behave differently.
    """
    # keep only methods that have at least one value
    keep = []
    for mi in range(len(methods)):
        if np.any(~np.isnan(mat_mean[:, mi])):
            keep.append(mi)

    labels = [methods[i] for i in keep]
    x = np.arange(len(labels))

    plt.figure()
    for ei, exp_name in enumerate(exp_names):
        y = mat_mean[ei, keep]
        # Replace NaN with masked values so lines break
        y_masked = np.ma.masked_invalid(y)
        plt.plot(x, y_masked, marker="o", linewidth=1, label=exp_name)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Chamfer (mean)")
    plt.title("Per-experiment mean Chamfer across methods")
    # If too many experiments, legend can be huge; still optional
    if len(exp_names) <= 12:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"🖼️ Saved plot: {out_png}")


def main(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    exps = load_rlb_files(input_dir)
    methods, exp_names, mat_mean, mat_median, mat_p95, mat_std, exp_meta = build_matrix(exps)

    agg = aggregate_across_experiments(methods, mat_mean, mat_median, mat_p95, mat_std)

    out_json = os.path.join(output_dir, "rlb_aggregate.json")
    # save_aggregate_json(out_json, exp_meta, methods, exp_names,
                        # mat_mean, mat_median, mat_p95, mat_std, agg)

    plot_bar_avg_mean(methods, mat_mean, os.path.join(output_dir, "avg_mean_chamfer.png"))
    plot_box_mean(methods, mat_mean, os.path.join(output_dir, "box_mean_chamfer.png"))
    plot_experiment_lines(methods, exp_names, mat_mean, os.path.join(output_dir, "experiments_lines.png"))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="./results/rlbench/chamfer_distance/", help="Folder containing rlb_*.json")
    p.add_argument("--output_dir", default="./results/rlbench/chamfer_distance/", help="Where to write aggregate JSON + plots")
    args = p.parse_args()
    main(args.input_dir, args.output_dir)
