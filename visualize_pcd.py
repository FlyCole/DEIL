#!/usr/bin/env python3
"""Visualize GT vs predicted point clouds from the PCD cache.

Usage examples:
  # Visualize rollout 0, step 5 for map_anything on phone_on_base
  python visualize_pcd.py --method map_anything --task phone_on_base --step 5

  # Visualize demo predictions vs demo GT, step 3
  python visualize_pcd.py --method map_anything --task phone_on_base --mode demo --step 3

  # Animate through all steps of a rollout
  python visualize_pcd.py --method map_anything --task phone_on_base --animate

  # Compare multiple methods side by side at a given step
  python visualize_pcd.py --method map_anything pi3 moge --task phone_on_base --step 5

  # Save figure instead of showing
  python visualize_pcd.py --method map_anything --task phone_on_base --step 5 --save out.png
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

CACHE_ROOT = "results/temporal_consistency/pcd_cache"


def find_task_dir(method_or_gt_dir: str, task: str) -> str | None:
    """Find the task config directory matching a task name prefix."""
    if not os.path.isdir(method_or_gt_dir):
        return None
    for name in sorted(os.listdir(method_or_gt_dir)):
        if name.startswith(task):
            return os.path.join(method_or_gt_dir, name, "iter_0")
    return None


def load_rollout_pcd(method: str, task: str, rollout: int, cache_root: str = CACHE_ROOT):
    """Load pred and gt arrays from a rollout npz."""
    method_dir = os.path.join(cache_root, method)
    task_dir = find_task_dir(method_dir, task)
    if task_dir is None:
        sys.exit(f"No task dir matching '{task}' in {method_dir}")
    path = os.path.join(task_dir, f"rollout_{rollout:03d}.npz")
    if not os.path.exists(path):
        sys.exit(f"File not found: {path}")
    data = np.load(path)
    n = int(data["num_steps"])
    preds = [data[f"pred_{i}"] for i in range(n)]
    gts = [data[f"gt_{i}"] for i in range(n)]
    return preds, gts, n


def load_demo_pcd(method: str, task: str, demo_idx: int = 0, cache_root: str = CACHE_ROOT):
    """Load demo pred and demo gt arrays."""
    method_dir = os.path.join(cache_root, method)
    gt_dir = os.path.join(cache_root, "_demo_gt")
    task_dir_method = find_task_dir(method_dir, task)
    task_dir_gt = find_task_dir(gt_dir, task)
    if task_dir_method is None:
        sys.exit(f"No task dir matching '{task}' in {method_dir}")
    if task_dir_gt is None:
        sys.exit(f"No task dir matching '{task}' in {gt_dir}")
    pred_path = os.path.join(task_dir_method, f"demo_pred_{demo_idx:03d}.npz")
    gt_path = os.path.join(task_dir_gt, f"demo_gt_{demo_idx:03d}.npz")
    if not os.path.exists(pred_path):
        sys.exit(f"File not found: {pred_path}")
    if not os.path.exists(gt_path):
        sys.exit(f"File not found: {gt_path}")
    pred_data = np.load(pred_path)
    gt_data = np.load(gt_path)
    n_pred = int(pred_data["num_steps"])
    n_gt = int(gt_data["num_steps"])
    n = min(n_pred, n_gt)
    preds = [pred_data[f"pred_{i}"] for i in range(n)]
    gts = [gt_data[f"gt_{i}"] for i in range(n)]
    return preds, gts, n


def scatter_pcd(ax, pcd, color, label, alpha=0.3, size=0.5):
    """Plot a point cloud on a 3D axis."""
    ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2],
               c=color, s=size, alpha=alpha, label=label)


def set_equal_axes(ax, pcds):
    """Set equal aspect ratio for 3D axes based on point clouds."""
    all_pts = np.concatenate(pcds, axis=0)
    mid = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.1
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def visualize_single(preds, gts, step, method, task, mode, save_path=None):
    """Visualize GT vs pred for a single step."""
    if step >= len(preds):
        sys.exit(f"Step {step} out of range (max {len(preds)-1})")
    fig = plt.figure(figsize=(14, 6))
    # GT only
    ax1 = fig.add_subplot(131, projection='3d')
    scatter_pcd(ax1, gts[step], 'blue', 'GT')
    set_equal_axes(ax1, [gts[step]])
    ax1.set_title('GT')
    # Pred only
    ax2 = fig.add_subplot(132, projection='3d')
    scatter_pcd(ax2, preds[step], 'red', 'Pred')
    set_equal_axes(ax2, [preds[step]])
    ax2.set_title(f'Pred ({method})')
    # Overlay
    ax3 = fig.add_subplot(133, projection='3d')
    scatter_pcd(ax3, gts[step], 'blue', 'GT', alpha=0.2)
    scatter_pcd(ax3, preds[step], 'red', 'Pred', alpha=0.2)
    set_equal_axes(ax3, [gts[step], preds[step]])
    ax3.set_title('Overlay')
    ax3.legend(markerscale=5)

    fig.suptitle(f"{method} | {task} | {mode} step {step}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def visualize_multi_method(methods, task, mode, rollout, step, demo_idx,
                           cache_root, save_path=None):
    """Compare multiple methods at the same step."""
    n_methods = len(methods)
    fig = plt.figure(figsize=(6 * n_methods, 6))
    for i, method in enumerate(methods):
        if mode == "rollout":
            preds, gts, n = load_rollout_pcd(method, task, rollout, cache_root)
        else:
            preds, gts, n = load_demo_pcd(method, task, demo_idx, cache_root)
        if step >= n:
            print(f"Warning: step {step} out of range for {method} (max {n-1}), using last step")
            s = n - 1
        else:
            s = step
        ax = fig.add_subplot(1, n_methods, i + 1, projection='3d')
        scatter_pcd(ax, gts[s], 'blue', 'GT', alpha=0.2)
        scatter_pcd(ax, preds[s], 'red', 'Pred', alpha=0.2)
        set_equal_axes(ax, [gts[s], preds[s]])
        ax.set_title(method)
        if i == 0:
            ax.legend(markerscale=5)

    fig.suptitle(f"{task} | {mode} step {step}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def animate(preds, gts, method, task, mode, save_dir=None):
    """Step through all timesteps interactively or save frames."""
    n = len(preds)
    for step in range(n):
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(131, projection='3d')
        scatter_pcd(ax1, gts[step], 'blue', 'GT')
        set_equal_axes(ax1, [gts[step]])
        ax1.set_title('GT')

        ax2 = fig.add_subplot(132, projection='3d')
        scatter_pcd(ax2, preds[step], 'red', 'Pred')
        set_equal_axes(ax2, [preds[step]])
        ax2.set_title(f'Pred ({method})')

        ax3 = fig.add_subplot(133, projection='3d')
        scatter_pcd(ax3, gts[step], 'blue', 'GT', alpha=0.2)
        scatter_pcd(ax3, preds[step], 'red', 'Pred', alpha=0.2)
        set_equal_axes(ax3, [gts[step], preds[step]])
        ax3.set_title('Overlay')
        ax3.legend(markerscale=5)

        fig.suptitle(f"{method} | {task} | {mode} step {step}/{n-1}")
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"step_{step:03d}.png")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Saved frame {step}/{n-1}")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize GT vs predicted PCDs")
    parser.add_argument("--method", nargs="+", required=True,
                        help="Method name(s), e.g. map_anything pi3")
    parser.add_argument("--task", required=True,
                        help="Task name prefix, e.g. phone_on_base")
    parser.add_argument("--mode", default="rollout", choices=["rollout", "demo"],
                        help="Visualize rollout or demo PCDs")
    parser.add_argument("--rollout", type=int, default=0,
                        help="Rollout index (for mode=rollout)")
    parser.add_argument("--demo", type=int, default=0,
                        help="Demo index (for mode=demo)")
    parser.add_argument("--step", type=int, default=0,
                        help="Timestep to visualize")
    parser.add_argument("--animate", action="store_true",
                        help="Iterate through all steps")
    parser.add_argument("--save", default=None,
                        help="Save figure to path (single step) or directory (animate)")
    parser.add_argument("--cache_root", default=CACHE_ROOT,
                        help="PCD cache root directory")
    args = parser.parse_args()

    # Multi-method comparison
    if len(args.method) > 1:
        if args.animate:
            sys.exit("--animate not supported with multiple methods")
        visualize_multi_method(args.method, args.task, args.mode,
                               args.rollout, args.step, args.demo,
                               args.cache_root, args.save)
        return

    method = args.method[0]
    if args.mode == "rollout":
        preds, gts, n = load_rollout_pcd(method, args.task, args.rollout, args.cache_root)
    else:
        preds, gts, n = load_demo_pcd(method, args.task, args.demo, args.cache_root)

    print(f"Loaded {n} steps for {method}/{args.task} ({args.mode})")

    if args.animate:
        animate(preds, gts, method, args.task, args.mode, save_dir=args.save)
    else:
        visualize_single(preds, gts, args.step, method, args.task, args.mode, args.save)


if __name__ == "__main__":
    main()
