"""Analyze main_results_fixed_demo CSVs.

For each method, groups by (camera_names, resolution, mask_input) and
computes the mean and std of avg_sr across all tasks.

Usage:
    python analyze_main_results.py
    python analyze_main_results.py --dir results/rlbench/main_results_fixed_demo
    python analyze_main_results.py --file results/rlbench/main_results_fixed_demo/pi3.csv
"""

import argparse
import glob
import os
import pandas as pd


def summarize(df):
    group_cols = ['camera_names', 'resolution', 'mask_input']
    summary = df.groupby(group_cols).agg(
        num_tasks=('task_name', 'nunique'),
        mean_sr=('avg_sr', 'mean'),
        std_sr=('avg_sr', 'std'),
    ).reset_index()
    return summary


def print_table(method, summary):
    header = (f"{'Cameras':<30} {'Res':>4} {'Mask':>5} "
              f"{'#T':>3} {'Mean SR':>8} {'Std SR':>8}")
    print(f"\n{'=' * len(header)}")
    print(f"  {method}")
    print(f"{'=' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")
    for _, row in summary.iterrows():
        print(f"{row['camera_names']:<30} {row['resolution']:>4} "
              f"{str(row['mask_input']):>5} {row['num_tasks']:>3} "
              f"{row['mean_sr']:>8.4f} {row['std_sr']:>8.4f}")
    print(f"{'=' * len(header)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        default='results/rlbench/main_results_fixed_demo')
    parser.add_argument('--file', type=str, default=None,
                        help='Analyze a single CSV file instead of all in --dir.')
    args = parser.parse_args()

    if args.file:
        csv_files = [args.file]
    else:
        csv_files = sorted(glob.glob(os.path.join(args.dir, '*.csv')))

    for csv_path in csv_files:
        method = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)
        summary = summarize(df)
        print_table(method, summary)
