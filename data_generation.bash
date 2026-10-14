#!/usr/bin/env bash
set -e  # Stop the script if any command fails

# Ask user for the dataset name
read -p "Enter dataset name (name): " NAME

# List of estimators to run
ESTIMATORS=("vggt" "moge" "mast3r" "map_anything")
# ESTIMATORS=("vggt" "moge" "map_anything")
# GT-depth noise sweep (standard deviations are encoded in the Zarr keys):
# ESTIMATORS=("gt_noisy_1mm" "gt_noisy_5mm" "gt_noisy_10mm" "gt_noisy_20mm")
# D435-like disparity corruption presets:
# ESTIMATORS=("gt_d435_mild" "gt_d435_strong")

for EST in "${ESTIMATORS[@]}"; do
  echo "Running with estimator: $EST"
  python rlb_demo_process_dp3.py \
    --zarr_path "3D-Diffusion-Policy/3D-Diffusion-Policy/data/${NAME}.zarr" \
    --estimators "$EST" \
    --resolution 256 \
    --num_points 1024
    # --resume
done
