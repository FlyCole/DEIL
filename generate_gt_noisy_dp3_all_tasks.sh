#!/usr/bin/env bash

set -euo pipefail

# Edit this list to add, remove, or reorder datasets.
TASKS=(
    rlb_close_microwave
    rlb_toilet_seat_down
    rlb_toilet_seat_up
)

# These defaults can be overridden without editing the script, for example:
#   ESTIMATOR=gt_noisy_30mm GT_NOISE_SEED=1 bash generate_gt_noisy_dp3_all_tasks.sh
ESTIMATOR=${ESTIMATOR:-gt_noisy_10mm}
GT_NOISE_SEED=${GT_NOISE_SEED:-0}
RESOLUTION=${RESOLUTION:-256}
NUM_POINTS=${NUM_POINTS:-1024}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
data_dir="${script_dir}/3D-Diffusion-Policy/3D-Diffusion-Policy/data"

for task in "${TASKS[@]}"; do
    zarr_path="${data_dir}/${task}.zarr"
    if [[ ! -d ${zarr_path} ]]; then
        echo "Dataset not found: ${zarr_path}" >&2
        exit 1
    fi

    echo "[$task] Generating ${ESTIMATOR} point clouds"
    python "${script_dir}/rlb_demo_process_dp3.py" \
        --zarr_path "${zarr_path}" \
        --estimators "${ESTIMATOR}" \
        --resolution "${RESOLUTION}" \
        --num_points "${NUM_POINTS}" \
        --gt_noise_seed "${GT_NOISE_SEED}" \
        "$@"
done

echo "Completed ${#TASKS[@]} tasks."
