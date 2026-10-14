#!/usr/bin/env bash

set -euo pipefail

# With no positional arguments, these tasks are processed and trained in order.
# Alternatively, pass one or more task names on the command line.
DEFAULT_TASKS=(
    rlb_close_microwave
    rlb_toilet_seat_down
    rlb_toilet_seat_up
)

if (( $# > 0 )); then
    TASKS=("$@")
else
    TASKS=("${DEFAULT_TASKS[@]}")
fi

ESTIMATOR=${ESTIMATOR:-gt_noisy_10mm}
RUN_TAG=${RUN_TAG:-$(date +%m%d)}
TRAIN_GPU_ID=${TRAIN_GPU_ID:-0}
PROCESS_GPU_ID=${PROCESS_GPU_ID:-${TRAIN_GPU_ID}}
TRAIN_SEED=${TRAIN_SEED:-0}

RESOLUTION=${RESOLUTION:-256}
NUM_POINTS=${NUM_POINTS:-1024}
GT_NOISE_STD_M=${GT_NOISE_STD_M:-0.010}
GT_NOISE_SEED=${GT_NOISE_SEED:-0}
D435_BASELINE_M=${D435_BASELINE_M:-0.05}
D435_DISPARITY_NOISE_STD_PX=${D435_DISPARITY_NOISE_STD_PX:-0.05}
D435_DISPARITY_BIAS_PX=${D435_DISPARITY_BIAS_PX:-0.02}
D435_DISPARITY_QUANTIZATION_PX=${D435_DISPARITY_QUANTIZATION_PX:-0.03125}
D435_NOISE_SEED=${D435_NOISE_SEED:-0}
PROCESS_RESUME=${PROCESS_RESUME:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
policy_root="${script_dir}/3D-Diffusion-Policy/3D-Diffusion-Policy"
data_dir="${policy_root}/data"
task_config_dir="${policy_root}/diffusion_policy_3d/config/task"
train_script="${script_dir}/3D-Diffusion-Policy/scripts/train_policy.sh"

process_options=()
case ${PROCESS_RESUME,,} in
    1|true|yes)
        process_options+=(--resume)
        ;;
    0|false|no)
        ;;
    *)
        echo "PROCESS_RESUME must be 0/1, true/false, or yes/no; got: ${PROCESS_RESUME}" >&2
        exit 2
        ;;
esac

# Validate all inputs before modifying any dataset or starting a training run.
for task in "${TASKS[@]}"; do
    if [[ ! -d ${data_dir}/${task}.zarr ]]; then
        echo "Dataset not found: ${data_dir}/${task}.zarr" >&2
        exit 1
    fi
    if [[ ! -f ${task_config_dir}/${task}.yaml ]]; then
        echo "Training config not found: ${task_config_dir}/${task}.yaml" >&2
        exit 1
    fi
done

echo "Estimator: ${ESTIMATOR}"
echo "Tasks: ${TASKS[*]}"
echo "Processing GPU: ${PROCESS_GPU_ID}; training GPU: ${TRAIN_GPU_ID}"
echo "Run tag: ${RUN_TAG}; training seed: ${TRAIN_SEED}"

if [[ ${ESTIMATOR,,} == gt ]]; then
    echo "[processing] ESTIMATOR=gt uses the existing /data/point_cloud; skipping reconstruction."
else
    echo "[processing] Starting data processing for ${#TASKS[@]} task(s)."
    for task in "${TASKS[@]}"; do
        echo "[processing:${task}] Generating ${ESTIMATOR} point clouds"
        CUDA_VISIBLE_DEVICES="${PROCESS_GPU_ID}" "${PYTHON_BIN}" \
            "${script_dir}/rlb_demo_process_dp3.py" \
            --zarr_path "${data_dir}/${task}.zarr" \
            --estimators "${ESTIMATOR}" \
            --resolution "${RESOLUTION}" \
            --num_points "${NUM_POINTS}" \
            --gt_noise_std_m "${GT_NOISE_STD_M}" \
            --gt_noise_seed "${GT_NOISE_SEED}" \
            --d435_baseline_m "${D435_BASELINE_M}" \
            --d435_disparity_noise_std_px "${D435_DISPARITY_NOISE_STD_PX}" \
            --d435_disparity_bias_px "${D435_DISPARITY_BIAS_PX}" \
            --d435_disparity_quantization_px "${D435_DISPARITY_QUANTIZATION_PX}" \
            --d435_noise_seed "${D435_NOISE_SEED}" \
            "${process_options[@]}"
    done
    echo "[processing] All datasets completed successfully."
fi

echo "[training] Starting sequential training for ${#TASKS[@]} task(s)."
for task in "${TASKS[@]}"; do
    echo "[training:${task}] Starting ${ESTIMATOR} training"
    TRAIN_GPU_ID="${TRAIN_GPU_ID}" TRAIN_SEED="${TRAIN_SEED}" \
        bash "${train_script}" dp3 "${task}" "${RUN_TAG}" "${ESTIMATOR}"
done

echo "Completed processing and training for ${#TASKS[@]} task(s)."
