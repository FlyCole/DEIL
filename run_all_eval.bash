#!/usr/bin/env bash
set -euo pipefail

# Process existing datasets, train each depth variant, and evaluate its best
# checkpoint. Policy scripts receive the estimator as argument four.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python}
run_mode=${RUN_MODE:-all}
run_tag=${RUN_TAG:-release}
resume_preprocessing=${PREPROCESS_RESUME:-1}
dry_run=${DRY_RUN:-0}
read -r -a estimators <<< "${ESTIMATORS:-gt pi3 vggt moge mast3r map_anything}"
tasks=()

usage() {
    cat <<'EOF'
Usage: bash run_all_eval.bash [--dry-run] [rlb_task ...]

Default task: rlb_toilet_seat_up, or TASK_NAME.
ESTIMATORS: space-separated sources (default: gt pi3 vggt moge mast3r map_anything).
RUN_MODE: all (preprocess/train/evaluate), preprocess, train, or evaluate.
RUN_TAG: training/evaluation run tag (default: release).
PREPROCESS_RESUME: 1 resumes a compatible transactional output; 0 safely rebuilds.
DRY_RUN: 1 prints commands without changing datasets or starting models.
PYTHON_BIN: Python executable for preprocessing (default: python).
TRAIN_SEED/TRAIN_GPU_ID and EVAL_SEED/EVAL_GPU_ID select runs/devices.
EVAL_PROTOCOL defaults to paper (50 rollouts); EVAL_EPISODES can override it.
EOF
}

for argument in "$@"; do
    case "$argument" in
        --dry-run|--dry_run) dry_run=1 ;;
        --help|-h) usage; exit 0 ;;
        rlb_*) tasks+=("$argument") ;;
        *) echo "Unknown argument: $argument" >&2; usage >&2; exit 2 ;;
    esac
done
if (( ${#tasks[@]} == 0 )); then
    tasks=("${TASK_NAME:-rlb_toilet_seat_up}")
fi
case "$run_mode" in
    all|preprocess|train|evaluate) ;;
    *) echo "RUN_MODE must be all, preprocess, train, or evaluate" >&2; exit 2 ;;
esac
if [[ ! $resume_preprocessing =~ ^[01]$ || ! $dry_run =~ ^[01]$ ]]; then
    echo "PREPROCESS_RESUME and DRY_RUN must be 0 or 1" >&2
    exit 2
fi
if [[ -z $run_tag || ${#estimators[@]} == 0 ]]; then
    echo "RUN_TAG and ESTIMATORS must be nonempty" >&2
    exit 2
fi
for task in "${tasks[@]}"; do
    if [[ ! $task =~ ^rlb_[a-z_]+$ ]]; then
        echo "Invalid RLBench task config: $task" >&2
        exit 2
    fi
done
for estimator in "${estimators[@]}"; do
    if [[ ! $estimator =~ ^(gt|pi3|vggt|moge|mast3r|map_anything|gt_noisy(_[0-9]+mm)?|gt_d435(_(mild|strong))?)$ ]]; then
        echo "Unsupported depth source: $estimator" >&2
        exit 2
    fi
done

run() {
    if [[ $dry_run == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

cd "$script_dir"
for task in "${tasks[@]}"; do
    dataset="3D-Diffusion-Policy/3D-Diffusion-Policy/data/${task}.zarr"
    if [[ $dry_run == 0 && ($run_mode == all || $run_mode == preprocess) && ! -d $dataset ]]; then
        echo "Missing dataset: $dataset. Collect demonstrations first." >&2
        exit 2
    fi
    for estimator in "${estimators[@]}"; do
        if [[ ($run_mode == all || $run_mode == preprocess) && $estimator != gt ]]; then
            process_arguments=(
                "$python_bin" rlb_demo_process_dp3.py --zarr_path "$dataset"
                --estimators "$estimator" --resolution 256 --num_points 1024
            )
            if [[ $resume_preprocessing == 1 ]]; then
                process_arguments+=(--resume)
            fi
            run "${process_arguments[@]}"
        fi
        if [[ $run_mode == all || $run_mode == train ]]; then
            run bash 3D-Diffusion-Policy/scripts/train_policy.sh dp3 "$task" "$run_tag" "$estimator"
        fi
        if [[ $run_mode == all || $run_mode == evaluate ]]; then
            run env "EVAL_SEED=${EVAL_SEED:-${TRAIN_SEED:-0}}" \
                "EVAL_GPU_ID=${EVAL_GPU_ID:-${TRAIN_GPU_ID:-0}}" \
                "EVAL_PROTOCOL=${EVAL_PROTOCOL:-paper}" \
                bash 3D-Diffusion-Policy/scripts/eval_policy.sh dp3 "$task" "$run_tag" "$estimator"
        fi
    done
done
