# DEIL

**Published in IEEE Robotics and Automation Letters (RA-L), 2026.**

[![Paper](https://img.shields.io/badge/Paper-RA--L%202026-00629B?style=for-the-badge&logo=googlescholar&logoColor=white)](https://doi.org/10.1109/LRA.2026.3734880)
[![Project Page](https://img.shields.io/badge/Project_Page-Website-4285F4?style=for-the-badge&logo=googlechrome&logoColor=white)](https://flycole.github.io/deil.github.io/)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](./)

Code for *Depth Camera or RGB-to-Depth? A Study of RGB-to-Depth Estimation for
Robot Policy Learning*. The simulation experiments compare public Instant
Policy and 3D Diffusion Policy using GT, Pi3, VGGT, MoGe, MASt3R and MapAnything.

Run all commands from the repository root unless stated otherwise.

## Environment

Use Linux, Python 3.10, an NVIDIA GPU, Git and a C/C++ compiler. This setup uses
PyTorch 2.7 with CUDA 12.8, including support for RTX 50-series GPUs.

```bash
export DEIL_ROOT="$PWD"
conda create -n deil python=3.10.15 pip -y
conda activate deil
conda install -c nvidia cuda-toolkit=12.8 -y
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-analysis.txt
python -m pip install numpy==1.26.4 \
  diffusers==0.31.0 hydra-core==1.3.2 omegaconf==2.3.0 \
  einops==0.8.1 dill==0.3.5.1 moviepy==1.0.3 lightning==2.6.5 \
  wandb imageio av termcolor open3d==0.19.0 gymnasium==1.1.0 \
  gdown timm safetensors pillow-heif plyfile ninja fvcore iopath \
  torch-geometric==2.7.0
python -m pip install --no-deps --only-binary=:all: \
  torch-cluster==1.6.3 pyg-lib==0.5.0 \
  -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

Keep NumPy at 1.26.4 for compatibility with Zarr 2.12. The NVIDIA driver must
support CUDA 12.8. Historical environment YAML files are machine snapshots;
use the commands above for installation.

### Simulator and policy libraries

Install CoppeliaSim 4.1 into a new directory:

```bash
export COPPELIASIM_ROOT="$HOME/CoppeliaSim-4.1"
wget -O /tmp/deil-coppeliasim.tar.xz \
  https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p "$COPPELIASIM_ROOT"
tar -xf /tmp/deil-coppeliasim.tar.xz -C "$COPPELIASIM_ROOT" --strip-components=1
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"

python scripts/bootstrap_runtime_dependencies.py --install
python 3D-Diffusion-Policy/release/bootstrap.py
python -m pip install -e ./3D-Diffusion-Policy/3D-Diffusion-Policy --no-deps
python -m pip install -e ./3D-Diffusion-Policy/third_party/pytorch3d_simplified \
  --no-build-isolation --no-deps
```

Keep the simulator and CUDA environment variables in your shell or Conda
activation script. `--headless` still requires a graphics context; on a server
without a display, use the upstream headless setup or `xvfb-run -a`.

The bootstrap scripts fetch fixed upstream versions and apply the supplied
patches. SDK setup may update CoppeliaSim's `system/usrset.txt`. Version and
patch records are in `runtime_dependencies.json`, `depth_dependencies.json`
and `3D-Diffusion-Policy/release/source_manifest.json`.

### Depth models and public Instant Policy weights

Install only the depth models you need. For example:

```bash
python scripts/bootstrap_depth_dependencies.py --only pi3 --install
```

`--only` accepts `pi3`, `vggt`, `moge`, `mast3r` and `map_anything`; omit it to
install all five. Model weights are downloaded on first use. GT needs no
learned depth model. For MASt3R, also build CroCo after installing its source:

```bash
python scripts/bootstrap_depth_dependencies.py --only mast3r --build-curope
```

Download the official public Instant Policy checkpoint:

```bash
cd "$DEIL_ROOT/instant_policy"
bash download_weights.sh
cd "$DEIL_ROOT"
```

Its default path is `instant_policy/checkpoints/model.pt`. Use
`--checkpoint_path` to select another copy of the public weights. The public
deployment binary is included; no separate Instant Policy package install is
needed.

## Instant Policy evaluation

Run one paper condition:

```bash
python rlb_task_eval.py --protocol paper --task_name phone_on_base \
  --depth_est_name pi3 --camera_setups front_side --resolution 256 \
  --no_mask_input --headless
```

Each paper condition uses 20 single-demonstration contexts with 10 rollouts
each. Select a different task or estimator with the corresponding arguments.
For `gt`, omit `--no_mask_input`. `map_anything_no_ext` uses MapAnything without
known camera poses. To run the complete paper grid:

```bash
python rlb_task_eval.py --config configs/ip_eval/paper.json --headless
```

Demonstrations are collected automatically and cached in `demos_ip/`. Results
are saved under `results/ip_eval/<protocol>/<execution_id>/`; compatible
completed rollouts are reused. For a small run, set `--protocol custom`,
`--num_demo_iter 1` and `--num_rollouts 1`. `rlb_demo_collection_ip.py` accepts the same
protocol selectors if demonstrations need to be collected separately.

## DP3 collection, training and evaluation

Collect demonstrations:

```bash
python rlb_demo_collection_dp3.py --headless
```

Set `task_names` and `num_demos` in the script's main block. The defaults are
`close_box`, `plate_out` and `basketball`, with 100 demonstrations per task,
front/left-shoulder cameras and resolution 256. Datasets are saved to
`3D-Diffusion-Policy/3D-Diffusion-Policy/data/rlb_<task>.zarr`.

Generate point clouds for a learned estimator:

```bash
python rlb_demo_process_dp3.py \
  --zarr_path 3D-Diffusion-Policy/3D-Diffusion-Policy/data/rlb_close_box.zarr \
  --estimators pi3 --resolution 256 --num_points 1024
```

GT point clouds are already in the collected dataset; do not preprocess `gt`.
For noisy GT, select `gt_noisy_10mm`, `gt_noisy_20mm`, `gt_noisy_30mm` or
`gt_noisy_40mm`. Use `--resume` for compatible interrupted processing or
`--rebuild` for old outputs without transaction metadata.

Train and evaluate using the same task, run tag and estimator. The fourth
argument selects the actual depth estimator:

```bash
wandb login
bash 3D-Diffusion-Policy/scripts/train_policy.sh dp3 rlb_close_box release pi3
EVAL_PROTOCOL=paper \
  bash 3D-Diffusion-Policy/scripts/eval_policy.sh dp3 rlb_close_box release pi3
```

Training uses W&B online logging. `EVAL_PROTOCOL=paper` evaluates 50 rollouts.
Checkpoints are stored in `data/outputs/`; evaluation outputs are stored in
`data/eval_outputs/`. `TRAIN_SEED` and `EVAL_SEED` must select the same training
run. Scene evaluation seeds are fixed separately across depth methods.

To preprocess, train and evaluate multiple estimators on an existing dataset:

```bash
ESTIMATORS="gt pi3" RUN_TAG=release bash run_all_eval.bash rlb_close_box
```

`RUN_MODE` selects `all`, `preprocess`, `train` or `evaluate`. For legacy derived
arrays, set `PREPROCESS_RESUME=0` once to rebuild them.

## Result analysis

Generate plots from the included DP3 statistics:

```bash
python -m data_analysis.dp3.plot_dp3_rollout_analysis \
  --input_dir data_analysis/dp3/stats --output_dir results/dp3_figures
```

Process your own rollout artifacts:

```bash
python -m data_analysis.dp3.process_dp3_rollout_artifacts \
  --input_dir /path/to/rollouts --stats_dir /path/to/stats --skip_videos
python -m data_analysis.ip.process_ip_rollout_artifacts \
  --input_dir /path/to/ip/execution_id --output_xlsx /path/to/ip.xlsx \
  --expected_rollouts 200 --skip_videos
python -m data_analysis.ip.plot_ip_rollout_analysis \
  --workbook /path/to/ip.xlsx --output_dir /path/to/figures
```

For IP, use the specific execution directory printed by the evaluator and set
`--expected_rollouts` to the actual count. `run_tc_eval.py` provides optional
temporal-consistency experiments; use `--help` for its arguments.

## Citation

If you use this code in your research, please cite:

```bibtex
@ARTICLE{11694202,
  author={Ren, Yifei and Vitiello, Pietro and Johns, Edward},
  journal={IEEE Robotics and Automation Letters},
  title={Depth Camera or RGB-to-Depth? A Study of RGB-to-Depth Estimation for Robot Policy Learning},
  year={2026},
  volume={11},
  number={11},
  pages={12887-12894},
  keywords={Cameras;Depth measurement;Clouds;Imitation learning;Learning (artificial intelligence);Modeling;Robots;Context;Measurement;Cloning;Performance evaluation and benchmarking;perception for grasping and manipulation;imitation learning},
  doi={10.1109/LRA.2026.3734880}
}
```

## Acknowledgements

Built on [Instant Policy](https://github.com/vv19/instant_policy),
[DP3](https://github.com/YanjieZe/3D-Diffusion-Policy),
[PyRep](https://github.com/stepjam/PyRep), [RLBench](https://github.com/stepjam/RLBench),
[Pi3](https://github.com/yyfz/Pi3), [VGGT](https://github.com/facebookresearch/vggt),
[MoGe](https://github.com/microsoft/MoGe), [MASt3R](https://github.com/naver/mast3r)
and [MapAnything](https://github.com/facebookresearch/map-anything).

Third-party code and weights retain their original licenses and notices;
some are restricted to noncommercial research. Keep the bundled license files.
Public Instant Policy uses PolyForm Noncommercial 1.0.0. MapAnything's source
is Apache 2.0, while its selected weights use CC BY-NC 4.0.
