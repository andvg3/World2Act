# World2Act: Latent Post-Training from World Model Dynamics

World2Act learns action representations from world-model dynamics. This repository contains the AMD/ROCm implementation built on NVIDIA Cosmos Predict2.

## Released

- [x] WM pretraining / fine-tuning code
- [x] Stage 1: Action VAE training with a frozen world model
- [ ] Post-train VLAs — coming in a future release

## Data availability

The data was collected by the World2Act authors. The simulation data used for world-model fine-tuning will be published later. The post-training dataset will also be released later. This release includes training and inference code; datasets and private model checkpoints are not bundled.

## Get started

1. Prepare the [AMD/ROCm environment](documentations/setup_rocm.md).
2. Prepare local videos, text embeddings, and the appropriate checkpoints using the [training guide](documentations/world2act_training.md).
3. Choose world-model fine-tuning or Stage 1 Action VAE training.
4. Run [RoboCasa, LIBERO, or real-world Franka inference](documentations/world2act_inference.md).

## Two training modes

| Mode | What is trained? | Inputs |
| --- | --- | --- |
| World-model fine-tuning (`wm`, default) | World-model denoising network | Simulation videos and text embeddings |
| Stage 1 (`action_vae`) | Action VAE and video bridger, with the world model frozen | Videos, text embeddings, and matching robot actions |

World-model fine-tuning uses the original experiment name:

```bash
export WORLD2ACT_DATASET_DIR="/path/to/local/simulation_dataset"
export WM_CHECKPOINT="/path/to/local/world_model.pt"
EXP=predict2_video2world_training_2b_groot_gr1_480

torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py \
  --training_mode wm \
  -- experiment="${EXP}" \
  model.config.model_manager_config.dit_path="$WM_CHECKPOINT"
```

For Stage 1, select `--training_mode action_vae` and provide the action dataset:

```bash
export WORLD2ACT_DATASET_DIR="/path/to/local/paired_video_dataset"
export ACTION_DATA_ROOT="/path/to/local/action_dataset"
export WM_CHECKPOINT="/path/to/local/fine_tuned_world_model.pt"
EXP=predict2_video2world_training_2b_groot_gr1_480

torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py \
  --training_mode action_vae \
  --action_dataset simpler_env \
  --action_data_root "$ACTION_DATA_ROOT" \
  --action_output_dir checkpoints/action_vae \
  -- experiment="${EXP}" \
  model.config.model_manager_config.dit_path="$WM_CHECKPOINT" \
  job.group=stage1_action_vae
```

Stage 1 currently uses one GPU per run. See the [full training instructions](documentations/world2act_training.md) for data formats, other action loaders, checkpoint formats, and configuration overrides. VLA post-training is not part of this release.

## Inference

The [inference guide](documentations/world2act_inference.md) documents all four entry points:

- `examples.video2world_robocasa` — sequential RoboCasa atomic actions
- `examples.video2world_robocasa_cp` — RoboCasa full-task generation
- `examples.video2world_libero` — sequential LIBERO atomic actions
- `examples.video2world_gr00t_franka_realworld` — real-world Franka demos

Inputs, checkpoints, output directories, and GPU partitioning are explicit command-line options. All `.pt` files remain excluded from Git.

## Acknowledgments and license

This implementation builds on [NVIDIA Cosmos Predict2](https://github.com/nvidia-cosmos/cosmos-predict2). Existing NVIDIA copyright notices, [LICENSE](LICENSE), and [ATTRIBUTIONS.md](ATTRIBUTIONS.md) are retained. Upstream Cosmos examples and documentation remain available for reference; their CUDA setup instructions are separate from the AMD workflow above.
