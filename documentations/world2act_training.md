# World2Act training

This release contains two training modes. Both use the Cosmos Predict2 video world model, but they update different parameters.

| Mode | CLI selection | Updated parameters | Required data |
| --- | --- | --- | --- |
| WM pretraining / fine-tuning | `--training_mode wm` (default) | World-model denoising network | Simulation videos and their text embeddings |
| Stage 1: Action VAE | `--training_mode action_vae` | Action VAE and video bridger; world model stays frozen | Videos, text embeddings, and corresponding robot actions |

VLA post-training is a future release. Stage 1 is included now and is separate from VLA post-training.

## Data availability

The data was collected by the World2Act authors. The simulation fine-tuning data is not bundled with this code release and will be published later. The post-training dataset will also be released later. Until then, supply your own compatible local data; cloning the repository does not download either dataset or private model weights.

## Environment and pretrained weights

Follow [AMD/ROCm setup](setup_rocm.md) first. Run commands from the repository root on an AMD GPU machine. The Mac is suitable for Git operations, not these GPU workloads.

Download the public base model and tokenizer, plus T5 text-encoder weights, into `checkpoints/`. The existing downloader also fetches the prompt-refiner and guardrail models; some require accepting their model terms and authenticating with Hugging Face.

```bash
python scripts/download_checkpoints.py \
  --model_types video2world --model_sizes 2B --resolution 480 --fps 16
```

For the commands below, choose a local checkpoint:

```bash
export WM_CHECKPOINT="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"
```

That is a public base model, not World2Act-trained weights. To use your own fine-tuned model, point `WM_CHECKPOINT` at your local checkpoint. Checkpoint files remain ignored by Git.

## Prepare world-model data

Set the dataset root explicitly:

```bash
export WORLD2ACT_DATASET_DIR="/path/to/local/simulation_dataset"
```

The loader expects these paths:

| Path relative to the dataset root | Contents |
| --- | --- |
| `videos/<sample>.mp4` | RGB video, at least 93 frames for the default preset |
| `t5_xxl/<sample>.pickle` | Matching cached T5 embedding, produced by the script below |
| `metadata.csv` | Header row followed by video filename and prompt columns |

Example metadata:

```csv
video,prompt
episode_000001.mp4,"pick the mango from the cabinet"
```

Precompute embeddings (the filename stem must match the video):

```bash
python -m scripts.get_t5_embeddings_from_groot_dataset \
  --dataset_path "$WORLD2ACT_DATASET_DIR" \
  --meta_csv "$WORLD2ACT_DATASET_DIR/metadata.csv" \
  --prompt_prefix ""
```

Use prompts consistent with your video's camera layout; see [inference input conventions](world2act_inference.md). The loader selects matching video/embedding pairs and raises an error for an empty dataset or an unreadable sample. It resizes to 480×832 and samples 93 frames in the released preset.

## Mode 1: world-model fine-tuning

This is the original experiment name and launch structure, with paths supplied explicitly:

```bash
source setup_rocm.sh
export IMAGINAIRE_OUTPUT_ROOT="outputs"
EXP=predict2_video2world_training_2b_groot_gr1_480

torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py \
  --training_mode wm \
  -- experiment="${EXP}" \
  model.config.model_manager_config.dit_path="$WM_CHECKPOINT"
```

`wm` is the default, so omitting `--training_mode wm` preserves the short launch command. It backpropagates the world-model loss and saves normal world-model checkpoints under the configured job output directory. No action files are required for this mode.

The default run is 400,000 updates and saves every 200 updates. Override settings after `--`, for example `trainer.max_iter=1000 checkpoint.save_iter=100`. Use a short run to validate your own data and environment before a full training run.

## Mode 2: Stage 1 Action VAE training

Stage 1 freezes the chosen world model and trains the Action VAE and its video bridger with reconstruction, contrastive alignment, and KL losses. It does not update a VLA or the world model. Use the fine-tuned world-model weights that match your data when available.

Install the additional dataset readers in the prepared environment:

```bash
python -m pip install h5py pyarrow
```

The released Stage 1 trainer supports one GPU per run:

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
  --action_save_iter 100 \
  -- experiment="${EXP}" \
  model.config.model_manager_config.dit_path="$WM_CHECKPOINT" \
  job.group=stage1_action_vae
```

All CLI options such as `--training_mode` and `--action_data_root` go **before** the standalone `--`. Experiment/configuration overrides go after it.

Select the loader that matches your local action data:

| `--action_dataset` | Action data and video pairing convention | Action dimensions |
| --- | --- | --- |
| `simpler_env` | LeRobot-style `data/chunk-*/episode_*.parquet`, optionally `meta/episodes.jsonl`; video named `episode_<id>.mp4` | 7 |
| `robocasa` | RoboCasa `mg` HDF5 files and `video_metadata.json`; video named `<task>_mg_demo_<id>_aa_<index>.mp4` | 12 |
| `libero` | LIBERO HDF5 demonstrations; atomic video filenames encode demo, task, and action order | 7 |
| `franka` | Franka HDF5 demonstrations; video named `<task>-demo-<id>.mp4` | 7 |

The loaders are implemented in [action_dataloader.py](../imaginaire/action_pretraining/action_dataloader.py); check its filename parsing against your exports. For RoboCasa, pass `--action_metadata /path/to/video_metadata.json` if it is not at the action root. These are input formats, not dataset download links. Keep video/action ordering and temporal alignment consistent when preparing data.

Stage 1 uses 128-step padded/truncated action windows, four-step action chunks, and a 32-dimensional latent representation by default. It saves Action VAE/bridger state dictionaries as `checkpoints/action_vae/ckpt_<step>.pt`. `--action_init_checkpoint` can warm-start from one of these files; it does not restore an optimizer, RNG state, or step counter. World-model checkpoints and Action VAE checkpoints are different formats and are not interchangeable.

## Validation status

The release cleanup includes CPU checks for command planning, path handling, checkpoint exclusion, and trainer control flow. Full training and generation still require validation on the authors' AMD environment with compatible data and weights; CPU checks do not establish numerical reproduction.
