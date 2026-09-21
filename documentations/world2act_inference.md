# World2Act inference

Run these commands from the repository root in the prepared [AMD environment](setup_rocm.md). Inputs and checkpoints are local: the authors' data and private weights are not included in this release.

## Choose weights and inputs

```bash
source setup_rocm.sh
export WM_CHECKPOINT="/path/to/local/world_model.pt"
export ROBOCASA_INPUT="/path/to/local/first_frame_groot"
export ROBOCASA_FULL_INPUT="/path/to/local/first_frame_full_cosmos"
export LIBERO_INPUT="/path/to/local/first_frame_libero_groot"
export FRANKA_INPUT="/path/to/local/first_frames_realworld"
```

`WM_CHECKPOINT` is a world-model checkpoint, not an Action VAE checkpoint. For a public base-model smoke run, use `checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt` after downloading it as described in [training](world2act_training.md). This does not reproduce a fine-tuned World2Act model. The 2B commands require `--dit_path`; they no longer look for a private checkpoint hidden in the code.

## Commands

### RoboCasa: sequential atomic actions

```bash
python -m examples.video2world_robocasa \
  --model_size 2B --gr00t_variant droid \
  --dit_path "$WM_CHECKPOINT" \
  --prompt "pick the mango from the cabinet" \
  --input_path "$ROBOCASA_INPUT" \
  --prompt_prefix "" --disable_guardrail \
  --save_path output/robocasa.mp4 --gpu_index 0
```

### RoboCasa: full-task generation

```bash
python -m examples.video2world_robocasa_cp \
  --model_size 2B --gr00t_variant droid \
  --dit_path "$WM_CHECKPOINT" \
  --prompt "pick the mango from the cabinet" \
  --input_path "$ROBOCASA_FULL_INPUT" \
  --prompt_prefix "" --disable_guardrail \
  --save_path output/robocasa_full.mp4 --gpu_index 0
```

### LIBERO: sequential atomic actions

```bash
python -m examples.video2world_libero \
  --model_size 2B --gr00t_variant droid \
  --dit_path "$WM_CHECKPOINT" \
  --prompt "pick the mango from the cabinet" \
  --input_path "$LIBERO_INPUT" \
  --prompt_prefix "" --disable_guardrail \
  --save_path output/libero.mp4 --gpu_index 0
```

### Franka: real-world demos

```bash
python -m examples.video2world_gr00t_franka_realworld \
  --model_size 2B --gr00t_variant droid \
  --dit_path "$WM_CHECKPOINT" \
  --prompt "pick the mango from the cabinet" \
  --input_path "$FRANKA_INPUT" \
  --prompt_prefix "" --disable_guardrail \
  --save_path output/franka.mp4 --gpu_index 0
```

The commands retain the original workflow's `--disable_guardrail` option. Omit it to enable the existing checks; that requires the corresponding guardrail models. The GR00T prompt refiner is disabled in these robot-task workflows.

## Input layouts and prompts

All four commands accept a single image/video with `--prompt`, or a directory tree containing episodes:

| Entry point | Files in each episode directory | Behavior |
| --- | --- | --- |
| `video2world_robocasa` | `first_frame.png`, `aa_0.txt`, `aa_1.txt`, … | Numeric action order; the generated final frame conditions the next action |
| `video2world_robocasa_cp` | `first_frame.png`, `prompt.txt` | One full-task video per episode |
| `video2world_libero` | `first_frame.png`, `aa_0.txt`, `aa_1.txt`, … | Sequential actions with the LIBERO camera-layout prompt |
| `video2world_gr00t_franka_realworld` | `frame0.jpg`, `prompt.txt` | One video per real-world demo |

Episodes are found recursively, so existing task/seed/episode folders can be kept. No particular seed values or eight-GPU arrangement are required. Atomic-action files take precedence in the two sequential workflows. Otherwise `prompt.txt` takes precedence over the CLI `--prompt`, which acts as a fallback.

The RoboCasa and Franka templates describe the existing four-view grid (left, right, end-effector, inactive). LIBERO describes agent-view, eye-in-hand, and two inactive views. Prepare inputs to match the selected template. `--prompt_prefix ""` removes only the extra prefix, not the camera-layout template.

Use `--num_conditional_frames 1` for images. Five-frame conditioning requires a suitable video input. Sequential directory workflows use single-frame conditioning.

## Outputs and multiple GPUs

For a single file, `--save_path` is the exact `.mp4` output filename. For a directory, the `.mp4` suffix is removed to form an output root. Relative episode paths are preserved beneath it. For example:

| Input / option | Output |
| --- | --- |
| Input `<root>/task/seed_12/episode_0/first_frame.png`, `--save_path output/robocasa.mp4` | `output/robocasa/task/seed_12/episode_0/output_aa_0.mp4` |
| Full-task directory with the same episode structure | `output/robocasa_full/task/seed_12/episode_0/output_video.mp4` |

Prompts (`.txt`), generated latents (`.pt`), and any feedback frames are stored beside each generated video. Re-running the same output location overwrites its results; choose a new `--save_path` to keep another run. Outputs stay outside the input dataset tree. All `.pt` files and the default `output/` directory are ignored by Git.

By default one process handles every episode. `--gpu_index` selects the visible GPU; it does not filter seed folders. To split the work between two independent processes, use the same input/output roots and:

```bash
# Append to the chosen command in worker 0:
--gpu_index 0 --num_shards 2 --shard_index 0

# Append to the chosen command in worker 1:
--gpu_index 1 --num_shards 2 --shard_index 1
```

Partitioning assigns entire episodes, so atomic sequences remain together. Each process uses one GPU (`--num_gpus 1`). Add `--dry_run` to any command to inspect its planned inputs/outputs without importing the GPU stack or loading a checkpoint. This checks the file layout, not image decoding or GPU compatibility.
