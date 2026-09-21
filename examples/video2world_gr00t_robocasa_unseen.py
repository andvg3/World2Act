# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A RoboCasa unseen-task variant of predict2_video2world.py for GR00T models that:
1. Supports prompt prefix for robot task descriptions
2. Turns off the guardrail and prompt refiner
3. Supports two GR00T variants: GR1 and DROID

Added:
- RoboCasa dataset mode via --input_dir (expects metadata.csv, videos/, metas/, and t5_xxl/)
- Deterministic selection of the first N atomic samples from every held-out task
- Extraction of the first frame from each selected video
- Sharding via --shard_rank / --shard_world_size (for running one process per GPU in separate terminals)
"""

import argparse
import csv
import json
import os
import re

import imageio.v3 as iio
from decord import VideoReader, cpu
from imaginaire.constants import (
    CosmosPredict2Gr00tModelSize,
    CosmosPredict2Video2WorldAspectRatio,
    get_cosmos_predict2_gr00t_checkpoint,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state
from tqdm import tqdm

from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT, validate_input_file
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video, save_text_prompts


# _DEFAULT_MULTIVIEW_TEMPLATE = (
#     "The robot arm is performing a task. A multi-view video shows that a robot {task}. "
#     "The video is split into four views: The top-left view shows the robotic arm from the left side, "
#     "the top-right view shows it from the right side, the bottom-left view shows a first-person "
#     "perspective from the robot's end-effector (gripper), and the bottom-right view is a black screen "
#     "(inactive view). The robot {task}"
# )

_DEFAULT_MULTIVIEW_TEMPLATE = (
    "The robot arm is performing a task. A multi-view video shows that a robot {task}. "
    "The video is split into four views: the top-left view shows the robotic arm from the agent-view side, "
    "the top-right view shows it from the eye-in-hand first-person perspective from the robot's end-effector (gripper), "
    "the bottom-left view is a black screen (inactive view), and the bottom-right view is a black screen (inactive view). "
    "The robot {task}."
)


_TRAINING_PROMPT_PREFIX = "The robot arm is performing a task. "
_EXPECTED_UNSEEN_TASKS = (
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter",
    "PnPCounterToStove",
    "PnPStoveToCounter",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "TurnOnStove",
)
_TASK_PATTERN = re.compile(r"^(?P<task>.+?)_mg_demo_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T Video-to-World Generation with Cosmos Predict2")
    parser.add_argument(
        "--model_size",
        choices=CosmosPredict2Gr00tModelSize.__args__,
        default="14B",
        help="Size of the model to use for GR00T video-to-world generation",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    parser.add_argument(
        "--load_ema",
        action="store_true",
        help="Use EMA weights for generation.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Text prompt for video generation",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="assets/video2world/input0.jpg",
        help="Path to input image or video for conditioning (include file extension)",
    )

    # RoboCasa unseen-task dataset mode
    parser.add_argument(
        "--input_dir",
        type=str,
        default="",
        help="RoboCasa dataset root containing metadata.csv, videos/, metas/, and t5_xxl/",
    )
    # Sharding across processes (run one terminal per GPU)
    parser.add_argument(
        "--shard_rank",
        type=int,
        default=0,
        help="Shard rank for folder loop (0..world_size-1).",
    )
    parser.add_argument(
        "--shard_world_size",
        type=int,
        default=1,
        help="Number of shards for folder loop.",
    )
    parser.add_argument(
        "--frame_name",
        type=str,
        default="last_frame.jpg",
        help="Frame filename inside each demo folder (folder loop mode).",
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default="prompt_text.txt",
        help="Prompt filename inside each demo folder (folder loop mode).",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="generated.mp4",
        help="Generated video filename inside each selected sample output directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/robocasa_atomicgeneralize_unseen_iter_000018800",
        help="Output root for extracted first frames, exact prompts, generated videos, and logs.",
    )
    parser.add_argument(
        "--samples_per_task",
        type=int,
        default=4,
        help="Number of metadata-order samples to generate for each held-out task.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate outputs that already exist.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate task selection, files, prompts, and sharding without loading the model or writing files.",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Validate inputs and extract first-frame images without loading the model.",
    )

    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative text prompt for video-to-world generation",
    )
    parser.add_argument(
        "--aspect_ratio",
        choices=CosmosPredict2Video2WorldAspectRatio.__args__,
        default="16:9",
        type=str,
        help="Aspect ratio of the generated output (width:height)",
    )
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=1,
        choices=[1, 5],
        help="Number of frames to condition on (1 for single frame, 5 for multi-frame conditioning)",
    )
    parser.add_argument(
        "--batch_input_json",
        type=str,
        default=None,
        help="Path to JSON file containing batch inputs. Each entry should have 'input_video', 'prompt', and 'output_video' fields.",
    )
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_video.mp4",
        help="Path to save the generated video (include file extension)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference (should be a divisor of the total frames)",
    )
    parser.add_argument(
        "--disable_guardrail",
        action="store_true",
        help="Disable guardrail checks on prompts",
    )
    parser.add_argument(
        "--gr00t_variant", type=str, required=True, help="GR00T variant to use", choices=["gr1", "droid"]
    )
    parser.add_argument(
        "--prompt_prefix", type=str, default="The robot arm is performing a task. ", help="Prefix to add to all prompts"
    )

    return parser.parse_args()


def _resolve_shard_args(args: argparse.Namespace):
    """
    IMPORTANT for your use-case (one terminal per GPU with CUDA_VISIBLE_DEVICES set):
    - Do NOT call torch.cuda.set_device(rank) here.
    - Each process should see exactly 1 GPU (visible as cuda:0), so pipeline uses cuda:0 safely.
    """
    rank = int(args.shard_rank)
    world = int(args.shard_world_size)
    if world <= 0:
        world = 1
    if rank < 0 or rank >= world:
        raise ValueError(f"Invalid shard_rank={rank} for shard_world_size={world}")
    return rank, world


def _load_robocasa_unseen_items(dataset_root: str, samples_per_task: int) -> list[dict[str, str]]:
    if samples_per_task <= 0:
        raise ValueError(f"samples_per_task must be positive, got {samples_per_task}")

    metadata_path = os.path.join(dataset_root, "metadata.csv")
    videos_dir = os.path.join(dataset_root, "videos")
    metas_dir = os.path.join(dataset_root, "metas")
    t5_dir = os.path.join(dataset_root, "t5_xxl")
    for required_path in (metadata_path, videos_dir, metas_dir, t5_dir):
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required RoboCasa dataset path is missing: {required_path}")

    rows_by_task: dict[str, list[dict[str, str]]] = {task: [] for task in _EXPECTED_UNSEEN_TASKS}
    observed_tasks: set[str] = set()
    with open(metadata_path, newline="", encoding="utf-8") as metadata_file:
        reader = csv.DictReader(metadata_file)
        if reader.fieldnames is None or not {"file_name", "text"}.issubset(reader.fieldnames):
            raise ValueError(f"Expected file_name,text columns in {metadata_path}; got {reader.fieldnames}")

        for row in reader:
            file_name = row["file_name"].strip()
            match = _TASK_PATTERN.match(file_name)
            if match is None:
                raise ValueError(f"Cannot extract task from metadata filename: {file_name}")
            task = match.group("task")
            observed_tasks.add(task)
            if task in rows_by_task and len(rows_by_task[task]) < samples_per_task:
                rows_by_task[task].append(row)

    expected_tasks = set(_EXPECTED_UNSEEN_TASKS)
    if observed_tasks != expected_tasks:
        missing = sorted(expected_tasks - observed_tasks)
        unexpected = sorted(observed_tasks - expected_tasks)
        raise ValueError(f"Unexpected held-out task split; missing={missing}, unexpected={unexpected}")

    selected: list[dict[str, str]] = []
    for task in _EXPECTED_UNSEEN_TASKS:
        task_rows = rows_by_task[task]
        if len(task_rows) != samples_per_task:
            raise ValueError(f"Task {task} has only {len(task_rows)} selectable rows; need {samples_per_task}")

        for row in task_rows:
            file_name = row["file_name"].strip()
            stem, extension = os.path.splitext(file_name)
            if extension.lower() != ".mp4":
                raise ValueError(f"Expected an MP4 metadata entry, got: {file_name}")

            video_path = os.path.join(videos_dir, file_name)
            prompt_path = os.path.join(metas_dir, f"{stem}.txt")
            t5_path = os.path.join(t5_dir, f"{stem}.pickle")
            for required_path in (video_path, prompt_path, t5_path):
                if not os.path.isfile(required_path):
                    raise FileNotFoundError(f"Selected sample asset is missing: {required_path}")

            with open(prompt_path, encoding="utf-8") as prompt_file:
                training_prompt = prompt_file.read().strip()
            expected_training_prompt = _TRAINING_PROMPT_PREFIX + row["text"].strip()
            if training_prompt != expected_training_prompt:
                raise ValueError(
                    f"Training prompt mismatch for {file_name}: metas text is not the exact prefixed metadata text"
                )

            selected.append(
                {
                    "task": task,
                    "stem": stem,
                    "video_path": video_path,
                    "prompt": training_prompt,
                    "prompt_path": prompt_path,
                    "t5_path": t5_path,
                }
            )

    return selected


def _items_for_shard(items: list[dict[str, str]], rank: int, world: int) -> list[dict[str, str]]:
    task_rank = {task: task_index % world for task_index, task in enumerate(_EXPECTED_UNSEEN_TASKS)}
    return [item for item in items if task_rank[item["task"]] == rank]


def _sample_output_paths(args: argparse.Namespace, item: dict[str, str]) -> tuple[str, str, str]:
    sample_dir = os.path.join(args.output_dir, item["task"], item["stem"])
    first_frame_path = os.path.join(sample_dir, "first_frame.png")
    exact_prompt_path = os.path.join(sample_dir, "training_prompt_exact.txt")
    output_path = os.path.join(sample_dir, args.out_name)
    return first_frame_path, exact_prompt_path, output_path


def _prepare_first_frame(video_path: str, output_path: str, overwrite: bool) -> None:
    if os.path.isfile(output_path) and not overwrite:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    if len(reader) == 0:
        raise ValueError(f"Video has no frames: {video_path}")
    first_frame = reader[0].asnumpy()
    del reader
    iio.imwrite(output_path, first_frame)


def _prepare_robocasa_shard(args: argparse.Namespace, write_inputs: bool) -> list[dict[str, str]]:
    rank, world = _resolve_shard_args(args)
    selected = _load_robocasa_unseen_items(args.input_dir, args.samples_per_task)
    my_items = _items_for_shard(selected, rank, world)
    assigned_tasks = [task for task in _EXPECTED_UNSEEN_TASKS if _EXPECTED_UNSEEN_TASKS.index(task) % world == rank]
    log.info(
        f"[RoboCasa] validated {len(selected)} samples across {len(_EXPECTED_UNSEEN_TASKS)} held-out tasks; "
        f"rank={rank}/{world} has {len(my_items)} samples from tasks={assigned_tasks}"
    )

    if write_inputs:
        for item in tqdm(my_items, desc=f"Prepare rank={rank}/{world}"):
            first_frame_path, exact_prompt_path, _ = _sample_output_paths(args, item)
            _prepare_first_frame(item["video_path"], first_frame_path, args.overwrite)
            os.makedirs(os.path.dirname(exact_prompt_path), exist_ok=True)
            with open(exact_prompt_path, "w", encoding="utf-8") as prompt_file:
                prompt_file.write(item["prompt"])

    return my_items


def setup_pipeline(args: argparse.Namespace):
    resolution = "480"
    fps = 16
    config = get_cosmos_predict2_video2world_pipeline(model_size=args.model_size, resolution=resolution, fps=fps)
    config.prompt_refiner_config.enabled = False
    if args.dit_path:
        dit_path = args.dit_path
    else:
        dit_path = get_cosmos_predict2_gr00t_checkpoint(
            gr00t_variant=args.gr00t_variant,
            model_size=args.model_size,
            resolution=resolution,
            fps=fps,
            aspect_ratio=args.aspect_ratio,
        )
    log.info(f"Loading model from: {dit_path}")

    misc.set_random_seed(seed=args.seed, by_rank=True)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Keep original behavior: only init distributed context-parallel if num_gpus > 1
    if args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")

    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False

    log.info(f"Initializing Video2WorldPipeline with GR00T variant: {args.gr00t_variant}")
    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=dit_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=args.load_ema,
        load_prompt_refiner=False,
    )

    return pipe


def process_single_generation(
    pipe: Video2WorldPipeline,
    input_path: str,
    prompt: str,
    output_path: str,
    negative_prompt: str,
    aspect_ratio: str,
    num_conditional_frames: int,
    guidance: float,
    seed: int,
    prompt_prefix: str,
) -> bool:
    if not validate_input_file(input_path, num_conditional_frames):
        log.warning(f"Input file validation failed: {input_path}")
        return False

    full_prompt = prompt_prefix + prompt
    log.info(f"Running Video2WorldPipeline\ninput: {input_path}\nprompt: {full_prompt}")

    video, prompt_used = pipe(
        prompt=full_prompt,
        negative_prompt=negative_prompt,
        aspect_ratio=aspect_ratio,
        input_path=input_path,
        num_conditional_frames=num_conditional_frames,
        guidance=guidance,
        seed=seed,
        return_prompt=True,
    )

    if video is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving generated video to: {output_path}")
        save_image_or_video(video, output_path, fps=16)
        log.success(f"Successfully saved video to: {output_path}")

        output_prompt_path = os.path.splitext(output_path)[0] + ".txt"
        prompts_to_save = {"prompt": prompt, "negative_prompt": negative_prompt}
        if (
            pipe.prompt_refiner is not None
            and getattr(pipe.config, "prompt_refiner_config", None) is not None
            and getattr(pipe.config.prompt_refiner_config, "enabled", False)
        ):
            prompts_to_save["refined_prompt"] = prompt_used
        save_text_prompts(prompts_to_save, output_prompt_path)
        log.success(f"Successfully saved prompt file to: {output_prompt_path}")
        return True
    return False


def generate_video(args: argparse.Namespace, pipe: Video2WorldPipeline) -> None:
    # RoboCasa unseen-task mode (sharded by whole task).
    if args.input_dir:
        rank, world = _resolve_shard_args(args)
        my_items = _prepare_robocasa_shard(args, write_inputs=True)
        ok = skip = fail = 0
        for item in tqdm(my_items, desc=f"RoboCasa rank={rank}/{world}"):
            first_frame_path, _, output_path = _sample_output_paths(args, item)
            if os.path.isfile(output_path) and not args.overwrite:
                log.info(f"Output exists, skipping: {output_path}")
                skip += 1
                continue

            success = process_single_generation(
                pipe=pipe,
                input_path=first_frame_path,
                prompt=item["prompt"],
                output_path=output_path,
                negative_prompt=args.negative_prompt,
                aspect_ratio=args.aspect_ratio,
                num_conditional_frames=args.num_conditional_frames,
                guidance=args.guidance,
                seed=args.seed,
                prompt_prefix="",  # metas/*.txt is already the exact full training prompt.
            )

            if success:
                ok += 1
            else:
                fail += 1

        log.info(f"[RoboCasa rank={rank}] done. OK={ok} SKIP={skip} FAIL={fail}")
        return

    # Original batch mode
    if args.batch_input_json is not None:
        log.info(f"Loading batch inputs from JSON file: {args.batch_input_json}")
        with open(args.batch_input_json) as f:
            batch_inputs = json.load(f)

        for idx, item in enumerate(tqdm(batch_inputs)):
            input_video = item.get("input_video", "")
            prompt = item.get("prompt", "")
            output_video = item.get("output_video", f"output_{idx}.mp4")

            if not input_video or not prompt:
                log.warning(f"Skipping item {idx}: Missing input_video or prompt")
                continue

            process_single_generation(
                pipe=pipe,
                input_path=input_video,
                prompt=prompt,
                output_path=output_video,
                negative_prompt=args.negative_prompt,
                aspect_ratio=args.aspect_ratio,
                num_conditional_frames=args.num_conditional_frames,
                guidance=args.guidance,
                seed=args.seed,
                prompt_prefix=args.prompt_prefix,
            )
    else:
        # Original single mode
        process_single_generation(
            pipe=pipe,
            input_path=args.input_path,
            prompt=args.prompt,
            output_path=args.save_path,
            negative_prompt=args.negative_prompt,
            aspect_ratio=args.aspect_ratio,
            num_conditional_frames=args.num_conditional_frames,
            guidance=args.guidance,
            seed=args.seed,
            prompt_prefix=args.prompt_prefix,
        )


def cleanup_distributed():
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.validate_only and args.prepare_only:
            raise ValueError("Choose at most one of --validate_only and --prepare_only")
        if args.validate_only or args.prepare_only:
            if not args.input_dir:
                raise ValueError("--input_dir is required with --validate_only or --prepare_only")
            _prepare_robocasa_shard(args, write_inputs=args.prepare_only)
            mode = "preparation" if args.prepare_only else "validation"
            log.success(f"RoboCasa unseen-task {mode} completed successfully")
        else:
            pipe = setup_pipeline(args)
            generate_video(args, pipe)
    finally:
        cleanup_distributed()
