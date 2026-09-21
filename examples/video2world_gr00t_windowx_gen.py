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
A variant of predict2_video2world.py for GR00T models that:
1. Supports prompt prefix for robot task descriptions
2. Turns off the guardrail and prompt refiner
3. Supports two GR00T variants: GR1 and DROID
"""

import argparse
import json
import os
import pathlib

from imaginaire.constants import (
    CosmosPredict2Gr00tModelSize,
    CosmosPredict2Video2WorldAspectRatio,
    get_cosmos_predict2_gr00t_checkpoint,
)

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state
from tqdm import tqdm

from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT, validate_input_file
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video, save_text_prompts


_DEFAULT_MULTIVIEW_TEMPLATE = (
    "The robot arm is performing a task. A single-view video shows that a robot {task}. The view shows the first-person egocentric view from a robot-mounted camera. The robot {task}"
)


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

    parser.add_argument(
        "--seed_start",
        type=int,
        default=12,
        help="First seed index to process.",
    )
    parser.add_argument(
        "--seed_end",
        type=int,
        default=15,
        help="Last seed index to process, inclusive.",
    )

    return parser.parse_args()


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
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False

    # Load models
    log.info(f"Initializing Video2WorldPipeline with GR00T variant: {args.gr00t_variant}")
    pipe = Video2WorldPipeline.from_config(
        config=config,
        dit_path=dit_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=args.load_ema,
        load_prompt_refiner=False,  # Disable prompt refiner for GR00T
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
    # Validate input file
    if not validate_input_file(input_path, num_conditional_frames):
        log.warning(f"Input file validation failed: {input_path}")
        return False

    # Add prefix to prompt
    task = prompt.strip()
    if task.endswith("."):
        task = task[:-1]
    prompt = _DEFAULT_MULTIVIEW_TEMPLATE.format(task=task)
    
    full_prompt = prompt_prefix + prompt
    log.info(f"Running Video2WorldPipeline\ninput: {input_path}\nprompt: {full_prompt}")

    video, prompt_used, latent_x0 = pipe(
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
        # save the generated video
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        log.info(f"Saving generated video to: {output_path}")
        save_image_or_video(video, output_path, fps=16)
        log.success(f"Successfully saved video to: {output_path}")

        # save latent_x0 tensor
        if latent_x0 is not None:
            output_latent_path = os.path.splitext(output_path)[0] + ".pt"
            log.info(f"Saving latent tensor to: {output_latent_path}")
            torch.save(latent_x0.cpu(), output_latent_path)
            log.success(f"Successfully saved latent tensor to: {output_latent_path}")

        # save the prompts used to generate the video
        output_prompt_path = os.path.splitext(output_path)[0] + ".txt"
        prompts_to_save = {
            "prompt": prompt,
            "full_prompt": full_prompt,
            "negative_prompt": negative_prompt,
        }

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
    # Keep original batch JSON behavior if explicitly provided
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

        return

    input_root = pathlib.Path(args.input_path)

    # Preserve original single-file behavior
    if input_root.is_file():
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
        return

    # Recursive WidowX structure:
    # input_root/
    #   simpler_env_widowx/
    #     task_name/
    #       seed_12/
    #         first_frame.png
    #         prompt.txt
    #
    # This works whether input_root is:
    #   first_frame_simpler_env/
    # or:
    #   first_frame_simpler_env/simpler_env_widowx/
    # or:
    #   first_frame_simpler_env/simpler_env_widowx/widowx_carrot_on_plate/
    seed_jobs = []

    for seed_dir in sorted(input_root.rglob("seed_*")):
        if not seed_dir.is_dir():
            continue

        try:
            seed_idx = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            log.warning(f"Skipping malformed seed folder: {seed_dir}")
            continue

        if seed_idx < args.seed_start or seed_idx > args.seed_end:
            continue

        input_frame = seed_dir / "first_frame.png"
        prompt_path = seed_dir / "prompt.txt"

        if not input_frame.exists():
            log.warning(f"Skipping {seed_dir}: first_frame.png not found.")
            continue

        if not prompt_path.exists():
            log.warning(f"Skipping {seed_dir}: prompt.txt not found.")
            continue

        seed_jobs.append((seed_idx, seed_dir, input_frame, prompt_path))

    log.info(
        f"Found {len(seed_jobs)} WidowX seed folders under {input_root} "
        f"for seeds [{args.seed_start}, {args.seed_end}]"
    )

    if len(seed_jobs) == 0:
        log.warning(
            "No valid WidowX seed folders found. Expected folders like: "
            "<input_root>/.../<task_name>/seed_12/first_frame.png and prompt.txt"
        )
        return

    for seed_idx, seed_dir, input_frame, prompt_path in tqdm(seed_jobs, desc="Processing WidowX seeds"):
        with open(prompt_path, "r") as f:
            prompt_text = f.read().strip()

        if not prompt_text:
            log.warning(f"Skipping {seed_dir}: prompt.txt is empty.")
            continue

        task_name = seed_dir.parent.name
        output_video_path = seed_dir / "generated_video.mp4"

        log.info(f"Generating WidowX video for task={task_name}, seed={seed_idx}")
        log.info(f"Input frame: {input_frame}")
        log.info(f"Prompt file: {prompt_path}")
        log.info(f"Output video: {output_video_path}")

        process_single_generation(
            pipe=pipe,
            input_path=str(input_frame),
            prompt=prompt_text,
            output_path=str(output_video_path),
            negative_prompt=args.negative_prompt,
            aspect_ratio=args.aspect_ratio,
            num_conditional_frames=args.num_conditional_frames,
            guidance=args.guidance,
            seed=args.seed,
            prompt_prefix=args.prompt_prefix,
        )

    return


def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    try:
        pipe = setup_pipeline(args)
        generate_video(args, pipe)
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()
