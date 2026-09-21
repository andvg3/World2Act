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
4. Iterates through specific seed folders
"""

import argparse
import json
import os
import pathlib
import re
import cv2
import numpy as np
from PIL import Image

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from megatron.core import parallel_state
from tqdm import tqdm

from imaginaire.constants import (
    CosmosPredict2Gr00tModelSize,
    CosmosPredict2Video2WorldAspectRatio,
    get_cosmos_predict2_gr00t_checkpoint,
)
from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT, validate_input_file
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video, save_text_prompts

_DEFAULT_MULTIVIEW_TEMPLATE = (
    "The robot arm is performing a task. A multi-view video shows that a robot {task}. "
    "The video is split into four views: the top-left view shows the robotic arm from the agent-view side, "
    "the top-right view shows it from the eye-in-hand first-person perspective from the robot's end-effector (gripper), "
    "the bottom-left view is a black screen (inactive view), and the bottom-right view is a black screen (inactive view). "
    "The robot {task}."
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
        help="Text prompt (unused in folder mode, but kept for compatibility)",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="assets/video2world/input0.jpg",
        help="Root path to input folder structure",
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
        help="Path to JSON file containing batch inputs (Optional override).",
    )
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument(
        "--save_path",
        type=str,
        default="output/generated_video.mp4",
        help="Default save path (unused in folder mode)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference",
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
        "--gpu_index", type=int, default=0, help="Index of the GPU for sharding logic"
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


def extract_and_process_last_frame(video_path: str, output_image_path: str, target_size=(256, 256)) -> bool:
    """
    Extracts the last frame, crops it to a center square based on the width,
    and resizes it to the target dimensions.
    """
    if not os.path.exists(video_path):
        log.warning(f"Video path does not exist: {video_path}")
        return False

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.warning(f"Could not open video: {video_path}")
            return False

        # Get total frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            log.warning(f"Video has no frames: {video_path}")
            return False

        # Set position to last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            log.warning(f"Failed to read last frame from: {video_path}")
            return False

        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # --- Center Square Crop Logic ---
        width, height = img.size
        # The square side will be the smaller of the two dimensions (480 in your case)
        side_length = min(width, height)
        
        left = (width - side_length) / 2
        top = (height - side_length) / 2
        right = (width + side_length) / 2
        bottom = (height + side_length) / 2

        # Crop to the center square
        img = img.crop((left, top, right, bottom))

        # --- Resize to Target Size ---
        if target_size is not None:
             img = img.resize(target_size, Image.LANCZOS)

        img.save(output_image_path)
        log.info(f"Extracted, cropped, and resized last frame to: {output_image_path}")
        return True

    except Exception as e:
        log.error(f"Error extracting last frame: {e}")
        return False


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

    # Note: This unpacking expects the pipeline to return 3 values.
    # If using standard public pipe, ensure it supports returning latent_x0.
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

        # Save the latent_x0 tensor
        output_latent_path = os.path.splitext(output_path)[0] + ".pt"
        log.info(f"Saving latent tensor to: {output_latent_path}")
        torch.save(latent_x0.cpu(), output_latent_path)
        log.success(f"Successfully saved latent tensor to: {output_latent_path}")

        # save the prompts used to generate the video
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
    input_root = pathlib.Path(args.input_path)
    
    # --- THE MAPPING ---
    # Maps GPU 0-7 to Seed folders 12-82
    seed_mapping = {0: 12, 1: 22, 2: 32, 3: 42, 4: 52, 5: 62, 6: 72, 7: 82}
    
    # Ensure args.gpu_index is treated as an int key
    gpu_id = int(args.gpu_index)
    target_seed_id = seed_mapping[gpu_id]

    log.info(f"🚀 GPU {gpu_id} active. Target Seed Folder: {target_seed_id}")

    # 1. Identify all Episode folders
    # Structure: <suite>/<task>/<seed>/<eps_idx>/first_frame.png
    episode_dirs = sorted(list(set([p.parent for p in input_root.rglob("first_frame.png")])))

    for episode_dir in tqdm(episode_dirs, desc=f"GPU {gpu_id} | Seed {target_seed_id}"):
        # episode_dir is .../<seed>/<eps_idx>
        # episode_dir.parent.name is the seed folder (e.g., "32")
        try:
            current_seed_val = int(episode_dir.parent.name)
        except ValueError:
            continue
        
        # Only process if it matches the mapping for this GPU
        if current_seed_val != target_seed_id:
            continue

        log.info(f"✅ Processing: {episode_dir}")

        # 2. Find and Sort Action Files (aa_*.txt)
        action_files = sorted(
            list(episode_dir.glob("aa_*.txt")), 
            key=lambda p: int(re.search(r"aa_(\d+)\.txt", p.name).group(1)) if re.search(r"aa_(\d+)\.txt", p.name) else -1
        )

        if not action_files:
            continue

        # 3. Sequential Processing
        current_input_frame = episode_dir / "first_frame.png"
        
        for i, prompt_path in enumerate(action_files):
            # Extract index for naming
            match = re.search(r"aa_(\d+)\.txt", prompt_path.name)
            aa_index = int(match.group(1)) if match else i

            output_video_path = episode_dir / f"output_aa_{aa_index}.mp4"
            
            # Skip if already exists
            if output_video_path.exists():
                current_input_frame = episode_dir / f"last_frame_aa_{aa_index}.png"
                continue

            with open(prompt_path, "r") as f:
                prompt_text = f.read().strip()

            success = process_single_generation(
                pipe=pipe,
                input_path=str(current_input_frame),
                prompt=prompt_text,
                output_path=str(output_video_path),
                negative_prompt=args.negative_prompt,
                aspect_ratio=args.aspect_ratio,
                num_conditional_frames=args.num_conditional_frames,
                guidance=args.guidance,
                seed=args.seed, 
                prompt_prefix=args.prompt_prefix,
            )

            if not success:
                break

            # 4. Feedback Loop
            if i < len(action_files) - 1:
                next_input_path = episode_dir / f"last_frame_aa_{aa_index}.png"
                img_proc_success = extract_and_process_last_frame(
                    video_path=str(output_video_path),
                    output_image_path=str(next_input_path),
                    target_size=(256, 256)
                )
                if img_proc_success:
                    current_input_frame = next_input_path
                else:
                    break


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