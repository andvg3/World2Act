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
import subprocess
import json
import os
import re
import pathlib
from tqdm import tqdm
import cv2

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
    "The robot arm is performing a task. A multi-view video shows that a robot {task}. "
    "The video is split into four views: The top-left view shows the robotic arm from the left side, "
    "the top-right view shows it from the right side, the bottom-left view shows a first-person "
    "perspective from the robot's end-effector (gripper), and the bottom-right view is a black screen "
    "(inactive view). The robot {task}"
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
        "--gpu_index",
        default="0",
        type=str,
        help="GPU index",
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


def get_crop_filter(video_path: str) -> str:
    """
    Runs ffmpeg cropdetect on the video to find the optimal crop parameters.
    Replicates the logic: cropdetect=24:16:0
    """
    # Command to scan the video (skipping first 0.5s, scanning 2s)
    cmd = [
        "ffmpeg", "-hide_banner",
        "-ss", "0.5",
        "-t", "2",
        "-i", video_path,
        "-vf", "cropdetect=24:16:0",
        "-f", "null", "-"
    ]
    
    try:
        # Run ffmpeg and capture stderr (where cropdetect prints info)
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stderr
        
        # Regex to find the last occurrence of crop=w:h:x:y
        matches = re.findall(r"crop=[0-9]+:[0-9]+:[0-9]+:[0-9]+", output)
        
        if matches:
            return matches[-1] # Return the last detected crop
        else:
            return None
            
    except Exception as e:
        # If logging is available, log warning here
        print(f"Error detecting crop for {video_path}: {e}")
        return None

def extract_and_process_last_frame(video_path: str, output_image_path: str, target_size=(256, 256)) -> bool:
    """
    Extracts the last frame from a video using ffmpeg, applies the detected crop,
    and resizes it to target_size.
    """
    if not os.path.exists(video_path):
        return False

    # 1. Get the crop parameters
    crop_filter = get_crop_filter(video_path)
    
    # Base filter chain: Scale to target_size is the final goal
    if crop_filter:
        # Apply detected crop -> Scale to target_size
        vf_chain = f"{crop_filter},scale={target_size[0]}:{target_size[1]}"
    else:
        # Fallback: Just scale if crop detection failed
        vf_chain = f"scale={target_size[0]}:{target_size[1]}"

    # 2. Extract last frame with applied filters
    # -sseof -0.5: Seek to 0.5 seconds before end (to ensure we capture the end)
    # -update 1: Overwrite the image until the stream ends (getting the absolute last frame)
    cmd = [
        "ffmpeg", "-hide_banner", "-y",
        "-sseof", "-0.5", 
        "-i", video_path,
        "-vf", vf_chain,
        "-frames:v", "1",
        "-update", "1",
        output_image_path
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log.info(f"Saving processed image at {output_image_path} with filter: {vf_chain}")
        return True
    except subprocess.CalledProcessError:
        log.warning(f"FFmpeg failed to extract frame from {video_path}")
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
        # output_latent_path = os.path.splitext(output_path)[0] + ".pt"
        # log.info(f"Saving latent tensor to: {output_latent_path}")
        # torch.save(latent_x0.cpu(), output_latent_path)
        # log.success(f"Successfully saved latent tensor to: {output_latent_path}")

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

def generate_video(args, pipe) -> None:
    # 1. Define base paths mapping to your intent
    base_dir = pathlib.Path("/vast/users/tianyu.wang/anv_workspace/ThinkPlan/playground/data/PnPCabToCounter/3_view")
    prompts_dir = base_dir / "prompts"
    images_cosmos_dir = base_dir / "images_cosmos"
    
    # Updated output paths
    out_base = base_dir / "cosmos_full"
    out_video = out_base / "video" 
    out_intermediates = out_base / "intermediates" # Added intermediates directory
    
    # Create required directories
    out_video.mkdir(parents=True, exist_ok=True)
    out_intermediates.mkdir(parents=True, exist_ok=True)

    # 2. Main loop through demos 1 to 54
    for demo_idx in tqdm(range(1, 55), desc="Processing Demos"):
        demo_name = f"demo_{demo_idx}"
        demo_prompt_dir = prompts_dir / demo_name
        
        if not demo_prompt_dir.exists():
            log.warning(f"Skipping {demo_name}: Prompt directory not found.")
            continue

        first_frame_path = images_cosmos_dir / f"demo_{demo_idx}_first_frame.png"
        if not first_frame_path.exists():
            log.warning(f"Skipping {demo_name}: First frame not found at {first_frame_path}.")
            continue

        # 3. Target aa_0 and aa_1 specifically
        prompt_0_path = demo_prompt_dir / "aa_0.txt"
        prompt_1_path = demo_prompt_dir / "aa_1.txt"
        
        # Ensure both prompt files exist before proceeding
        if not prompt_0_path.exists() or not prompt_1_path.exists():
            log.warning(f"Skipping {demo_name}: Requires both aa_0.txt and aa_1.txt.")
            continue
            
        # Read and combine prompts
        with open(prompt_0_path, "r") as f:
            prompt_0_text = f.read().strip()
            
        with open(prompt_1_path, "r") as f:
            prompt_1_text = f.read().strip()
            
        combined_prompt = f"{prompt_0_text} and {prompt_1_text}"

        # Setup intermediate and final file paths
        part1_path = out_intermediates / f"{demo_name}_part1.mp4"
        mid_frame_path = out_intermediates / f"{demo_name}_mid_frame.png"
        part2_path = out_intermediates / f"{demo_name}_part2.mp4"
        concat_list_path = out_intermediates / f"{demo_name}_concat.txt"
        final_video_path = out_video / f"{demo_name}_full_sequence.mp4"
        
        log.info(f"Generating Horizon 1 for {demo_name}...")

        # 4. Generate the FIRST horizon
        success_1 = process_single_generation(
            pipe=pipe,
            input_path=str(first_frame_path),
            prompt=combined_prompt,
            output_path=str(part1_path),
            negative_prompt=args.negative_prompt,
            aspect_ratio=args.aspect_ratio,
            num_conditional_frames=args.num_conditional_frames,
            guidance=args.guidance,
            seed=args.seed,
            prompt_prefix=args.prompt_prefix,
        )

        if not success_1:
            log.error(f"Failed to generate Horizon 1 for {demo_name}. Skipping.")
            continue

        # 5. Extract the last frame of the first horizon
        try:
            cap = cv2.VideoCapture(str(part1_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1) # Target the very last frame
            ret, frame = cap.read()
            
            if not ret:
                log.error(f"Could not extract last frame from {part1_path}")
                cap.release()
                continue
                
            cv2.imwrite(str(mid_frame_path), frame)
            cap.release()
        except Exception as e:
            log.error(f"OpenCV error extracting frame for {demo_name}: {e}")
            continue

        log.info(f"Generating Horizon 2 for {demo_name}...")

        # 6. Generate the SECOND horizon using the extracted mid_frame
        success_2 = process_single_generation(
            pipe=pipe,
            input_path=str(mid_frame_path),
            prompt=combined_prompt,
            output_path=str(part2_path),
            negative_prompt=args.negative_prompt,
            aspect_ratio=args.aspect_ratio,
            num_conditional_frames=args.num_conditional_frames,
            guidance=args.guidance,
            seed=args.seed,
            prompt_prefix=args.prompt_prefix,
        )

        if not success_2:
            log.error(f"Failed to generate Horizon 2 for {demo_name}. Skipping.")
            continue

        # 7. Concatenate the videos using FFMPEG
        log.info(f"Concatenating part 1 and part 2 for {demo_name}...")
        
        # Create a text file containing the files to concatenate
        with open(concat_list_path, "w") as f:
            f.write(f"file '{part1_path.name}'\n")
            f.write(f"file '{part2_path.name}'\n")
            
        # Call ffmpeg concat demuxer (copies streams without re-encoding)
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",                   # Overwrite existing files automatically
            "-f", "concat",         # Use the concat demuxer
            "-safe", "0",           # Avoid unsafe file path errors
            "-i", str(concat_list_path), 
            "-c", "copy",           # Copy codec to prevent quality loss/re-encoding time
            str(final_video_path)
        ]
        
        try:
            # Run inside intermediates dir so relative paths in concat.txt work correctly
            subprocess.run(
                ffmpeg_cmd, 
                cwd=str(out_intermediates), 
                check=True, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            log.info(f"Successfully generated full sequence at {final_video_path}")
        except subprocess.CalledProcessError as e:
            log.error(f"FFmpeg concatenation failed for {demo_name}: {e}")

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