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

"""Shared CLI for the four released World2Act robot-video workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


MULTIVIEW_PROMPT = (
    "The robot arm is performing a task. A multi-view video shows that a robot {task}. "
    "The video is split into four views: The top-left view shows the robotic arm from the left side, "
    "the top-right view shows it from the right side, the bottom-left view shows a first-person "
    "perspective from the robot's end-effector (gripper), and the bottom-right view is a black screen "
    "(inactive view). The robot {task}"
)
LIBERO_PROMPT = (
    "The robot arm is performing a task. A multi-view video shows that a robot {task}. "
    "The video is split into four views: the top-left view shows the robotic arm from the agent-view side, "
    "the top-right view shows it from the eye-in-hand first-person perspective from the robot's end-effector (gripper), "
    "the bottom-left view is a black screen (inactive view), and the bottom-right view is a black screen (inactive view). "
    "The robot {task}."
)
PROFILES = {"robocasa", "robocasa_cp", "libero", "franka"}


@dataclass(frozen=True)
class Generation:
    input_path: Path
    prompt: str
    output_path: Path
    next_frame_path: Path | None = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_size", choices=["2B", "14B"], default="2B")
    parser.add_argument("--dit_path", default="", help="Local world-model weights; required for 2B")
    parser.add_argument("--load_ema", action="store_true")
    parser.add_argument("--gr00t_variant", choices=["gr1", "droid"], required=True)
    parser.add_argument("--input_path", required=True, help="A conditioning image/video or an episode directory tree")
    parser.add_argument("--prompt", default="", help="Prompt for a single file; fallback when a directory has no prompt file")
    parser.add_argument("--prompt_prefix", default="The robot arm is performing a task. ")
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--aspect_ratio", choices=["1:1", "4:3", "3:4", "16:9", "9:16"], default="16:9")
    parser.add_argument("--num_conditional_frames", type=int, choices=[1, 5], default=1)
    parser.add_argument("--guidance", type=float, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", default="output/sample_test.mp4",
                        help="Single-file output; for directory input, .mp4 is removed to form the output root")
    parser.add_argument("--gpu_index", type=int, default=0, help="Index within the visible devices")
    parser.add_argument("--num_gpus", type=int, choices=[1], default=1,
                        help="One GPU per process; use num_shards/shard_index for independent workers")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--disable_guardrail", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print the input/output plan without loading models")
    args = parser.parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    if args.gpu_index < 0:
        parser.error("gpu_index must be nonnegative")
    return args


def plan_generations(args, profile: str) -> list[list[Generation]]:
    """Plan complete episodes so sequential actions stay on the same worker."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown inference profile: {profile}")
    source = Path(args.input_path).expanduser()
    destination = Path(args.save_path).expanduser()
    if source.is_file():
        if not args.prompt.strip():
            raise ValueError("A single input file requires --prompt")
        if destination.suffix.lower() != ".mp4":
            raise ValueError("For a single input file, --save_path must end in .mp4")
        if source.resolve() == destination.resolve():
            raise ValueError("Input and output paths must differ")
        return [[Generation(source, args.prompt.strip(), destination)]] if args.shard_index == 0 else []
    if args.num_conditional_frames != 1:
        raise ValueError("Directory workflows require --num_conditional_frames 1")
    if not source.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    if destination.suffix.lower() == ".mp4":
        destination = destination.with_suffix("")
    if destination.resolve() == source.resolve() or source.resolve() in destination.resolve().parents:
        raise ValueError("Keep --save_path outside the input directory tree")
    frame_name = "frame0.jpg" if profile == "franka" else "first_frame.png"
    episodes = sorted({frame.parent for frame in source.rglob(frame_name)})
    if not episodes:
        raise ValueError(f"No episodes containing {frame_name} found under {source}")
    sequences = []
    for index, episode in enumerate(episodes):
        if index % args.num_shards != args.shard_index:
            continue
        prompts = []
        if profile in {"robocasa", "libero"}:
            action_files = [p for p in episode.glob("aa_*.txt") if re.fullmatch(r"aa_\d+\.txt", p.name)]
            for path in sorted(action_files, key=lambda p: int(p.stem[3:])):
                prompts.append((path.stem, path.read_text().strip()))
        if not prompts:
            prompt_file = episode / "prompt.txt"
            prompt = prompt_file.read_text().strip() if prompt_file.is_file() else args.prompt.strip()
            prompts = [("video", prompt)]
        if any(not prompt for _, prompt in prompts):
            raise ValueError(f"Empty/missing prompt in {episode}; provide prompt.txt, aa_N.txt, or --prompt")
        output_dir = destination / episode.relative_to(source)
        current_input = episode / frame_name
        sequence = []
        for prompt_index, (name, prompt) in enumerate(prompts):
            next_frame = output_dir / f"last_frame_{name}.png" if prompt_index + 1 < len(prompts) else None
            sequence.append(Generation(current_input, prompt, output_dir / f"output_{name}.mp4", next_frame))
            if next_frame is not None:
                current_input = next_frame
        sequences.append(sequence)
    return sequences


def setup_pipeline(args):
    import torch

    from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
    from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
    from imaginaire.constants import get_cosmos_predict2_gr00t_checkpoint
    from imaginaire.utils import misc

    torch.cuda.set_device(args.gpu_index)
    config = get_cosmos_predict2_video2world_pipeline(model_size=args.model_size, resolution="480", fps=16)
    config.prompt_refiner_config.enabled = False
    config.guardrail_config.enabled = not args.disable_guardrail
    dit_path = args.dit_path or get_cosmos_predict2_gr00t_checkpoint(
        gr00t_variant=args.gr00t_variant, model_size=args.model_size,
        resolution="480", fps=16, aspect_ratio=args.aspect_ratio,
    )
    misc.set_random_seed(seed=args.seed, by_rank=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    return Video2WorldPipeline.from_config(
        config=config, dit_path=dit_path, device="cuda", torch_dtype=torch.bfloat16,
        load_ema_to_reg=args.load_ema, load_prompt_refiner=False,
    )


def extract_last_frame(video: Path, image: Path):
    """Retain the existing crop/256x256 feedback convention for atomic actions."""
    scan = subprocess.run(
        ["ffmpeg", "-hide_banner", "-ss", "0.5", "-t", "2", "-i", str(video),
         "-vf", "cropdetect=24:16:0", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    crops = re.findall(r"crop=\d+:\d+:\d+:\d+", scan.stderr)
    filters = f"{crops[-1]},scale=256:256" if crops else "scale=256:256"
    # Decode the end of the clip and overwrite the image through the final frame.
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-sseof", "-1", "-i", str(video),
         "-vf", filters, "-update", "1", str(image)], check=True,
    )
    if not image.is_file():
        raise RuntimeError(f"FFmpeg did not produce a feedback frame for {video}")


def generate(pipe, job: Generation, args, profile: str):
    import torch

    from examples.video2world import _DEFAULT_NEGATIVE_PROMPT, validate_input_file
    from imaginaire.utils.io import save_image_or_video, save_text_prompts

    if not validate_input_file(str(job.input_path), args.num_conditional_frames):
        raise ValueError(f"Invalid conditioning file: {job.input_path}")
    template = LIBERO_PROMPT if profile == "libero" else MULTIVIEW_PROMPT
    full_prompt = args.prompt_prefix + template.format(task=job.prompt.rstrip("."))
    negative_prompt = _DEFAULT_NEGATIVE_PROMPT if args.negative_prompt is None else args.negative_prompt
    result = pipe(
        prompt=full_prompt, negative_prompt=negative_prompt, aspect_ratio=args.aspect_ratio,
        input_path=str(job.input_path), num_conditional_frames=args.num_conditional_frames,
        guidance=args.guidance, seed=args.seed, return_prompt=True,
    )
    # Guardrail rejection returns (None, prompt), rather than a latent tensor.
    if result[0] is None:
        raise RuntimeError(f"Generation was rejected for {job.input_path}")
    video, prompt_used, latent = result
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image_or_video(video, str(job.output_path), fps=16)
    if latent is not None:
        torch.save(latent.cpu(), job.output_path.with_suffix(".pt"))
    save_text_prompts({"prompt": prompt_used, "negative_prompt": negative_prompt}, str(job.output_path.with_suffix(".txt")))
    if job.next_frame_path is not None:
        extract_last_frame(job.output_path, job.next_frame_path)


def main(profile: str, argv=None):
    args = parse_args(argv)
    sequences = plan_generations(args, profile)
    if args.dry_run:
        print(json.dumps([[{"input": str(j.input_path), "output": str(j.output_path), "prompt": j.prompt}
                           for j in sequence] for sequence in sequences], indent=2))
        return
    if not sequences:
        print("No episodes assigned to this shard.")
        return
    if args.model_size == "2B" and not args.dit_path:
        raise ValueError("2B inference requires --dit_path pointing to local world-model weights")
    if any(job.next_frame_path for sequence in sequences for job in sequence) and not shutil.which("ffmpeg"):
        raise RuntimeError("Sequential atomic-action inference requires ffmpeg on PATH")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    pipe = setup_pipeline(args)
    for sequence in sequences:
        for job in sequence:
            generate(pipe, job, args, profile)

