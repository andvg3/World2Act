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

import argparse
import importlib
import os

from loguru import logger as logging

from imaginaire.config import Config, pretty_print_overrides
from imaginaire.lazy_config import instantiate
from imaginaire.lazy_config.lazy import LazyConfig
from imaginaire.utils import distributed
from imaginaire.utils.config_helper import get_config_module, override

@logging.catch(reraise=True)
def launch(config: Config, args: argparse.Namespace) -> None:
    # Need to initialize the distributed environment before calling config.validate() because it tries to synchronize
    # a buffer across ranks. If you don't do this, then you end up allocating a bunch of buffers on rank 0, and also that
    # check doesn't actually do anything.
    distributed.init()

    # Check that the config is valid
    config.validate()
    # Freeze the config so developers don't change it during training.
    config.freeze()  # type: ignore
    if args.training_mode == "action_vae":
        from imaginaire.action_pretraining.trainer import ActionVAETrainer

        trainer = ActionVAETrainer(config)
    else:
        trainer = config.trainer.type(config)
    # Create the model
    model = instantiate(config.model)
    # Create the dataloaders.
    dataloader_train = instantiate(config.dataloader_train)
    dataloader_val = instantiate(config.dataloader_val)
    if args.training_mode == "action_vae":
        from imaginaire.action_pretraining.action_dataloader import (
            FrankaArmActionDataset, LIBEROActionDataset, RoboCasaActionDataset, SimplerEnvActionDataset,
        )
        from imaginaire.action_pretraining.action_vae import ActionVAE

        datasets = {
            "robocasa": (RoboCasaActionDataset, 12),
            "libero": (LIBEROActionDataset, 7),
            "franka": (FrankaArmActionDataset, 7),
            "simpler_env": (SimplerEnvActionDataset, 7),
        }
        dataset_class, action_dim = datasets[args.action_dataset]
        kwargs = {"root_dir": args.action_data_root}
        if args.action_dataset == "robocasa":
            kwargs["metadata_path"] = args.action_metadata
        data_action = dataset_class(**kwargs)
        trainer.train(
            model, dataloader_train, dataloader_val, data_action, ActionVAE(action_dim=action_dim),
            dataset_type=args.action_dataset, output_dir=args.action_output_dir,
            learning_rate=args.action_learning_rate, init_checkpoint=args.action_init_checkpoint,
            save_iter=args.action_save_iter,
        )
    else:
        trainer.train(model, dataloader_train, dataloader_val)



if __name__ == "__main__":
    # Usage: torchrun --nproc_per_node=1 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_training_2b_groot_gr1_480

    # Get the config file from the input arguments.
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", help="Path to the config file", required=True)
    parser.add_argument("--training_mode", choices=["wm", "action_vae"], default="wm",
                        help="wm: fine-tune world model; action_vae: Stage 1 with a frozen world model")
    parser.add_argument("--action_dataset", choices=["robocasa", "libero", "franka", "simpler_env"])
    parser.add_argument("--action_data_root", help="Local action dataset; required for Stage 1")
    parser.add_argument("--action_metadata", help="RoboCasa video_metadata.json; defaults to action_data_root/video_metadata.json")
    parser.add_argument("--action_output_dir", default="checkpoints/action_vae")
    parser.add_argument("--action_learning_rate", type=float, default=1e-4)
    parser.add_argument("--action_save_iter", type=int, default=100)
    parser.add_argument("--action_init_checkpoint", help="Warm-start Action VAE state dict; optimizer and step counter start fresh")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Do a dry run without training. Useful for debugging the config.",
    )
    args = parser.parse_args()
    if args.training_mode == "action_vae":
        if not args.action_dataset or not args.action_data_root:
            parser.error("--training_mode action_vae requires --action_dataset and --action_data_root")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            parser.error("Stage 1 currently supports --nproc_per_node=1")
        if args.action_save_iter < 1 or args.action_learning_rate <= 0:
            parser.error("Action VAE learning rate and save interval must be positive")
    config_module = get_config_module(args.config)
    config = importlib.import_module(config_module).make_config()
    config = override(config, args.opts)
    if args.dryrun:
        logging.info(
            "Config:\n" + config.pretty_print(use_color=True) + "\n" + pretty_print_overrides(args.opts, use_color=True)
        )
        os.makedirs(config.job.path_local, exist_ok=True)
        LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        print(f"{config.job.path_local}/config.yaml")
    else:
        # Launch the training job.
        launch(config, args)
