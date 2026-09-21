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

"""Stage 1 Action VAE training against a frozen world model."""

import os
from pathlib import Path

import torch

from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import distributed, log, misc


class ActionVAETrainer(ImaginaireTrainer):
    """Single-GPU Stage 1 trainer; only the Action VAE and video bridger are updated."""

    def train(
        self, model, dataloader_train, dataloader_val, data_action, action_model,
        *, dataset_type, output_dir, learning_rate=1e-4, init_checkpoint=None, save_iter=100,
    ):
        if distributed.get_world_size() != 1:
            raise ValueError("Stage 1 Action VAE training currently supports --nproc_per_node=1.")
        if save_iter < 1:
            raise ValueError("--action_save_iter must be positive.")
        if len(data_action) == 0:
            raise ValueError("No action episodes found. Check --action_data_root and --action_dataset.")
        if self.config.trainer.max_iter < 1:
            raise ValueError("trainer.max_iter must be positive.")
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)
        model.on_train_start(self.config.trainer.memory_format)
        model.requires_grad_(False)
        model.eval()
        action_model = action_model.float().to("cuda")
        if init_checkpoint:
            action_model.load_state_dict(torch.load(init_checkpoint, map_location="cpu", weights_only=True))
        self.data_action = data_action
        self.action_output_dir = Path(output_dir)
        self.action_output_dir.mkdir(parents=True, exist_ok=True)
        self.action_optimizer = torch.optim.AdamW(
            action_model.parameters(), lr=learning_rate, weight_decay=1e-2, betas=(0.9, 0.95)
        )
        self._action_counter = 0
        log.info("Stage 1: world model frozen; training Action VAE and video bridger.")
        while self._action_counter < self.config.trainer.max_iter:
            before_epoch = self._action_counter
            for batch in dataloader_train:
                batch = misc.to(batch, device="cuda")
                actions, negatives = self._get_action_data_batch(batch, data_type=dataset_type)
                if actions is None or actions.shape[0] != batch["video"].shape[0]:
                    raise ValueError("Video/action pairing failed. Check dataset type and video filenames.")
                with torch.no_grad():
                    output, _ = model.training_step(batch, self._action_counter)
                    latent = output["model_pred"].x0.detach()
                self._train_action_model(action_model, actions, negatives, latent)
                if self._action_counter % save_iter == 0:
                    self._save_action_weight(action_model)
                if self._action_counter >= self.config.trainer.max_iter:
                    break
            if self._action_counter == before_epoch:
                raise ValueError("The video dataloader yielded no batches.")
        if self._action_counter % save_iter:
            self._save_action_weight(action_model)
        log.success(f"Stage 1 finished after {self._action_counter} updates.")

    def _save_action_weight(self, action_model):
        destination = self.action_output_dir / f"ckpt_{self._action_counter}.pt"
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(action_model.state_dict(), temporary)
        os.replace(temporary, destination)
        log.info(f"Saved Action VAE weights to {destination}")

    def _get_action_data_batch(self, data_batch, num_negatives=16, hard_ratio=0.75, data_type="robocasa"):
        """
        Parses video paths and returns both anchor and negative action batches.

        Returns:
            anchor_actions: (B, 128, 12)
            negative_actions: (B, N_neg, 128, 12)
        """
        if data_type == "robocasa":
            video_paths = data_batch['video_name']['video_path']
            if isinstance(video_paths, str):
                video_paths = [video_paths]

            batch_anchors = []
            batch_negatives = []

            max_len = 128
            action_dim = 12

            for video_path in video_paths:
                # 1. Parse Metadata from Filename
                # Example format: PnPSinkToCounter_mg_demo_1266_aa_1.mp4
                file_name = os.path.basename(video_path)
                clean_name = file_name.replace('.mp4', '')
                parts = clean_name.split('_')

                # Default fallback values
                task_name = parts[0]
                demo_id = 0
                aa_idx = 0

                try:
                    # Locate 'demo' and 'aa' keywords dynamically in the filename parts
                    if 'demo' in parts:
                        demo_idx_loc = parts.index('demo')
                        demo_id = int(parts[demo_idx_loc + 1])

                    if 'aa' in parts:
                        aa_idx_loc = parts.index('aa')
                        aa_idx = int(parts[aa_idx_loc + 1])

                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing filename {file_name}: {e}. Using defaults.")

                # --- Anchor Processing ---
                # get_action now returns a Tensor directly, sliced by aa_idx
                anchor_tensor = self.data_action.get_action(task_name, demo_id, aa_idx)

                # Pass the tensor directly to _prepare_actions (previously it was anchor_sample['actions'])
                anchor_actions = self._prepare_actions(anchor_tensor, max_len, action_dim)
                batch_anchors.append(anchor_actions)

                # --- Negative Processing ---
                # sample_negative_actions returns a list of Tensors (fixed aa_idx=0 internally)
                neg_tensors = self.data_action.sample_negative_actions(
                    task_name, demo_id, num_negatives=num_negatives, hard_ratio=hard_ratio
                )

                sample_negatives = []
                for neg_tensor in neg_tensors:
                    # Process each negative tensor
                    neg_processed = self._prepare_actions(neg_tensor, max_len, action_dim)
                    sample_negatives.append(neg_processed)

                # Stack negatives for this specific batch item -> (N_neg, 128, 12)
                batch_negatives.append(torch.stack(sample_negatives, dim=0))

            # Final Stacking
            # anchor_tensor: (B, 128, 12)
            anchor_tensor = torch.stack(batch_anchors, dim=0).cuda()
            # negative_tensor: (B, N_neg, 128, 12)
            negative_tensor = torch.stack(batch_negatives, dim=0).cuda()

            return anchor_tensor, negative_tensor
        elif data_type == "libero":
            video_paths = data_batch['video_name']['video_path']
            if isinstance(video_paths, str):
                video_paths = [video_paths]

            batch_anchors = []
            batch_negatives = []

            max_len = 128
            action_dim = 7

            for video_path in video_paths:
                # 1. Parse Metadata from Filename
                # Example format: demo_3-put_the_bowl_on_the_plate-place_the_bowl_on_the_plate-order2.mp4
                file_name = os.path.basename(video_path)
                clean_name = file_name.replace('.mp4', '')

                # Split by hyphen '-' based on new structure
                parts = clean_name.split('-')

                # Default fallback values
                task_name = "unknown"
                demo_id = 0
                aa_idx = 0

                try:
                    # Part 0: demo_id (e.g., 'demo_3')
                    demo_part = parts[0]
                    if 'demo_' in demo_part:
                        demo_id = int(demo_part.split('_')[1])

                    # Part 1: task_name (e.g., 'put_the_bowl_on_the_plate')
                    if len(parts) > 1:
                        task_name = parts[1]

                    # Part 2: prompt (Ignored)

                    # Part 3 (Last part): aa_idx (e.g., 'order2')
                    order_part = parts[-1]
                    if 'order' in order_part:
                        # Extract number from 'order2', convert to int, then subtract 1
                        order_num = int(order_part.replace('order', ''))
                        aa_idx = order_num - 1

                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing filename {file_name}: {e}. Using defaults.")

                # --- Anchor Processing ---
                # get_action now returns a Tensor directly, sliced by aa_idx
                anchor_tensor = self.data_action.get_action(task_name, demo_id, aa_idx)
                if anchor_tensor is None:
                    return None, []

                # Pass the tensor directly to _prepare_actions
                anchor_actions = self._prepare_actions(anchor_tensor, max_len, action_dim)
                batch_anchors.append(anchor_actions)

                # --- Negative Processing ---
                # sample_negative_actions returns a list of Tensors
                neg_tensors = self.data_action.sample_negative_actions(
                    task_name, demo_id, num_negatives=num_negatives, hard_ratio=hard_ratio
                )

                sample_negatives = []
                for neg_tensor in neg_tensors:
                    neg_processed = self._prepare_actions(neg_tensor, max_len, action_dim)
                    sample_negatives.append(neg_processed)

                batch_negatives.append(torch.stack(sample_negatives, dim=0))

            # Final Stacking
            # anchor_tensor: (B, 128, 12)
            anchor_tensor = torch.stack(batch_anchors, dim=0).cuda()
            # negative_tensor: (B, N_neg, 128, 12)
            negative_tensor = torch.stack(batch_negatives, dim=0).cuda()

            return anchor_tensor, negative_tensor
        elif data_type == "franka":
            video_paths = data_batch['video_name']['video_path']
            if isinstance(video_paths, str):
                video_paths = [video_paths]

            batch_anchors = []
            batch_negatives = []

            max_len = 128
            action_dim = 7
            for video_path in video_paths:
                # 1. Parse Metadata from Filename
                # New format: task_name-demo-ID.mp4 (e.g., pick_the_cup-demo-13.mp4)
                file_name = os.path.basename(video_path)
                clean_name = file_name.replace('.mp4', '')

                # Default fallback values
                task_name = "unknown"
                demo_id = 0

                try:
                    # Split cleanly around '-demo-'
                    parts = clean_name.split('-demo-')

                    if len(parts) == 2:
                        task_name = parts[0]          # e.g., 'pick_the_cup'
                        demo_id = int(parts[1])       # e.g., 13
                    else:
                        raise ValueError("Filename does not match 'task_name-demo-ID' format.")

                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing filename {file_name}: {e}. Skipping.")
                    continue # Skip to the next video if parsing fails

                # --- Anchor Processing ---
                # Since it's single-atomic, aa_idx is strictly None.
                # The get_action method will return the entire action sequence.
                anchor_tensor = self.data_action.get_action(task_name, demo_id, aa_idx=None)

                if anchor_tensor is None:
                    return None, []

                # Pass the tensor directly to _prepare_actions
                anchor_actions = self._prepare_actions(anchor_tensor, max_len, action_dim)
                batch_anchors.append(anchor_actions)

                # --- Negative Processing ---
                # sample_negative_actions returns a list of Tensors
                neg_tensors = self.data_action.sample_negative_actions(
                    task_name, demo_id, num_negatives=num_negatives, hard_ratio=hard_ratio
                )

                sample_negatives = []
                for neg_tensor in neg_tensors:
                    neg_processed = self._prepare_actions(neg_tensor, max_len, action_dim)
                    sample_negatives.append(neg_processed)

                batch_negatives.append(torch.stack(sample_negatives, dim=0))

            # Final Stacking
            # anchor_tensor: (B, 128, 12)
            anchor_tensor = torch.stack(batch_anchors, dim=0).cuda()
            # negative_tensor: (B, N_neg, 128, 12)
            negative_tensor = torch.stack(batch_negatives, dim=0).cuda()

            return anchor_tensor, negative_tensor

        elif data_type == "simpler_env":
            video_paths = data_batch['video_name']['video_path']
            if isinstance(video_paths, str):
                video_paths = [video_paths]

            batch_anchors = []
            batch_negatives = []

            max_len = 128
            action_dim = 7

            for video_path in video_paths:
                file_name = os.path.basename(video_path)
                clean_name = file_name.replace(".mp4", "")

                try:
                    # episode_049347.mp4 -> 49347
                    episode_id = int(clean_name.split("episode_")[-1])
                except Exception as e:
                    print(f"Warning: cannot parse SimplerEnv filename {file_name}: {e}")
                    continue

                anchor_tensor = self.data_action.get_action(episode_id)
                if anchor_tensor is None:
                    return None, []

                anchor_actions = self._prepare_actions(anchor_tensor, max_len, action_dim)
                batch_anchors.append(anchor_actions)

                neg_tensors = self.data_action.sample_negative_actions(
                    episode_id,
                    num_negatives=num_negatives,
                    hard_ratio=hard_ratio,
                )

                sample_negatives = []
                for neg_tensor in neg_tensors:
                    neg_processed = self._prepare_actions(neg_tensor, max_len, action_dim)
                    sample_negatives.append(neg_processed)

                batch_negatives.append(torch.stack(sample_negatives, dim=0))

            if len(batch_anchors) == 0:
                return None, []

            anchor_tensor = torch.stack(batch_anchors, dim=0).cuda()
            negative_tensor = torch.stack(batch_negatives, dim=0).cuda()

            return anchor_tensor, negative_tensor

    def _prepare_actions(self, actions, max_len, action_dim):
        """Helper to handle tensor conversion, truncation, and padding."""
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, dtype=torch.float32)

        seq_len = actions.shape[0]
        if seq_len > max_len:
            actions = actions[:max_len, :]
        elif seq_len < max_len:
            padding = torch.zeros((max_len - seq_len, action_dim),
                                dtype=actions.dtype,
                                device=actions.device)
            actions = torch.cat([actions, padding], dim=0)
        return actions

    def _train_action_model(self, action_model, action_data_batch, negative_action_batch, latent_x0):
        action_model.train()
        self.action_optimizer.zero_grad(set_to_none=True)
        self._action_counter += 1

        # --- 1. PREPARE ACTIONS ---
        # Split [B, 200, 12] -> [B, 50, 4, 12]
        # We keep the Batch dimension separate (B, 50...) for correct alignment inside the model
        CHUNK_SIZE = 4
        ACTION_DIM = action_data_batch.shape[-1]
        B_size = action_data_batch.shape[0]
        TOTAL_CHUNKS = action_data_batch.shape[1] // CHUNK_SIZE

        action_input = action_data_batch.view(B_size, TOTAL_CHUNKS, CHUNK_SIZE, ACTION_DIM)

        # --- 2. PREPARE NEGATIVES ---
        # [B, N_neg, 200, 12] -> [B, N_neg, 50, 4, 12]
        B_size, N_neg, _, _ = negative_action_batch.shape
        neg_input = negative_action_batch.view(B_size, N_neg, TOTAL_CHUNKS, CHUNK_SIZE, ACTION_DIM)

        # --- 3. FORWARD PASS ---
        # latent_x0 is already [B, 16, 24, 60, 104]. We pass it directly.
        # The model will internally slice action_input to match the 24 video frames.

        recon_loss, contrastive_loss, mu_all, logvar_all = action_model(
            action_input,
            neg_input,
            video_latent_x0=latent_x0.float()
        )

        # --- 4. LOSS AGGREGATION ---
        # KL on everything (all 50 chunks)
        kl_loss = -0.5 * torch.sum(1 + logvar_all - mu_all.pow(2) - logvar_all.exp(), dim=-1).mean()

        total_loss = (1.0 * recon_loss) + \
                    (0.1 * contrastive_loss) + \
                    (0.0001 * kl_loss)

        if self._action_counter % 10 == 0:
            log.info("loss: {}, recon: {}, contr: {}".format(total_loss.item(), recon_loss.item(), contrastive_loss.item()))

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(action_model.parameters(), max_norm=1.0)
        self.action_optimizer.step()

        return {
            "loss": total_loss.item(),
            "recon": recon_loss.item(),
            "contrast": contrastive_loss.item()
        }

