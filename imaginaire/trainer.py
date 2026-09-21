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

import functools
import inspect
import os
import signal

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.data

from imaginaire.utils.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
    print("Megatron-core is not installed.")


from imaginaire.lazy_config import LazyConfig, instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import callback, distributed, log, misc
from imaginaire.utils.checkpointer import Checkpointer


class ImaginaireTrainer:
    """The base trainer class of Imaginaire.

    All trainers in Imaginaire should inherit ImaginaireTrainer. It contains the basic functionality for model training
    (particularly suited for large-scale training), including data parallel (DDP/FSDP), model weight average (EMA),
    mixed-precision training (fp16/bf16).

    Attributes:
        checkpointer (Checkpointer): checkpointer object to save/load model weights and optimizer states.
        training_timer (misc.Timer): Timer object to time code blocks and functions.
    """

    def __init__(self, config):
        """Constructor of the trainer.

        Args:
            config (Config): The config object for the Imaginaire codebase.
        """
        super().__init__()
        self.config = config
        # Set up the distributed computing environment.
        with misc.timer("init_distributed"):
            distributed.init()
            # Set up parallel states.
            if hasattr(config.model, "context_parallel_size"):
                if config.model_parallel.context_parallel_size > 1:
                    raise ValueError(
                        "Both config.model.context_parallel_size and config.model_parallel.context_parallel_size are set. "
                        "config.model.context_parallel_size is deprecated. Please only set config.model_parallel.context_parallel_size."
                    )
                else:
                    log.critical(
                        "Using deprecated config.model.context_parallel_size. Please use config.model_parallel.context_parallel_size instead."
                    )
                    config.model_parallel.context_parallel_size = config.model.context_parallel_size
            if USE_MEGATRON:
                if (
                    "create_gloo_process_groups"
                    in inspect.signature(parallel_state.initialize_model_parallel).parameters
                ):
                    parallel_state.initialize_model_parallel(
                        pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                        tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                        context_parallel_size=config.model_parallel.context_parallel_size,
                        create_gloo_process_groups=False,
                    )
                else:
                    parallel_state.initialize_model_parallel(
                        pipeline_model_parallel_size=config.model_parallel.pipeline_model_parallel_size,
                        tensor_model_parallel_size=config.model_parallel.tensor_model_parallel_size,
                        context_parallel_size=config.model_parallel.context_parallel_size,
                    )
                # `config.model_parallel.sequence_parallel` is a bool that indicates whether to use sequence parallelism.
                # It is not part of the original `parallel_state` API, so we need to set it manually.
                parallel_state.sequence_parallel = config.model_parallel.sequence_parallel
                if parallel_state.sequence_parallel:
                    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # Create the local job directory, save the config file, and pipe to a local log.
        if distributed.is_rank0():
            os.makedirs(config.job.path_local, exist_ok=True)
            # Save the config as .pkl for reproducibility.
            LazyConfig.save_pkl(config, f"{config.job.path_local}/config.pkl")
            # Save the config as .yaml for reading or parsing experiment hyperparameters.
            LazyConfig.save_yaml(config, f"{config.job.path_local}/config.yaml")
        dist.barrier()
        log.init_loguru_file(f"{config.job.path_local}/stdout.log")
        if distributed.is_rank0():
            # Print important environment variables and the effective config.
            log.info("Config:\n" + config.pretty_print(use_color=True))
        misc.print_environ_variables(["TORCH_HOME", "IMAGINAIRE_OUTPUT_ROOT"])
        # Set the random seed. If multi-GPU, different ranks are set with different seeds.
        misc.set_random_seed(seed=config.trainer.seed, by_rank=True)
        # Initialize cuDNN.
        torch.backends.cudnn.deterministic = config.trainer.cudnn.deterministic
        torch.backends.cudnn.benchmark = config.trainer.cudnn.benchmark
        # Floating-point precision settings.
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        # Initialize the callback functions.
        self.callbacks = callback.CallBackGroup(config=config, trainer=self)
        # Initialize the model checkpointer.
        if config.checkpoint.type is None:
            self.checkpointer = Checkpointer(config.checkpoint, config.job, callbacks=self.callbacks)
        else:
            self.checkpointer: Checkpointer = instantiate(
                config.checkpoint.type, config.checkpoint, config.job, callbacks=self.callbacks
            )
        # Initialize the timer for speed benchmarking.
        self.training_timer = misc.TrainingTimer()
        # Send a TimeoutError if a training step takes over timeout_period seconds.
        signal.signal(signal.SIGALRM, functools.partial(misc.timeout_handler, config.trainer.timeout_period))  # type: ignore

    def train(
        self,
        model: ImaginaireModel,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_val: torch.utils.data.DataLoader,
        data_action: None,
        action_model: torch.nn.Module
    ) -> None:
        """The training function.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_train (torch.utils.data.DataLoader): The training data loader.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
        """
        # Leaving this for backward compability for now, but we can think about moving this to model.on_train_start for all models.
        model = model.to("cuda", memory_format=self.config.trainer.memory_format)  # type: ignore
        model.on_train_start(self.config.trainer.memory_format)

        # Initialize for action data & model
        action_model = action_model.to("cuda")
        self.data_action = data_action
        self.action_optimizer, self.action_scheduler = self._init_action_trainer(action_model=action_model)
        self._action_counter = 0

        # Initialize the optimizer, scheduler, and grad_scaler.
        self.callbacks.on_optimizer_init_start()
        optimizer, scheduler = model.init_optimizer_scheduler(self.config.optimizer, self.config.scheduler)
        grad_scaler = torch.amp.GradScaler("cuda", **self.config.trainer.grad_scaler_args)
        self.callbacks.on_optimizer_init_end()
        # Load the model checkpoint and get the starting iteration number.
        iteration = self.checkpointer.load(model, optimizer, scheduler, grad_scaler)
        grad_accum_iter = 0
        log.critical(f"Distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        if self.config.trainer.distributed_parallelism == "ddp":
            # Create a DDP model wrapper.
            model_ddp = distributed.parallel_model_wrapper(self.config.trainer.ddp, model)
        elif self.config.trainer.distributed_parallelism == "fsdp":
            model_ddp = model
        else:
            raise ValueError(f"Unknown distributed parallelism mode: {self.config.trainer.distributed_parallelism}")
        log.info("Starting training...")
        model_ddp.eval()
        log.info("Model frozen")
        self.callbacks.on_train_start(model, iteration=iteration)
        # Initial validation.
        if self.config.trainer.run_validation and iteration == 0:
            self.validate(model, dataloader_val, iteration=iteration)
            log.info("Initial validation done.")
        _end_training = False
        with (
            maybe_enable_profiling(self.config, global_step=iteration) as torch_profiler,
            maybe_enable_memory_snapshot(self.config, global_step=iteration) as memory_profiler,
        ):
            while True:
                dataloader_train_iter = iter(dataloader_train)
                while True:
                    self.callbacks.on_before_dataloading(iteration)
                    try:
                        with self.training_timer("dataloader_train"):
                            data_batch = next(dataloader_train_iter)
                    except StopIteration:
                        break
                    finally:
                        self.callbacks.on_after_dataloading(iteration)
                    # If max_iter is reached, exit the training loop.
                    if iteration >= self.config.trainer.max_iter:
                        _end_training = True
                        break
                    # Move all tensors in the data batch to GPU device.
                    data_batch = misc.to(data_batch, device="cuda")
                    # The actual training step.
                    self.callbacks.on_training_step_start(model, data_batch, iteration=iteration)
                    self.callbacks.on_training_step_batch_start(model, data_batch, iteration=iteration)
                    if not model.training:
                        model_ddp.train()
                    assert model_ddp.training, "model_ddp is not in training mode."
                    assert model.training, "model is not in training mode."
                    # Get action data batch
                    act_data_batch, negative_action_batch = self._get_action_data_batch(data_batch, data_type="simpler_env")
                    if act_data_batch is None:
                        continue
                    # Train step forward
                    output_batch, loss, grad_accum_iter = self.training_step(
                        model_ddp,
                        optimizer,
                        scheduler,
                        grad_scaler,
                        data_batch,
                        iteration=iteration,
                        grad_accum_iter=grad_accum_iter,
                        action_data_batch=act_data_batch,
                        action_model=action_model,
                        negative_action_batch=negative_action_batch
                    )
                    self.callbacks.on_training_step_batch_end(
                        model, data_batch, output_batch, loss, iteration=iteration
                    )
                    # If the gradients are still being accumulated, continue to load the next training batch.
                    if grad_accum_iter != 0:
                        continue
                    # Do the following when an actual optimizer (update) step has been made.
                    iteration += 1
                    # Save checkpoint. Disable for now
                    # if iteration % self.config.checkpoint.save_iter == 0:
                    #     self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
                    self.callbacks.on_training_step_end(model, data_batch, output_batch, loss, iteration=iteration)
                    # Validation.
                    if self.config.trainer.run_validation and iteration % self.config.trainer.validation_iter == 0:
                        self.validate(model, dataloader_val, iteration=iteration)
                    # This iteration is successful; reset the timeout signal.
                    signal.alarm(self.config.trainer.timeout_period)
                    if torch_profiler:
                        torch_profiler.step()
                    if memory_profiler:
                        memory_profiler.step()
                if _end_training:
                    break
        log.success("Done with training.")
        if iteration % self.config.checkpoint.save_iter != 0:
            self.checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=iteration)
        self.callbacks.on_train_end(model, iteration=iteration)
        self.checkpointer.finalize()
        distributed.barrier()
        self.callbacks.on_app_end()

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
    
    def _compute_action_loss(self, recon_x, x, mu, logvar, beta=0.0001):
        """
        Standard VAE Loss: Reconstruction (MSE) + KL Divergence.
        Beta controls the trade-off between reconstruction and latent regularity.
        """
        recon_loss = F.mse_loss(recon_x, x)
        
        # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= x.size(0) * x.size(1) * x.size(2) # Normalize by total dimensions
        
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

    def _init_action_trainer(self, action_model, learning_rate=1e-4, weight_decay=1e-2):
        """
        Initializes the optimizer and scheduler for the ActionVAE + VideoBridger.
        """
        # Filter parameters to ensure we only optimize things with gradients
        trainable_params = [p for p in action_model.parameters() if p.requires_grad]
        
        # Using AdamW for better regularization with the 3D CNN components
        optimizer = optim.AdamW(
            trainable_params, 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.95) # Standard for robotics/generative models
        )
        
        # Cosine Annealing is a good default for multimodal alignment
        # It allows the contrastive loss to settle after an initial exploration phase
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=100, # Adjust based on expected total epochs
            eta_min=learning_rate * 0.01
        )
        
        return optimizer, scheduler

    def _train_action_model(self, action_model, action_data_batch, negative_action_batch, latent_x0):
        action_model.train() 
        self.action_optimizer.zero_grad()
        self._action_counter += 1

        # --- 1. PREPARE ACTIONS ---
        # Split [B, 200, 12] -> [B, 50, 4, 12]
        # We keep the Batch dimension separate (B, 50...) for correct alignment inside the model
        CHUNK_SIZE = 4
        ACTION_DIM = 7
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
            video_latent_x0=latent_x0
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

    def _save_action_weight(self, action_model):
        """
        Saves ONLY the ActionVAE state_dict using the format: 
        ./imaginaire/action_pretraining/weights/ckpt_<num_counter>.pt
        """
        # 1. Ensure only the main process (Rank 0) writes to disk
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        # 2. Define the directory and filename
        save_dir = "./imaginaire/action_pretraining/weights"
        os.makedirs(save_dir, exist_ok=True)
        
        # Use the dynamic counter for the filename
        file_name = f"ckpt_{self._action_counter}.pt"
        save_path = os.path.join(save_dir, file_name)

        # 3. Handle DDP wrapper to save clean state_dict
        # Extracts the bare module to avoid 'module.' prefix in keys
        if hasattr(action_model, "module"):
            state_dict = action_model.module.state_dict()
        else:
            state_dict = action_model.state_dict()

        # 4. Save ONLY the model state dict
        try:
            torch.save(state_dict, save_path)
            log.info(f"[Rank 0] ActionVAE state_dict successfully saved to {save_path}")
        except Exception as e:
            log.error(f"[Rank 0] Error saving weights to {save_path}: {e}")

    def training_step(
        self,
        model_ddp: torch.nn.Module | distributed.DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        grad_scaler: torch.amp.GradScaler,
        data: dict[str, torch.Tensor],
        iteration: int = 0,
        grad_accum_iter: int = 0,
        action_data_batch: torch.Tensor = None,
        action_model: torch.nn.Module = None,
        negative_action_batch: torch.Tensor = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        
        # 1. Set main model to eval mode to freeze BatchNorm/Dropout behavior
        model_ddp.eval()

        # Only let DDP sync gradient at the last iteration of the gradient accumulation window
        with distributed.ddp_sync_grad(model_ddp, grad_accum_iter == self.config.trainer.grad_accum_iter - 1):
            self.callbacks.on_before_forward(iteration=iteration)
            
            with self.training_timer("forward"):
                # 2. Run main model in no_grad to save memory and prevent weight updates
                with torch.no_grad():
                    output_batch, original_loss = model_ddp.training_step(data, iteration)
            
            # 3. Update the latent action model (This internally handles its own gradients/optimizer)
            latent_x0 = output_batch["latent_x0"].detach()
            self._train_action_model(action_model=action_model, action_data_batch=action_data_batch, negative_action_batch=negative_action_batch, latent_x0=latent_x0)
            if self._action_counter % 100 == 0:
                self._save_action_weight(action_model)
            self.callbacks.on_after_forward(iteration=iteration)
            
            # 4. Create a dummy loss that requires grad to satisfy the trainer logic
            # We use a scalar 0.0 that requires grad so .backward() doesn't fail.
            dummy_loss = torch.tensor(0.0, device=original_loss.device, requires_grad=True)
            
            self.callbacks.on_before_backward(model_ddp, dummy_loss, iteration=iteration)
            
            with self.training_timer("backward"):
                # Scale and backward the dummy loss
                loss_scaled = grad_scaler.scale(dummy_loss / self.config.trainer.grad_accum_iter)
                loss_scaled.backward()
                
                # This is safe because gradients for model_ddp will be zero or None
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.on_after_backward()
                else:
                    model_ddp.on_after_backward()
            
            self.callbacks.on_after_backward(model_ddp, iteration=iteration)

        grad_accum_iter += 1
        if grad_accum_iter == self.config.trainer.grad_accum_iter:
            with self.training_timer("optimizer_step"):
                self.callbacks.on_before_optimizer_step(
                    model_ddp, optimizer, scheduler, grad_scaler, iteration=iteration
                )
                
                # 5. This will "step" the main optimizer, but since grads are 0/None, 
                # the weights of model_ddp will not change.
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
                
                self.callbacks.on_before_zero_grad(model_ddp, optimizer, scheduler, iteration=iteration)
                if self.config.trainer.distributed_parallelism == "ddp":
                    model_ddp.module.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                else:
                    model_ddp.on_before_zero_grad(optimizer, scheduler, iteration=iteration)
                
                optimizer.zero_grad(set_to_none=True)
            grad_accum_iter = 0

        # Return the real loss for logging purposes
        return output_batch, original_loss, grad_accum_iter

    @torch.no_grad()
    def validate(self, model: ImaginaireModel, dataloader_val: torch.utils.data.DataLoader, iteration: int = 0) -> None:
        """Validate on the full validation dataset.

        Args:
            model (ImaginaireModel): The PyTorch model.
            dataloader_val (torch.utils.data.DataLoader): The validation data loader.
            iteration (int): Current iteration number.
        """
        log.info(f"Validating at iteration {iteration}...")
        self.callbacks.on_validation_start(model, dataloader_val, iteration=iteration)
        model.eval()
        # Evaluate on the full validation set.
        with model.pipe.ema_scope(context="Validation", is_cpu=False):
            for val_iter, data_batch in enumerate(dataloader_val):
                if self.config.trainer.max_val_iter is not None and val_iter >= self.config.trainer.max_val_iter:
                    break
                data_batch = misc.to(data_batch, device="cuda")
                self.callbacks.on_validation_step_start(model, data_batch, iteration=iteration)
                output_batch, loss = model.validation_step(data_batch, iteration)
                self.callbacks.on_validation_step_end(model, data_batch, output_batch, loss, iteration=iteration)
        self.callbacks.on_validation_end(model, iteration=iteration)
