# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import psutil
import torch

from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.trainer import ImaginaireTrainer
from imaginaire.utils import distributed, log

# ---------------- NVML is optional (NVIDIA-only) ----------------
try:
    import pynvml  # type: ignore

    _HAS_PYNVML = True
except Exception:
    pynvml = None  # type: ignore
    _HAS_PYNVML = False

IS_ROCM = getattr(torch.version, "hip", None) is not None
IS_CUDA = getattr(torch.version, "cuda", None) is not None and not IS_ROCM


def log_prof_data(
    data_list: list[dict[str, Any]],
    iteration: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Create a table to log data with rank information
    columns = ["iteration", "rank"] + list(data_list[0].keys())  # noqa: RUF005
    data = []

    # Initialize dictionaries to store min and max values for each metric
    min_values = {key: float("inf") for key in columns[2:]}
    max_values = {key: float("-inf") for key in columns[2:]}
    sum_values = {key: 0.0 for key in columns[2:]}

    count = 0

    for _rank, prof_data in enumerate(data_list):
        row = [iteration, _rank] + [prof_data[key] for key in columns[2:]]
        data.append(row)
        count += 1

        # Update min, max, and sum values
        for key in columns[2:]:
            min_values[key] = min(min_values[key], prof_data[key])
            max_values[key] = max(max_values[key], prof_data[key])
            sum_values[key] += prof_data[key]

    # Calculate average values
    avg_values = {key: sum_values[key] / count for key in columns[2:]}

    df = pd.DataFrame(data, columns=columns)
    summary_df = pd.DataFrame({"Avg": avg_values, "Max": max_values, "Min": min_values})
    return df, summary_df


class DeviceMonitor(EveryN):
    """
    A callback to monitor device (CPU/GPU) usage and log it at regular intervals.

    Args:
        every_n (int, optional): The frequency at which the callback is invoked. Defaults to 200.
        step_size (int, optional): The step size for the callback. Defaults to 1.
        log_memory_detail (bool, optional): Whether to log the memory detail. Defaults to True.
    """

    def __init__(
        self,
        every_n: int = 200,
        step_size: int = 1,
        log_memory_detail: bool = True,
    ):
        super().__init__(every_n=every_n, step_size=step_size)
        self.name = self.__class__.__name__
        self.log_memory_detail = log_memory_detail

        self.world_size: int | None = None
        self.rank: int | None = None

        # NVML related state
        self.handle = None
        self.nvml_enabled = False

    def _init_nvml(self, local_rank: int) -> None:
        """Try to initialize NVML for the given local_rank. Safe on ROCm / non-NVIDIA."""
        # Only attempt on NVIDIA + NVML
        if not (_HAS_PYNVML and IS_CUDA):
            log.info(f"{self.name}: NVML not available or non-CUDA backend; NVML monitoring disabled.")
            self.nvml_enabled = False
            self.handle = None
            return

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            self.nvml_enabled = True
        except Exception as e:
            log.warning(f"{self.name}: failed to initialize NVML on local_rank {local_rank}; disabling NVML. Error: {e}")
            self.handle = None
            self.nvml_enabled = False

    def on_train_start(self, model, iteration: int = 0) -> None:  # noqa: ARG002
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.world_size = distributed.get_world_size()
        self.rank = distributed.get_rank()

        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if self.rank == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"{self.name} callback: local_dir: {self.local_dir}")

        local_rank = int(os.getenv("LOCAL_RANK", 0))
        self._init_nvml(local_rank)

    def _safe_cuda_metric(self, fn_name: str, default: float = 0.0) -> float:
        """Call a torch.cuda helper if it exists; otherwise return default."""
        if not torch.cuda.is_available():
            return default
        fn = getattr(torch.cuda, fn_name, None)
        if fn is None:
            return default
        try:
            return float(fn())
        except Exception:
            return default

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        # ---------------- CPU memory ----------------
        cur_process = psutil.Process(os.getpid())
        cpu_memory_usage = sum(
            p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True)
        )  # noqa: RUF005
        cpu_mem_gb = cpu_memory_usage / (1024**3)

        # ---------------- GPU stats via torch.cuda ----------------
        peak_gpu_mem_gb = 0.0
        peak_gpu_mem_reserved_gb = 0.0
        temp = 0.0
        power = 0.0
        util = 0.0
        clock = 0.0

        if torch.cuda.is_available():
            peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
            peak_gpu_mem_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)

            # These helpers are not standard in vanilla PyTorch; guard them carefully
            temp = self._safe_cuda_metric("temperature", 0.0)
            util = self._safe_cuda_metric("utilization", 0.0)
            clock = self._safe_cuda_metric("clock_rate", 0.0)
            power = self._safe_cuda_metric("power_draw", 0.0)

        # ---------------- GPU stats via NVML (optional) ----------------
        nvml_used_gpu_mem_gb = 0.0
        nvml_free_gpu_mem_gb = 0.0

        if self.nvml_enabled and self.handle is not None:
            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                nvml_used_gpu_mem_gb = memory_info.used / (1024**3)
                nvml_free_gpu_mem_gb = memory_info.free / (1024**3)
            except Exception as e:
                log.warning(f"{self.name}: NVML memory query failed, disabling NVML. Error: {e}")
                self.nvml_enabled = False
                self.handle = None

        # ---------------- Pack profiling data ----------------
        prof_data = {
            "cpu_mem_gb": cpu_mem_gb,
            "peak_gpu_mem_gb": peak_gpu_mem_gb,
            "peak_gpu_mem_reserved_gb": peak_gpu_mem_reserved_gb,
            "nvml_used_gpu_mem_gb": nvml_used_gpu_mem_gb,
            "nvml_free_gpu_mem_gb": nvml_free_gpu_mem_gb,
            "temp": temp,
            "power": power,
            "util": util,
            "clock": clock,
        }

        # ---------------- All-gather across ranks ----------------
        if self.world_size is None:
            self.world_size = distributed.get_world_size()

        data_list = [prof_data] * self.world_size

        if self.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_gather_object(data_list, prof_data)
            torch.distributed.barrier()

        df, summary_df = log_prof_data(data_list, iteration)  # noqa: F841

        if self.rank is None:
            self.rank = distributed.get_rank()

        if self.rank == 0:
            log.info(f"{self.name} Stats:\n{summary_df.to_string()}")
            if self.log_memory_detail and torch.cuda.is_available():
                # Just force PyTorch to compute memory stats; you can extend this to write them out if you like.
                _ = torch.cuda.memory_stats()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
