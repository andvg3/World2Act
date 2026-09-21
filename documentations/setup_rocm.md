# AMD / ROCm setup

Use a Linux AMD GPU machine with a compatible ROCm driver, ROCm-enabled PyTorch, and the compiled extensions used by Cosmos Predict2. Python 3.10 is the original project baseline. The exact original environment lock is not included, so the instructions below distinguish GPU-stack setup from installing this source tree.

## Prepare the GPU environment

Follow the [AMD PyTorch installation instructions](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html) for your GPU, OS, and ROCm version. Keep PyTorch and torchvision matched. The code imports `torch.distributed.fsdp.fully_shard`, so the installed PyTorch must expose that API.

This implementation also imports Transformer Engine, Megatron Core, FlashAttention, and Apex's compiled multi-tensor optimizer. Use ROCm-compatible builds matched to the same PyTorch/ROCm stack. See [ROCm Transformer Engine](https://github.com/ROCm/TransformerEngine) and the [FlashAttention AMD backend instructions](https://github.com/Dao-AILab/flash-attention#amd-rocm-support). Setting an environment variable alone does not install these extensions.

The upstream `cu126` extra, `uv.lock`, Dockerfile, and [upstream setup guide](setup.md) describe NVIDIA/CUDA environments. Do not use `uv sync --extra cu126` to prepare AMD dependencies or replace an existing working ROCm environment.

## Install this source tree

In your prepared environment:

```bash
git clone https://github.com/andvg3/World2Act.git
cd World2Act
python -m pip install -e . --no-deps
source setup_rocm.sh
```

`--no-deps` assumes that the runtime packages listed in `pyproject.toml` are already installed. In particular, retain the Triton supplied by your ROCm stack rather than blindly installing the upstream CUDA-oriented pin. Training also imports `wandb` and `psutil`; Stage 1 action datasets need `h5py` and `pyarrow`:

```bash
python -m pip install wandb psutil h5py pyarrow
python -m scripts.test_environment --training --stage1 --require_rocm
```

The check must pass before running training. If an import fails, complete or repair the matching runtime environment first; this source installation is not a replacement for a ROCm dependency build. NATTEN is only needed for the upstream sparse-attention variants and is not required by the released default preset.

`setup_rocm.sh` sets writable per-user MIOpen cache paths and the AMD Triton attention flag. It does not activate another user's environment, attach to a cluster job, or override the system MIOpen database.

For the sequential RoboCasa/LIBERO inference workflows, install `ffmpeg` with your system package manager and check `ffmpeg -version`.

## Verify the device

```bash
python -c 'import torch; print("PyTorch:", torch.__version__); print("ROCm:", torch.version.hip); print("GPU:", torch.cuda.get_device_name(0))'
```

PyTorch's ROCm backend still uses the `torch.cuda` API and the `cuda` device string in application code. A non-empty `torch.version.hip` and an accessible GPU are required here.

Proceed to [training](world2act_training.md) or [inference](world2act_inference.md). The GPU training and inference runs have not been reproduced in the CPU-only release-cleanup environment.
