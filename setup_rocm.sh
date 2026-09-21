#!/usr/bin/env bash
# Source this file after activating your ROCm-compatible Python environment.
# Usage: source setup_rocm.sh
export MIOPEN_ENABLE_LOGGING="${MIOPEN_ENABLE_LOGGING:-0}"
export MIOPEN_ENABLE_LOGGING_CMD="${MIOPEN_ENABLE_LOGGING_CMD:-0}"
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-${XDG_CACHE_HOME:-$HOME/.cache}/world2act/miopen/db}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/world2act/miopen/cache}"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
export FLASH_ATTENTION_TRITON_AMD_ENABLE="${FLASH_ATTENTION_TRITON_AMD_ENABLE:-TRUE}"
