srun --jobid 39978 --overlap --pty bash


conda activate tuan.wan.worldmodel.pytorch271
export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_ENABLE_LOGGING_CMD=0
mkdir -p ~/miopen_db_cosmos
mkdir -p ~/miopen_cache_cosmos

export MIOPEN_USER_DB_PATH=$HOME/miopen_db_cosmos
export MIOPEN_SYSTEM_DB_PATH=$HOME/miopen_db_cosmos        
export MIOPEN_CUSTOM_CACHE_DIR=$HOME/miopen_cache_cosmos

export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE