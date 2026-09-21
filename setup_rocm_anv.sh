export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_ENABLE_LOGGING_CMD=0
mkdir -p ~/miopen_db_cosmos_an
mkdir -p ~/miopen_cache_cosmos_an

export MIOPEN_USER_DB_PATH=$HOME/miopen_db_cosmos_an/
export MIOPEN_SYSTEM_DB_PATH=$HOME/miopen_db_cosmos_an/
export MIOPEN_CUSTOM_CACHE_DIR=$HOME/miopen_cache_cosmos_an/

export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE