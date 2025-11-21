import os

# Root directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT_DIR, "code")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# Data directories
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")

# Output subdirectories for each approach
ACCELERATE_DDP_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "accelerate_ddp")
DEEPSPEED_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "deepspeed_zero")
FSDP_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "fsdp")
AXOLOTL_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "axolotl")

# CONFIGS DIR PATH
CONFIGS_DIR = os.path.join(CODE_DIR, "configs")
# TRAINING CONFIGS DIR PATH
TRAINING_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "training")
# ACCELERATE CONFIGS DIR PATH
ACCELERATE_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "accelerate")
# DEEPSPEED CONFIGS DIR PATH
DEEPSPEED_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "deepspeed")
# FSDP CONFIGS DIR PATH
FSDP_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "fsdp")
# AXOLOTL CONFIGS DIR PATH
AXOLOTL_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "axolotl")
# CONFIG FILE PATH FOR QLORA BASELINE TRAINING
QLORA_CFG_FILE_PATH = os.path.join(CODE_DIR, "qlora.yaml")
