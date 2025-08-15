#!/bin/bash
"""
Simple script to run the evaluation with virtual environment
"""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐍 Running LFW evaluation with virtual environment${NC}"

# Check if virtual environment exists
VENV_DIR="/scratch/ps5218/qwen_venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}❌ Virtual environment not found at $VENV_DIR${NC}"
    echo -e "${BLUE}Run: python3 setup_venv.py${NC}"
    exit 1
fi

# Set environment variables
export QWEN_VENV_DIR="$VENV_DIR"
export PATH="$QWEN_VENV_DIR/bin:$PATH"
export VIRTUAL_ENV="$QWEN_VENV_DIR"

# Set scratch directories
export HF_HOME="/scratch/ps5218/huggingface_cache"
export HF_HUB_CACHE="/scratch/ps5218/huggingface_cache"
export TRANSFORMERS_CACHE="/scratch/ps5218/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/ps5218/huggingface_cache"
export TORCH_HOME="/scratch/ps5218/huggingface_cache"
export SCIKIT_LEARN_DATA="/scratch/ps5218/scikit_learn_data"
export TRITON_CACHE_DIR="/scratch/ps5218/triton_cache"
export TORCH_KERNEL_CACHE_PATH="/scratch/ps5218/torch_kernels"
export PIP_CACHE_DIR="/scratch/ps5218/pip_cache"
export TOKENIZERS_PARALLELISM="false"
export TRITON_DISABLE_JIT="1"
export TRITON_DISABLE_CACHE="1"
export PYTORCH_DISABLE_FLASH_ATTENTION="1"

echo -e "${GREEN}✓ Activated virtual environment${NC}"
echo "Python: $(which python)"
echo "Pip: $(which pip)"

# Run the evaluation
echo -e "${BLUE}🚀 Starting evaluation...${NC}"
python main_venv.py