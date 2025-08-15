"""
Configuration file for LFW Dataset Evaluation with Qwen2.5-VL
Contains environment setup, paths, and constants
"""

import os
import sys
import logging
from pathlib import Path

# Paths Configuration
SCRATCH_DIR = '/scratch/ps5218'
CACHE_DIR = f'{SCRATCH_DIR}/huggingface_cache'
PACKAGES_DIR = f'{SCRATCH_DIR}/python_packages'

# Environment Setup
def setup_environment():
    """Setup environment variables and directories"""
    # Set Hugging Face cache to scratch directory to avoid disk quota issues
    os.environ['HF_HOME'] = CACHE_DIR
    os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
    os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
    
    # Create cache directory
    cache_dir = Path(CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Add our packages to path
    sys.path.insert(0, PACKAGES_DIR)

# Model Configuration - Using correct Qwen2.5-VL model names that exist
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"  # Primary 7B model
FALLBACK_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"  # Smaller fallback model

# Package Requirements
REQUIRED_PACKAGES = [
    "qwen-vl-utils",
    "transformers>=4.40.0",
    "accelerate",
    "torch",
    "torchvision", 
    "Pillow",
    "requests",
    "scikit-learn",
    "tqdm",
    "numpy"
]

# Logging Configuration
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('lfw_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Model Generation Parameters
MODEL_GENERATION_CONFIG = {
    'max_new_tokens': 2048,
    'do_sample': False,
    'temperature': 0.1
}

# Evaluation Parameters
EVALUATION_CONFIG = {
    'default_max_pairs': 100,
    'progress_log_interval': 50,
    'intermediate_save_interval': 100
}