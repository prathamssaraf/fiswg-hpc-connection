#!/usr/bin/env python3
"""
Main entry point for LFW Dataset Evaluation with Qwen2.5-VL 72B
Designed to work with virtual environment setup
"""

import os
import logging

# CRITICAL: Set cache directories BEFORE any other imports
SCRATCH_DIR = '/scratch/ps5218'
SCRATCH_CACHE = f'{SCRATCH_DIR}/huggingface_cache'
SCRATCH_DATA = f'{SCRATCH_DIR}/scikit_learn_data'
SCRATCH_TRITON = f'{SCRATCH_DIR}/triton_cache'
SCRATCH_TORCH_KERNELS = f'{SCRATCH_DIR}/torch_kernels'
SCRATCH_PIP_CACHE = f'{SCRATCH_DIR}/pip_cache'

# Hugging Face cache directories
os.environ['HF_HOME'] = SCRATCH_CACHE
os.environ['HF_HUB_CACHE'] = SCRATCH_CACHE
os.environ['TRANSFORMERS_CACHE'] = SCRATCH_CACHE
os.environ['HF_DATASETS_CACHE'] = SCRATCH_CACHE
os.environ['TORCH_HOME'] = SCRATCH_CACHE

# Scikit-learn data directory
os.environ['SCIKIT_LEARN_DATA'] = SCRATCH_DATA

# Triton cache directory (for GPU kernels)
os.environ['TRITON_CACHE_DIR'] = SCRATCH_TRITON

# Torch kernel cache directory
os.environ['TORCH_KERNEL_CACHE_PATH'] = SCRATCH_TORCH_KERNELS

# Pip cache directory (avoid disk quota in home)
os.environ['PIP_CACHE_DIR'] = SCRATCH_PIP_CACHE

# Tokenizers parallelism (suppress warnings)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Triton compilation settings (prevent JIT compilation issues)
os.environ['TRITON_DISABLE_JIT'] = '1'
os.environ['TRITON_DISABLE_LINE_INFO'] = '1'
os.environ['TRITON_DISABLE_CUDA_GRAPHS'] = '1'
os.environ['TRITON_DISABLE_CACHE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Force use of PyTorch's SDPA instead of Flash Attention or Triton
os.environ['PYTORCH_DISABLE_FLASH_ATTENTION'] = '1'

# Simple imports - no complex module cache clearing needed in venv
from config import setup_environment, setup_logging
from evaluator import QwenVLEvaluator
from utils import save_results, create_timestamp_filename, print_evaluation_summary

def main():
    """Main execution function"""
    # Setup environment and logging
    setup_environment()
    logger = setup_logging()
    
    logger.info("Starting LFW evaluation with Qwen2.5-VL-72B-Instruct-AWQ (quantized)")
    logger.info(f"Using HF cache directory: {SCRATCH_CACHE}")
    logger.info(f"Using sklearn data directory: {SCRATCH_DATA}")
    logger.info(f"Using Triton cache directory: {SCRATCH_TRITON}")
    logger.info(f"Using Torch kernels directory: {SCRATCH_TORCH_KERNELS}")
    logger.info(f"Using Pip cache directory: {SCRATCH_PIP_CACHE}")
    
    # Check if running in virtual environment
    if 'VIRTUAL_ENV' in os.environ:
        logger.info(f"Running in virtual environment: {os.environ['VIRTUAL_ENV']}")
    else:
        logger.warning("Not running in virtual environment - this may cause package conflicts")
    
    try:
        # Initialize evaluator
        evaluator = QwenVLEvaluator()
        
        # Setup (install dependencies and load model)
        evaluator.setup()
        
        # Run evaluation (start with subset for testing)
        results = evaluator.evaluate_lfw(max_pairs=100)  # Start with 100 pairs for testing
        
        # Save final results
        final_filename = create_timestamp_filename("lfw_qwen_evaluation")
        save_results(results, final_filename)
        
        # Print summary
        print_evaluation_summary(results)
        print(f"Results saved to: {final_filename}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()