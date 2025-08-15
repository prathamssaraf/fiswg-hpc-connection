#!/usr/bin/env python3
"""
Main entry point for LFW Dataset Evaluation with Qwen2.5-VL 72B
Complete LFW Dataset Evaluation with forensic facial comparison
"""

import os
import logging

# CRITICAL: Set cache directories BEFORE any other imports
# This ensures we use scratch directory instead of home directory (which has disk quota issues)
SCRATCH_CACHE = '/scratch/ps5218/huggingface_cache'
SCRATCH_DATA = '/scratch/ps5218/scikit_learn_data'
SCRATCH_TRITON = '/scratch/ps5218/triton_cache'
SCRATCH_TORCH_KERNELS = '/scratch/ps5218/torch_kernels'

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

# Tokenizers parallelism (suppress warnings)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
    logger.info(f"HF_HOME: {os.environ.get('HF_HOME')}")
    logger.info(f"SCIKIT_LEARN_DATA: {os.environ.get('SCIKIT_LEARN_DATA')}")
    logger.info(f"TRITON_CACHE_DIR: {os.environ.get('TRITON_CACHE_DIR')}")
    logger.info(f"TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM')}")
    logger.info("Note: This requires transformers installed from source")
    
    try:
        # Initialize evaluator
        evaluator = QwenVLEvaluator()
        
        # Setup (install dependencies and load model)
        evaluator.setup()
        
        # Run evaluation (start with subset for testing)
        # Note: Due to detailed FISWG analysis, each pair will take longer to process
        # For full evaluation, remove max_pairs parameter
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