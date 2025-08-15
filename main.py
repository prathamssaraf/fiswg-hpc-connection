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
os.environ['HF_HOME'] = SCRATCH_CACHE
os.environ['HF_HUB_CACHE'] = SCRATCH_CACHE
os.environ['TRANSFORMERS_CACHE'] = SCRATCH_CACHE
os.environ['HF_DATASETS_CACHE'] = SCRATCH_CACHE
os.environ['TORCH_HOME'] = SCRATCH_CACHE

from config import setup_environment, setup_logging
from evaluator import QwenVLEvaluator
from utils import save_results, create_timestamp_filename, print_evaluation_summary

def main():
    """Main execution function"""
    # Setup environment and logging
    setup_environment()
    logger = setup_logging()
    
    logger.info("Starting LFW evaluation with Qwen2.5-VL")
    logger.info(f"Using cache directory: {SCRATCH_CACHE}")
    logger.info(f"HF_HOME environment variable: {os.environ.get('HF_HOME')}")
    
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