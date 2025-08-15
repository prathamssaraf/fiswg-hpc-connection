"""
Main Evaluator Class for LFW Dataset Evaluation
Orchestrates the complete evaluation process using Qwen2.5-VL
"""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
from tqdm import tqdm

from model_manager import ModelManager
from data_loader import LFWDataLoader
from image_analyzer import ImageAnalyzer
from utils import save_results, calculate_metrics
from config import EVALUATION_CONFIG, DEFAULT_MODEL_NAME

logger = logging.getLogger(__name__)

class QwenVLEvaluator:
    """Main evaluator class that orchestrates the complete LFW evaluation process"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_manager = ModelManager(model_name)
        self.data_loader = LFWDataLoader()
        self.image_analyzer = None
        
    def setup(self, skip_package_install: bool = False):
        """Setup the evaluator by installing packages and loading models"""
        logger.info("Setting up evaluator...")
        
        # Skip package installation if requested (e.g., when using virtual environment)
        if skip_package_install:
            logger.info("Skipping package installation (using existing environment)")
        else:
            # Install required packages
            self.model_manager.install_packages()
        
        # Load model
        self.model_manager.load_model()
        
        # Initialize image analyzer
        self.image_analyzer = ImageAnalyzer(self.model_manager)
        
        logger.info("✓ Evaluator setup complete")

    def evaluate_lfw(self, max_pairs: Optional[int] = None) -> Dict:
        """Run complete evaluation on LFW dataset"""
        logger.info("Starting LFW evaluation...")
        
        if not self.model_manager.is_loaded():
            raise RuntimeError("Model not loaded. Call setup() first.")
        
        # Load dataset
        pairs, targets = self.data_loader.get_subset(max_pairs or EVALUATION_CONFIG['default_max_pairs'])
        
        if max_pairs:
            logger.info(f"Limited evaluation to {max_pairs} pairs")
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        start_time = time.time()
        
        # Process each pair
        for i, (pair, target) in enumerate(tqdm(zip(pairs, targets), total=len(pairs), desc="Evaluating pairs")):
            img1, img2 = pair[0], pair[1]
            ground_truth = bool(target)  # True = same person, False = different
            
            try:
                # Get model prediction
                output_text, prediction = self.image_analyzer.analyze_image_pair(img1, img2)
                
                # Check if correct
                is_correct = prediction == ground_truth
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                # Store result
                result = {
                    'pair_id': i,
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'correct': is_correct,
                    'model_output': output_text[:2000]  # Keep detailed forensic analysis
                }
                results.append(result)
                
                # Log progress periodically
                if (i + 1) % EVALUATION_CONFIG['progress_log_interval'] == 0:
                    current_accuracy = correct_predictions / total_predictions
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed * 60  # pairs per minute
                    logger.info(f"Progress: {i+1}/{len(pairs)} pairs, Accuracy: {current_accuracy:.3f}, Rate: {rate:.1f} pairs/min")
                
                # Save intermediate results
                if (i + 1) % EVALUATION_CONFIG['intermediate_save_interval'] == 0:
                    save_results(results, f"intermediate_results_{i+1}.json")
                
            except Exception as e:
                logger.error(f"Error processing pair {i}: {e}")
                continue
        
        # Calculate final metrics
        evaluation_results = self._compile_results(results, start_time)
        
        self._log_final_results(evaluation_results)
        
        return evaluation_results

    def _compile_results(self, results: List[Dict], start_time: float) -> Dict:
        """Compile final evaluation results with metrics"""
        metrics = calculate_metrics(results)
        
        evaluation_results = {
            'model_name': self.model_manager.model_name,
            'total_pairs': len(results),
            'evaluation_time': time.time() - start_time,
            'results': results,
            **metrics
        }
        
        return evaluation_results

    def _log_final_results(self, evaluation_results: Dict):
        """Log final evaluation results"""
        logger.info(f"✓ Evaluation complete!")
        logger.info(f"Final Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"Precision: {evaluation_results['precision']:.4f}")
        logger.info(f"Recall: {evaluation_results['recall']:.4f}")
        logger.info(f"F1 Score: {evaluation_results['f1_score']:.4f}")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return self.model_manager.get_model_info()

    def get_dataset_info(self) -> Dict:
        """Get information about the loaded dataset"""
        if self.data_loader.loaded:
            return self.data_loader.get_dataset_stats()
        else:
            return {"status": "Dataset not loaded"}