"""
Utility functions for LFW evaluation
Contains helper functions for saving results, calculating metrics, etc.
"""

import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def save_results(results: List[Dict], filename: str):
    """Save results to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save results to {filename}: {e}")

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics from results"""
    if not results:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'correct_predictions': 0
        }
    
    # Count correct predictions
    correct_predictions = sum(1 for r in results if r['correct'])
    total_predictions = len(results)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Calculate confusion matrix values
    true_positives = sum(1 for r in results if r['ground_truth'] and r['prediction'] and r['correct'])
    false_positives = sum(1 for r in results if not r['ground_truth'] and r['prediction'])
    false_negatives = sum(1 for r in results if r['ground_truth'] and not r['prediction'])
    true_negatives = sum(1 for r in results if not r['ground_truth'] and not r['prediction'] and r['correct'])
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'correct_predictions': correct_predictions
    }

def print_evaluation_summary(results: Dict):
    """Print a formatted summary of evaluation results"""
    print("\n" + "="*80)
    print("LFW EVALUATION RESULTS")
    print("="*80)
    print(f"Model: {results['model_name']}")
    print(f"Total pairs evaluated: {results['total_pairs']}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Evaluation time: {results['evaluation_time']:.1f} seconds")
    print("="*80)

def create_timestamp_filename(base_name: str, extension: str = "json") -> str:
    """Create a filename with timestamp"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"

def load_results(filename: str) -> Dict:
    """Load results from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results from {filename}: {e}")
        raise

def validate_results(results: List[Dict]) -> bool:
    """Validate that results have the required structure"""
    required_keys = ['pair_id', 'ground_truth', 'prediction', 'correct']
    
    if not results:
        logger.warning("Results list is empty")
        return False
    
    for i, result in enumerate(results):
        if not isinstance(result, dict):
            logger.error(f"Result {i} is not a dictionary")
            return False
        
        for key in required_keys:
            if key not in result:
                logger.error(f"Result {i} missing required key: {key}")
                return False
    
    logger.info(f"✓ Validated {len(results)} results")
    return True

def get_error_analysis(results: List[Dict]) -> Dict:
    """Analyze errors in the evaluation results"""
    if not results:
        return {}
    
    false_positives = [r for r in results if not r['ground_truth'] and r['prediction']]
    false_negatives = [r for r in results if r['ground_truth'] and not r['prediction']]
    
    return {
        'false_positive_count': len(false_positives),
        'false_negative_count': len(false_negatives),
        'false_positive_rate': len(false_positives) / len(results),
        'false_negative_rate': len(false_negatives) / len(results),
        'false_positive_pairs': [r['pair_id'] for r in false_positives[:10]],  # First 10
        'false_negative_pairs': [r['pair_id'] for r in false_negatives[:10]]   # First 10
    }