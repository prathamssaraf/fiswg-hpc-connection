"""
Data Loader for LFW Dataset
Handles loading and processing of the Labeled Faces in the Wild dataset
"""

import os
import logging
import numpy as np
from typing import Tuple, List
from sklearn.datasets import fetch_lfw_pairs

logger = logging.getLogger(__name__)

class LFWDataLoader:
    """Handles loading and processing of LFW dataset"""
    
    def __init__(self):
        self.pairs = None
        self.targets = None
        self.loaded = False

    def load_dataset(self) -> Tuple[List, List]:
        """Load LFW pairs dataset"""
        logger.info("Loading LFW dataset...")
        
        try:
            # Use scratch directory for LFW dataset to avoid disk quota issues
            data_home = os.environ.get('SCIKIT_LEARN_DATA', '/scratch/ps5218/scikit_learn_data')
            logger.info(f"Using sklearn data directory: {data_home}")
            
            # Load LFW pairs with explicit data_home
            lfw_pairs = fetch_lfw_pairs(
                subset='test', 
                funneled=True, 
                resize=0.5,
                data_home=data_home
            )
            
            self.pairs = lfw_pairs.pairs
            self.targets = lfw_pairs.target  # 1 = same person, 0 = different person
            self.loaded = True
            
            logger.info(f"✓ Loaded {len(self.pairs)} image pairs")
            logger.info(f"Same person pairs: {np.sum(self.targets)}")
            logger.info(f"Different person pairs: {len(self.targets) - np.sum(self.targets)}")
            
            return self.pairs, self.targets
            
        except Exception as e:
            logger.error(f"Failed to load LFW dataset: {e}")
            raise

    def get_subset(self, max_pairs: int) -> Tuple[List, List]:
        """Get a subset of the dataset"""
        if not self.loaded:
            self.load_dataset()
            
        if max_pairs and max_pairs < len(self.pairs):
            subset_pairs = self.pairs[:max_pairs]
            subset_targets = self.targets[:max_pairs]
            logger.info(f"Created subset with {max_pairs} pairs")
            return subset_pairs, subset_targets
        else:
            return self.pairs, self.targets

    def get_pair_info(self, index: int) -> dict:
        """Get information about a specific pair"""
        if not self.loaded:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        if index >= len(self.pairs):
            raise IndexError(f"Index {index} out of range. Dataset has {len(self.pairs)} pairs.")
            
        pair = self.pairs[index]
        target = self.targets[index]
        
        return {
            'index': index,
            'pair': pair,
            'target': bool(target),
            'same_person': bool(target),
            'image1_shape': pair[0].shape,
            'image2_shape': pair[1].shape
        }

    def get_dataset_stats(self) -> dict:
        """Get statistics about the loaded dataset"""
        if not self.loaded:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        same_person_count = np.sum(self.targets)
        different_person_count = len(self.targets) - same_person_count
        
        return {
            'total_pairs': len(self.pairs),
            'same_person_pairs': int(same_person_count),
            'different_person_pairs': int(different_person_count),
            'same_person_percentage': float(same_person_count / len(self.targets) * 100),
            'different_person_percentage': float(different_person_count / len(self.targets) * 100),
            'image_shape': self.pairs[0][0].shape if len(self.pairs) > 0 else None
        }