"""
Model Manager for Qwen2.5-VL
Handles model loading, installation, and management
"""

import sys
import subprocess
import logging
import torch
import importlib.util
from transformers import AutoProcessor

# Import Qwen2.5VL model class (correct class name for 2.5)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    # Fallback for older transformers versions
    try:
        from qwen_vl_utils import Qwen2_5_VLForConditionalGeneration
    except ImportError:
        Qwen2_5_VLForConditionalGeneration = None
from config import (
    DEFAULT_MODEL_NAME, FALLBACK_MODEL_NAME, REQUIRED_PACKAGES, 
    PACKAGES_DIR, CACHE_DIR
)

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages Qwen2.5-VL model loading and operations"""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.tokenizer = None

    def _check_package_installed(self, package_name: str) -> bool:
        """Check if a package is already installed"""
        spec = importlib.util.find_spec(package_name.split('>=')[0].split('==')[0])
        return spec is not None
    
    def install_packages(self):
        """Install required packages for Qwen2.5-VL (only if not already installed)"""
        packages_to_install = []
        
        for package in REQUIRED_PACKAGES:
            package_name = package.split('>=')[0].split('==')[0]
            if not self._check_package_installed(package_name):
                packages_to_install.append(package)
            else:
                logger.info(f"✓ {package_name} already installed")
        
        if packages_to_install:
            logger.info(f"Installing {len(packages_to_install)} missing packages...")
            for package in packages_to_install:
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "--target", 
                        PACKAGES_DIR, package
                    ], check=True)
                    logger.info(f"✓ Installed {package}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {package}: {e}")
        else:
            logger.info("All required packages already installed")

    def check_gpu_memory(self):
        """Check available GPU memory and return info"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Available GPU memory: {gpu_memory:.1f} GB")
            return gpu_memory
        else:
            logger.info("No GPU available, using CPU")
            return 0

    def load_model(self):
        """Load Qwen2.5-VL model with optimization for large model"""
        logger.info(f"Loading {self.model_name}...")
        logger.info(f"Using cache directory: {CACHE_DIR}")
        
        try:
            gpu_memory = self.check_gpu_memory()
            
            # Determine loading strategy based on GPU memory
            load_config = self._get_load_config(gpu_memory)
            
            # Load Qwen2.5-VL-72B-Instruct-AWQ model with optimizations
            if Qwen2_5_VLForConditionalGeneration is None:
                raise ImportError("Qwen2_5_VLForConditionalGeneration not available. Please install transformers from source: pip install git+https://github.com/huggingface/transformers accelerate")
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **load_config
            )
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL-72B-Instruct-AWQ model: {e}")
            logger.error("Please ensure you have:")
            logger.error("1. Installed transformers from source: pip install git+https://github.com/huggingface/transformers accelerate")
            logger.error("2. Installed autoawq>=0.1.8: pip install 'autoawq>=0.1.8'")
            logger.error("3. Sufficient GPU memory (40GB+ recommended for AWQ)")
            logger.error("4. Proper cache directory permissions")
            raise

    def _get_load_config(self, gpu_memory: float) -> dict:
        """Get model loading configuration for Qwen2.5-VL-72B-Instruct-AWQ"""
        base_config = {
            "torch_dtype": "auto",  # Use auto for AWQ quantized model
            "device_map": "auto",
            "trust_remote_code": True,
            "cache_dir": CACHE_DIR,
            "attn_implementation": "eager"  # Use eager attention to avoid flash_attn requirement
        }
        
        logger.info(f"Loading AWQ quantized model with {gpu_memory:.1f}GB GPU memory")
        
        # AWQ model is already quantized, no need for additional quantization
        if gpu_memory < 40:  # AWQ model should work with 40GB+
            logger.warning("GPU memory may be insufficient for 72B model even with AWQ quantization. 40GB+ recommended.")
            
        return base_config


    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None and self.processor is not None

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded(),
            "device": str(self.model.device) if self.model else None,
            "dtype": str(self.model.dtype) if self.model else None
        }