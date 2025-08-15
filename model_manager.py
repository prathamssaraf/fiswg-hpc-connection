"""
Model Manager for Qwen2.5-VL
Handles model loading, installation, and management
"""

import sys
import subprocess
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
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

    def install_packages(self):
        """Install required packages for Qwen2.5-VL"""
        logger.info("Installing Qwen2.5-VL packages...")
        
        for package in REQUIRED_PACKAGES:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "--target", 
                    PACKAGES_DIR, package
                ], check=True)
                logger.info(f"✓ Installed {package}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to install {package}: {e}")

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
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
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
            logger.error(f"Failed to load model: {e}")
            self._try_fallback_model()

    def _get_load_config(self, gpu_memory: float) -> dict:
        """Get model loading configuration based on available GPU memory"""
        base_config = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
            "cache_dir": CACHE_DIR
        }
        
        if gpu_memory < 80:  # 72B model needs substantial memory
            logger.warning("GPU memory may be insufficient for 72B model. Enabling optimizations...")
            base_config["load_in_8bit"] = True
            
        if torch.cuda.is_available():
            base_config["attn_implementation"] = "flash_attention_2"
            
        return base_config

    def _try_fallback_model(self):
        """Try loading fallback model if main model fails"""
        logger.info(f"Attempting fallback to {FALLBACK_MODEL_NAME}...")
        try:
            self.model_name = FALLBACK_MODEL_NAME
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            logger.info("✓ Fallback model loaded successfully")
        except Exception as e2:
            logger.error(f"Failed to load fallback model: {e2}")
            raise

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