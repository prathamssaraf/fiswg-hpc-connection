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

# Check accelerate availability - will be rechecked after path setup
ACCELERATE_AVAILABLE = False

def check_accelerate_availability():
    """Check if accelerate is available after setting up paths"""
    global ACCELERATE_AVAILABLE
    try:
        import accelerate
        ACCELERATE_AVAILABLE = True
        logger = logging.getLogger(__name__)
        logger.info(f"✓ accelerate {accelerate.__version__} available")
        return True
    except ImportError:
        ACCELERATE_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning("✗ accelerate not available - will use CPU/single GPU mode")
        return False

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
    DEFAULT_MODEL_NAME, REQUIRED_PACKAGES, 
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
        """Check if a package is already installed in target directory"""
        clean_name = package_name.split('>=')[0].split('==')[0].split('[')[0]
        
        # First check if it's available in current Python path (system or already added)
        spec = importlib.util.find_spec(clean_name)
        if spec is not None:
            return True
            
        # Check if installed in our target packages directory
        import os
        from pathlib import Path
        
        packages_path = Path(PACKAGES_DIR)
        if packages_path.exists():
            # Common package directory patterns
            possible_paths = [
                packages_path / clean_name,
                packages_path / f"{clean_name}.py",
                packages_path / f"{clean_name}-*.dist-info",
            ]
            
            # Check for any matching directories/files
            for pattern in [f"{clean_name}*", f"{clean_name.replace('-', '_')}*"]:
                matches = list(packages_path.glob(pattern))
                if matches:
                    return True
                    
        return False
    
    def _check_autoawq_version(self):
        """Check if autoawq is installed with correct version"""
        try:
            import autoawq
            version = autoawq.__version__
            logger.info(f"AutoAWQ version: {version}")
            
            # For version 0.2.9, we know it's >= 0.1.8
            if version.startswith('0.2.'):
                return True
                
            # Parse version to check if >= 0.1.8
            try:
                major, minor, patch = map(int, version.split('.'))
                if major > 0 or (major == 0 and minor > 1) or (major == 0 and minor == 1 and patch >= 8):
                    return True
                else:
                    logger.warning(f"AutoAWQ version {version} is too old, need >= 0.1.8")
                    return False
            except ValueError:
                # Version string parsing failed, assume it's valid if we got here
                logger.info(f"Could not parse version {version}, assuming it's valid")
                return True
                
        except ImportError:
            logger.warning("AutoAWQ not found")
            return False
        except Exception as e:
            logger.warning(f"Error checking AutoAWQ version: {e}")
            return False

    def _check_numpy_version(self):
        """Check if numpy is installed with compatible version"""
        try:
            # Clear any cached numpy modules to ensure fresh import
            import sys
            for module_name in list(sys.modules.keys()):
                if module_name.startswith('numpy'):
                    del sys.modules[module_name]
            
            import numpy
            version = numpy.__version__
            logger.info(f"NumPy version: {version}")
            # Parse version to check if >= 1.21.0 and < 2.0.0
            parts = version.split('.')
            major, minor = int(parts[0]), int(parts[1])
            if major == 1 and minor >= 21:
                return True
            elif major > 1 and major < 2:
                return True
            else:
                logger.warning(f"NumPy version {version} may cause compatibility issues, need >= 1.21.0, < 2.0.0")
                return False
        except ImportError:
            logger.warning("NumPy not found")
            return False
        except Exception as e:
            logger.warning(f"Error checking NumPy version: {e}")
            return False

    def install_packages(self):
        """Install required packages for Qwen2.5-VL (only if not already installed)"""
        # Ensure pip cache is in scratch directory to avoid disk quota issues
        import os
        pip_cache_dir = '/scratch/ps5218/pip_cache'
        os.environ['PIP_CACHE_DIR'] = pip_cache_dir
        os.makedirs(pip_cache_dir, exist_ok=True)
        
        packages_to_install = []
        
        # Special handling for autoawq
        if not self._check_autoawq_version():
            packages_to_install.append("autoawq>=0.1.8")
            logger.info("✗ autoawq needs installation/upgrade")
        else:
            logger.info("✓ autoawq already installed with correct version")
        
        # Special handling for numpy
        if not self._check_numpy_version():
            packages_to_install.append("numpy>=1.21.0,<2.0.0")
            logger.info("✗ numpy needs installation/upgrade")
        else:
            logger.info("✓ numpy already installed with correct version")
        
        # Recheck accelerate availability after path setup
        check_accelerate_availability()
        
        # Special handling for accelerate
        if not ACCELERATE_AVAILABLE:
            packages_to_install.append("accelerate")
            logger.info("✗ accelerate needs installation")
        else:
            logger.info("✓ accelerate already available")
        
        for package in REQUIRED_PACKAGES:
            package_name = package.split('>=')[0].split('==')[0].split('[')[0]  # Handle [extras]
            if package_name in ["autoawq", "accelerate", "numpy"]:
                continue  # Already handled above
            if not self._check_package_installed(package_name):
                packages_to_install.append(package)
            else:
                logger.info(f"✓ {package_name} already installed")
        
        if packages_to_install:
            logger.info(f"Installing {len(packages_to_install)} missing packages...")
            
            # Install numpy first to scratch directory with no-deps to avoid conflicts
            numpy_packages = [pkg for pkg in packages_to_install if pkg.startswith('numpy')]
            if numpy_packages:
                for numpy_pkg in numpy_packages:
                    try:
                        logger.info(f"Installing {numpy_pkg} to scratch directory (no-deps)")
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", "--target", 
                            PACKAGES_DIR, "--upgrade", "--force-reinstall", "--no-deps", numpy_pkg
                        ], check=True)
                        logger.info(f"✓ Installed {numpy_pkg}")
                        packages_to_install.remove(numpy_pkg)
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Failed to install {numpy_pkg}: {e}")
            
            # Install remaining packages normally to scratch directory
            for package in packages_to_install:
                try:
                    if package.startswith('autoawq'):
                        # Special handling for autoawq to avoid build issues
                        logger.info(f"Installing {package} with special build handling")
                        try:
                            subprocess.run([
                                sys.executable, "-m", "pip", "install", "--target", 
                                PACKAGES_DIR, "--upgrade", "--force-reinstall", 
                                "--no-build-isolation", package
                            ], check=True)
                        except subprocess.CalledProcessError:
                            # Fallback to binary only
                            logger.info(f"Retrying {package} with binary-only installation")
                            subprocess.run([
                                sys.executable, "-m", "pip", "install", "--target", 
                                PACKAGES_DIR, "--upgrade", "--force-reinstall", 
                                "--only-binary=all", package
                            ], check=True)
                    elif package.startswith('accelerate'):
                        # Use --upgrade --force-reinstall for accelerate
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", "--target", 
                            PACKAGES_DIR, "--upgrade", "--force-reinstall", package
                        ], check=True)
                    else:
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", "--target", 
                            PACKAGES_DIR, package
                        ], check=True)
                    logger.info(f"✓ Installed {package}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {package}: {e}")
            
            # Recheck accelerate after installation if it was installed
            if any("accelerate" in pkg.lower() for pkg in packages_to_install):
                check_accelerate_availability()
        else:
            logger.info("All required packages already installed")

    def check_gpu_memory(self):
        """Check available GPU memory and return info for all GPUs"""
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            total_memory = 0
            
            logger.info(f"Found {num_gpus} GPU(s):")
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                gpu_memory = props.total_memory / (1024**3)
                total_memory += gpu_memory
                logger.info(f"  GPU {i}: {props.name} - {gpu_memory:.1f} GB")
            
            logger.info(f"Total GPU memory across all devices: {total_memory:.1f} GB")
            return total_memory
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
            
            # If accelerate is not available, manually move model to device
            if not ACCELERATE_AVAILABLE:
                num_gpus = torch.cuda.device_count()
                if torch.cuda.is_available() and num_gpus > 1 and "72B" in self.model_name:
                    # Try manual multi-GPU distribution for 72B model
                    logger.info(f"Attempting manual distribution across {num_gpus} GPUs...")
                    try:
                        # Use DataParallel for multi-GPU inference
                        device_ids = list(range(num_gpus))
                        logger.info(f"Using DataParallel with devices: {device_ids}")
                        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
                        self.model = self.model.to("cuda:0")
                        logger.info("✓ Model distributed across multiple GPUs using DataParallel")
                    except Exception as dp_error:
                        logger.warning(f"DataParallel failed: {dp_error}")
                        logger.info("Falling back to single GPU...")
                        # Clear GPU memory and try single GPU
                        torch.cuda.empty_cache()
                        # Try loading 7B model instead
                        if "72B" in self.model_name:
                            logger.warning("72B model too large for single GPU, switching to 7B model")
                            self.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
                            # Reload with 7B model
                            logger.info("Reloading with 7B model...")
                            return self.load_model()
                elif torch.cuda.is_available():
                    logger.info("Moving model to CUDA device...")
                    self.model = self.model.to("cuda:0")
                else:
                    logger.info("Using CPU for model...")
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                cache_dir=CACHE_DIR
            )
            
            logger.info("✓ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL-72B-Instruct model: {e}")
            logger.error("Please ensure you have:")
            logger.error("1. Installed transformers from source: pip install git+https://github.com/huggingface/transformers accelerate")
            logger.error("2. Sufficient GPU memory (80GB+ recommended for bfloat16)")
            logger.error("3. Proper cache directory permissions")
            raise

    def _get_load_config(self, gpu_memory: float) -> dict:
        """Get model loading configuration optimized for multi-GPU setup"""
        base_config = {
            "torch_dtype": torch.bfloat16,  # Use bfloat16 for memory efficiency
            "trust_remote_code": True,
            "cache_dir": CACHE_DIR,
            "attn_implementation": "sdpa",  # Use SDPA attention to avoid Triton compilation
            "low_cpu_mem_usage": True  # Reduce CPU memory usage during loading
        }
        
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        # Explicit multi-GPU configuration for Qwen2.5-VL
        if ACCELERATE_AVAILABLE and num_gpus > 1:
            # Create explicit device map for better control
            if "72B" in self.model_name:
                # For 72B model, use explicit device mapping to ensure proper distribution
                logger.info(f"Configuring explicit device mapping for 72B model across {num_gpus} GPUs")
                base_config["device_map"] = "auto"
                # Set max memory per GPU with buffer for CUDA operations
                max_memory_per_gpu = "30GiB"  # Reserve ~2GB per GPU for CUDA overhead
                base_config["max_memory"] = {i: max_memory_per_gpu for i in range(num_gpus)}
                
                # Force model parallelism
                base_config["torch_dtype"] = torch.bfloat16
                
                logger.info(f"Multi-GPU setup: {num_gpus} GPUs, {max_memory_per_gpu} per GPU")
                logger.info(f"Total available GPU memory: {gpu_memory:.1f}GB")
                
            elif "7B" in self.model_name:
                # 7B model can fit on single GPU but use multiple for inference speed
                logger.info(f"7B model with multi-GPU acceleration across {num_gpus} GPUs")
                base_config["device_map"] = "auto"
                max_memory_per_gpu = "15GiB"  # 7B model is smaller
                base_config["max_memory"] = {i: max_memory_per_gpu for i in range(num_gpus)}
                
        elif ACCELERATE_AVAILABLE and num_gpus == 1:
            # Single GPU configuration
            base_config["device_map"] = "auto"
            logger.info(f"Loading model on single GPU with {gpu_memory:.1f}GB memory")
            
        elif not ACCELERATE_AVAILABLE and num_gpus > 1 and "72B" in self.model_name:
            # Fallback: Try to use manual device distribution for 72B model
            logger.warning("accelerate not available - attempting manual multi-GPU fallback for 72B model")
            logger.info(f"Will attempt to load on CPU first, then distribute across {num_gpus} GPUs")
            # Load on CPU first to avoid single GPU OOM, then move to GPUs
            base_config["torch_dtype"] = torch.bfloat16
            # Don't set device - we'll handle this manually after loading
            
        else:
            # CPU or no accelerate fallback
            if not ACCELERATE_AVAILABLE:
                logger.warning("accelerate not available - loading on single device")
            else:
                logger.warning("No GPUs detected - using CPU")
            logger.info(f"Loading model for single device with {gpu_memory:.1f}GB memory")
        
        # Check memory requirements based on model size
        if "72B" in self.model_name:
            if gpu_memory < 80:
                logger.warning(f"GPU memory ({gpu_memory:.1f}GB) may be insufficient for 72B model.")
                logger.warning("80GB+ recommended for bfloat16. Consider using 7B model if errors occur.")
            else:
                logger.info(f"✓ Sufficient GPU memory ({gpu_memory:.1f}GB) for 72B model")
        elif "7B" in self.model_name:
            if gpu_memory < 16:
                logger.warning(f"GPU memory ({gpu_memory:.1f}GB) may be insufficient for 7B model.")
                logger.warning("16GB+ recommended for bfloat16.")
            else:
                logger.info(f"✓ Sufficient GPU memory ({gpu_memory:.1f}GB) for 7B model")
        else:
            logger.info("Unknown model size - proceeding with current memory configuration")
        
        return base_config


    def is_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None and self.processor is not None

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        info = {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded(),
            "dtype": str(self.model.dtype) if self.model else None
        }
        
        # Add device information for multi-GPU setups
        if self.model and hasattr(self.model, 'hf_device_map'):
            info["device_map"] = self.model.hf_device_map
            info["distributed_across_gpus"] = len(self.model.hf_device_map) > 1
        elif self.model:
            info["device"] = str(self.model.device)
            info["distributed_across_gpus"] = False
        else:
            info["device"] = None
            info["distributed_across_gpus"] = False
            
        return info
    
    def print_gpu_utilization(self):
        """Print current GPU utilization"""
        if torch.cuda.is_available():
            logger.info("=== GPU Utilization ===")
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                utilization = (memory_allocated / memory_total) * 100
                
                logger.info(f"GPU {i}: {memory_allocated:.1f}GB/{memory_total:.1f}GB ({utilization:.1f}%) allocated")
                logger.info(f"       {memory_reserved:.1f}GB reserved")
        else:
            logger.info("No CUDA GPUs available")