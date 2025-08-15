#!/usr/bin/env python3
"""
Setup virtual environment in scratch directory for clean package management
This completely avoids all system Python conflicts
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def setup_virtual_environment():
    """Create and setup virtual environment in scratch directory"""
    
    # Define paths
    scratch_dir = '/scratch/ps5218'
    venv_dir = f'{scratch_dir}/qwen_venv'
    
    print("🐍 Setting up virtual environment in scratch directory...")
    print("This will create a completely clean Python environment")
    
    # 1. Remove existing venv if it exists
    venv_path = Path(venv_dir)
    if venv_path.exists():
        print(f"1. Removing existing virtual environment: {venv_dir}")
        shutil.rmtree(venv_path, ignore_errors=True)
    else:
        print("1. No existing virtual environment found")
    
    # 2. Create new virtual environment
    print("2. Creating new virtual environment...")
    try:
        subprocess.run([
            sys.executable, "-m", "venv", venv_dir
        ], check=True)
        print(f"   ✓ Created virtual environment at {venv_dir}")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to create virtual environment: {e}")
        return False
    
    # 3. Get venv python and pip paths
    venv_python = f"{venv_dir}/bin/python"
    venv_pip = f"{venv_dir}/bin/pip"
    
    # 4. Set pip cache environment variable for venv
    pip_cache_dir = f"{scratch_dir}/pip_cache"
    os.makedirs(pip_cache_dir, exist_ok=True)
    
    # 5. Upgrade pip in venv with cache setting
    print("3. Upgrading pip in virtual environment...")
    venv_env = os.environ.copy()
    venv_env['PIP_CACHE_DIR'] = pip_cache_dir
    try:
        subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip"], 
                      env=venv_env, check=True)
        print("   ✓ Upgraded pip")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to upgrade pip: {e}")
    
    # 5. Install packages in correct order
    print("4. Installing packages in virtual environment...")
    
    # Core packages first
    packages = [
        "numpy==1.26.4",  # Specific compatible version
        "torch", 
        "torchvision",
        "accelerate",
        "Pillow",
        "requests", 
        "tqdm",
        "scikit-learn",
        "qwen-vl-utils[decord]==0.0.8"
    ]
    
    for package in packages:
        try:
            print(f"   Installing {package}...")
            subprocess.run([venv_pip, "install", package], env=venv_env, check=True)
            print(f"   ✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
    
    # 6. Install transformers from source
    print("5. Installing transformers from source...")
    try:
        subprocess.run([
            venv_pip, "install", "git+https://github.com/huggingface/transformers"
        ], env=venv_env, check=True)
        print("   ✓ Installed transformers from source")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install transformers: {e}")
        # Fallback to PyPI
        try:
            subprocess.run([venv_pip, "install", "transformers"], env=venv_env, check=True)
            print("   ✓ Installed transformers from PyPI")
        except subprocess.CalledProcessError as e2:
            print(f"   ❌ Failed to install transformers: {e2}")
    
    # 7. Install autoawq with special handling and cache
    print("6. Installing autoawq...")
    try:
        subprocess.run([venv_pip, "install", "autoawq>=0.1.8"], env=venv_env, check=True)
        print("   ✓ Installed autoawq")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install autoawq: {e}")
        # Try with --no-build-isolation
        try:
            subprocess.run([venv_pip, "install", "--no-build-isolation", "autoawq>=0.1.8"], 
                          env=venv_env, check=True)
            print("   ✓ Installed autoawq (no build isolation)")
        except subprocess.CalledProcessError as e2:
            print(f"   ❌ Failed to install autoawq: {e2}")
            print("   Note: AutoAWQ installation failed - will try without it")
    
    # 8. Test installation
    print("7. Testing installation...")
    try:
        # Test numpy
        result = subprocess.run([
            venv_python, "-c", 
            "import numpy; print(f'NumPy: {numpy.__version__}')"
        ], capture_output=True, text=True, check=True)
        print(f"   ✓ {result.stdout.strip()}")
        
        # Test torch
        result = subprocess.run([
            venv_python, "-c", 
            "import torch; print(f'PyTorch: {torch.__version__}')"
        ], capture_output=True, text=True, check=True)
        print(f"   ✓ {result.stdout.strip()}")
        
        # Test accelerate  
        result = subprocess.run([
            venv_python, "-c", 
            "import accelerate; print(f'Accelerate: {accelerate.__version__}')"
        ], capture_output=True, text=True, check=True)
        print(f"   ✓ {result.stdout.strip()}")
        
        # Test transformers
        subprocess.run([
            venv_python, "-c", 
            "from transformers import AutoProcessor; print('Transformers: Import successful')"
        ], check=True)
        print("   ✓ Transformers: Import successful")
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Testing failed: {e}")
        return False
    
    # 9. Create activation script
    print("8. Creating activation script...")
    activation_script = f"""#!/bin/bash
# Activate the Qwen virtual environment
export QWEN_VENV_DIR="{venv_dir}"
export PATH="$QWEN_VENV_DIR/bin:$PATH"
export VIRTUAL_ENV="$QWEN_VENV_DIR"

# Set environment variables for scratch directories
export HF_HOME="{scratch_dir}/huggingface_cache"
export HF_HUB_CACHE="{scratch_dir}/huggingface_cache"
export TRANSFORMERS_CACHE="{scratch_dir}/huggingface_cache"
export HF_DATASETS_CACHE="{scratch_dir}/huggingface_cache"
export TORCH_HOME="{scratch_dir}/huggingface_cache"
export SCIKIT_LEARN_DATA="{scratch_dir}/scikit_learn_data"
export TRITON_CACHE_DIR="{scratch_dir}/triton_cache"
export TORCH_KERNEL_CACHE_PATH="{scratch_dir}/torch_kernels"
export PIP_CACHE_DIR="{scratch_dir}/pip_cache"
export TOKENIZERS_PARALLELISM="false"
export TRITON_DISABLE_JIT="1"
export TRITON_DISABLE_CACHE="1"
export PYTORCH_DISABLE_FLASH_ATTENTION="1"

echo "✓ Activated Qwen virtual environment"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
"""
    
    with open("activate_qwen_venv.sh", "w") as f:
        f.write(activation_script)
    
    os.chmod("activate_qwen_venv.sh", 0o755)
    print("   ✓ Created activate_qwen_venv.sh")
    
    return True

if __name__ == "__main__":
    success = setup_virtual_environment()
    if success:
        print("\n🎉 Virtual environment setup complete!")
        print("\nTo use the environment:")
        print("1. source activate_qwen_venv.sh")
        print("2. python main.py")
        print("\nOr directly:")
        print("/scratch/ps5218/qwen_venv/bin/python main.py")
    else:
        print("\n❌ Virtual environment setup failed")