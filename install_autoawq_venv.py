#!/usr/bin/env python3
"""
Install autoawq in virtual environment with proper cache handling
"""

import os
import subprocess
import sys

def install_autoawq():
    """Install autoawq in virtual environment"""
    
    venv_dir = "/scratch/ps5218/qwen_venv"
    pip_cache_dir = "/scratch/ps5218/pip_cache"
    venv_pip = f"{venv_dir}/bin/pip"
    
    print("🔧 Installing autoawq in virtual environment...")
    
    # Check if venv exists
    if not os.path.exists(venv_dir):
        print(f"❌ Virtual environment not found: {venv_dir}")
        print("Please run: python3 setup_venv.py")
        return False
    
    # Set environment
    env = os.environ.copy()
    env['PIP_CACHE_DIR'] = pip_cache_dir
    
    # Try different installation methods
    methods = [
        # Method 1: Standard install
        [venv_pip, "install", "autoawq>=0.1.8"],
        # Method 2: No build isolation
        [venv_pip, "install", "--no-build-isolation", "autoawq>=0.1.8"],
        # Method 3: Pre-built wheels only
        [venv_pip, "install", "--only-binary=all", "autoawq>=0.1.8"],
        # Method 4: Specific version
        [venv_pip, "install", "autoawq==0.2.9"]
    ]
    
    for i, method in enumerate(methods, 1):
        try:
            print(f"Method {i}: {' '.join(method[2:])}")
            subprocess.run(method, env=env, check=True)
            print("✓ autoawq installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Method {i} failed: {e}")
            continue
    
    print("❌ All installation methods failed")
    print("The system may work without autoawq (will use non-quantized model)")
    return False

if __name__ == "__main__":
    install_autoawq()