#!/usr/bin/env python3
"""
Comprehensive package rebuild script to fix NumPy compatibility
Reinstalls all packages that depend on NumPy with the correct version
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def rebuild_all_packages():
    """Rebuild all packages with NumPy 1.26.4 compatibility"""
    
    # Set environment variables
    scratch_packages = '/scratch/ps5218/python_packages'
    scratch_pip_cache = '/scratch/ps5218/pip_cache'
    
    os.environ['PIP_CACHE_DIR'] = scratch_pip_cache
    os.makedirs(scratch_pip_cache, exist_ok=True)
    
    print("🔄 Comprehensive package rebuild for NumPy compatibility...")
    print("This will reinstall all packages that depend on NumPy")
    
    # 1. Remove ALL packages from scratch directory
    print("\n1. Removing all packages from scratch directory...")
    packages_path = Path(scratch_packages)
    if packages_path.exists():
        shutil.rmtree(packages_path, ignore_errors=True)
        print(f"   Removed {packages_path}")
    
    # Recreate directory
    packages_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Clear ALL module cache
    print("2. Clearing ALL Python module cache...")
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if any(pkg in module_name.lower() for pkg in 
               ['numpy', 'accelerate', 'transformers', 'torch', 'sklearn', 'pandas']):
            modules_to_clear.append(module_name)
    
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
    print(f"   Cleared {len(modules_to_clear)} modules")
    
    # 3. Install NumPy 1.26.4 first
    print("3. Installing NumPy 1.26.4...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--target", 
            scratch_packages, "--no-deps", "numpy==1.26.4"
        ], check=True)
        print("   ✓ Installed NumPy 1.26.4")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install NumPy: {e}")
        return False
    
    # 4. Install core packages in order
    core_packages = [
        "torch",
        "torchvision", 
        "accelerate",
        "autoawq>=0.1.8",
        "scikit-learn",
        "tqdm",
        "requests",
        "Pillow",
        "qwen-vl-utils[decord]==0.0.8"
    ]
    
    print("4. Installing core packages...")
    for package in core_packages:
        try:
            print(f"   Installing {package}...")
            if package.startswith('autoawq'):
                # Special handling for autoawq
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "--target", 
                        scratch_packages, "--upgrade", "--no-build-isolation", package
                    ], check=True)
                except subprocess.CalledProcessError:
                    subprocess.run([
                        sys.executable, "-m", "pip", "install", "--target", 
                        scratch_packages, "--upgrade", "--only-binary=all", package
                    ], check=True)
            else:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "--target", 
                    scratch_packages, package
                ], check=True)
            print(f"   ✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
    
    # 5. Install transformers from source (most important)
    print("5. Installing transformers from source...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--target", 
            scratch_packages, "--upgrade", 
            "git+https://github.com/huggingface/transformers"
        ], check=True)
        print("   ✓ Installed transformers from source")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install transformers: {e}")
        # Fallback to PyPI version
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--target", 
                scratch_packages, "--upgrade", "transformers"
            ], check=True)
            print("   ✓ Installed transformers from PyPI")
        except subprocess.CalledProcessError as e2:
            print(f"   ❌ Failed to install transformers from PyPI: {e2}")
    
    # 6. Verify installation
    print("6. Verifying installation...")
    
    # Add to path
    if scratch_packages not in sys.path:
        sys.path.insert(0, scratch_packages)
    
    # Test NumPy
    try:
        import numpy
        print(f"   ✓ NumPy version: {numpy.__version__}")
        assert numpy.__version__.startswith('1.26'), f"Wrong NumPy version: {numpy.__version__}"
    except Exception as e:
        print(f"   ❌ NumPy test failed: {e}")
        return False
    
    # Test accelerate
    try:
        import accelerate
        print(f"   ✓ Accelerate version: {accelerate.__version__}")
    except Exception as e:
        print(f"   ❌ Accelerate test failed: {e}")
        return False
    
    # Test transformers
    try:
        from transformers import AutoProcessor
        print(f"   ✓ Transformers import successful")
    except Exception as e:
        print(f"   ❌ Transformers test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = rebuild_all_packages()
    if success:
        print("\n🎉 All packages rebuilt successfully!")
        print("NumPy compatibility issues should be resolved.")
        print("You can now run: python3 main.py")
    else:
        print("\n❌ Package rebuild failed")
        print("Some packages may still have compatibility issues")