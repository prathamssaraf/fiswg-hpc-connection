#!/usr/bin/env python3
"""
Fix NumPy 2.0 compatibility issues by forcing NumPy 1.26.4
This script removes all numpy installations and reinstalls with compatible version
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def fix_numpy_compatibility():
    """Fix NumPy compatibility by forcing version 1.26.4"""
    
    # Set environment variables first
    scratch_packages = '/scratch/ps5218/python_packages'
    scratch_pip_cache = '/scratch/ps5218/pip_cache'
    
    os.environ['PIP_CACHE_DIR'] = scratch_pip_cache
    os.makedirs(scratch_pip_cache, exist_ok=True)
    
    print("🔧 Fixing NumPy 2.0 compatibility issues...")
    print("This is a known issue where packages compiled with NumPy 1.x can't run with NumPy 2.0")
    
    # 1. Remove ALL numpy installations from scratch directory
    print("\n1. Removing all NumPy installations...")
    packages_path = Path(scratch_packages)
    if packages_path.exists():
        for item in packages_path.glob("numpy*"):
            print(f"   Removing {item}")
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
    
    # 2. Clear Python module cache
    print("2. Clearing Python module cache...")
    modules_to_clear = []
    for module_name in list(sys.modules.keys()):
        if 'numpy' in module_name.lower():
            modules_to_clear.append(module_name)
    
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
            print(f"   Cleared {module}")
    
    # 3. Install NumPy 1.26.4 specifically
    print("3. Installing NumPy 1.26.4...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--target", 
            scratch_packages, "--upgrade", "--force-reinstall", 
            "--no-deps", "numpy==1.26.4"
        ], check=True)
        print("   ✓ Installed NumPy 1.26.4")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install NumPy 1.26.4: {e}")
        return False
    
    # 4. Remove and reinstall accelerate to use new NumPy
    print("4. Reinstalling accelerate with correct NumPy...")
    for item in packages_path.glob("accelerate*"):
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--target", 
            scratch_packages, "--upgrade", "--force-reinstall", "accelerate"
        ], check=True)
        print("   ✓ Reinstalled accelerate")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to reinstall accelerate: {e}")
    
    # 5. Verify installation
    print("5. Verifying NumPy installation...")
    try:
        # Clear cache again
        for module_name in list(sys.modules.keys()):
            if 'numpy' in module_name.lower():
                del sys.modules[module_name]
        
        # Add scratch packages to path
        if scratch_packages not in sys.path:
            sys.path.insert(0, scratch_packages)
        
        import numpy
        print(f"   ✓ NumPy version: {numpy.__version__}")
        
        if numpy.__version__.startswith('1.26'):
            print("   ✓ NumPy version is compatible!")
            return True
        else:
            print(f"   ❌ Wrong NumPy version: {numpy.__version__}")
            return False
            
    except ImportError as e:
        print(f"   ❌ Failed to import NumPy: {e}")
        return False

if __name__ == "__main__":
    success = fix_numpy_compatibility()
    if success:
        print("\n🎉 NumPy compatibility fixed!")
        print("You can now run: python3 main.py")
    else:
        print("\n❌ Failed to fix NumPy compatibility")
        print("You may need to manually remove all packages and reinstall")