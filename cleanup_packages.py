#!/usr/bin/env python3
"""
Cleanup script to remove problematic package installations
Run this before main.py if you're having numpy/accelerate conflicts
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def cleanup_packages():
    """Remove problematic packages and reinstall cleanly"""
    
    # Define paths
    scratch_packages = '/scratch/ps5218/python_packages'
    
    print("🧹 Cleaning up problematic package installations...")
    
    # Remove accelerate and numpy from scratch directory
    packages_to_remove = ['accelerate', 'numpy', 'autoawq']
    
    for package in packages_to_remove:
        # Remove from scratch directory
        scratch_path = Path(scratch_packages)
        if scratch_path.exists():
            for item in scratch_path.glob(f"{package}*"):
                print(f"Removing {item}")
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
    
    print("✓ Cleaned up scratch packages directory")
    
    # Clear Python module cache
    for module_name in list(sys.modules.keys()):
        if any(pkg in module_name for pkg in ['numpy', 'accelerate', 'autoawq']):
            del sys.modules[module_name]
    
    print("✓ Cleared Python module cache")
    
    # Install core packages to user directory with specific versions
    core_packages = [
        "numpy>=1.21.0,<2.0.0",
        "accelerate",
        "autoawq>=0.1.8"
    ]
    
    print("📦 Installing core packages to scratch directory...")
    
    # Install numpy first with no-deps to avoid conflicts
    try:
        print("Installing numpy (no-deps)...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--target", 
            scratch_packages, "--upgrade", "--force-reinstall", "--no-deps", 
            "numpy>=1.21.0,<2.0.0"
        ], check=True)
        print("✓ Installed numpy")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install numpy: {e}")
    
    # Install other packages
    other_packages = ["accelerate", "autoawq>=0.1.8"]
    for package in other_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--target", 
                scratch_packages, "--upgrade", "--force-reinstall", package
            ], check=True)
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    print("🎉 Package cleanup completed!")
    print("Now you can run: python3 main.py")

if __name__ == "__main__":
    cleanup_packages()