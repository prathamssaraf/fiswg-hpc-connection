#!/usr/bin/env python3
"""
Quick fix script to ensure accelerate is properly installed in the virtual environment
"""

import subprocess
import sys
import os

def fix_accelerate():
    """Reinstall accelerate in virtual environment"""
    
    venv_dir = "/scratch/ps5218/qwen_venv"
    venv_pip = f"{venv_dir}/bin/pip"
    venv_python = f"{venv_dir}/bin/python"
    
    # Set environment variables
    pip_cache_dir = "/scratch/ps5218/pip_cache"
    env = os.environ.copy()
    env['PIP_CACHE_DIR'] = pip_cache_dir
    
    print("🔧 Fixing accelerate installation...")
    
    # 1. Uninstall accelerate
    try:
        print("1. Uninstalling existing accelerate...")
        subprocess.run([venv_pip, "uninstall", "-y", "accelerate"], env=env, check=True)
        print("   ✓ Uninstalled accelerate")
    except subprocess.CalledProcessError as e:
        print(f"   Note: accelerate might not have been installed: {e}")
    
    # 2. Clear pip cache
    try:
        print("2. Clearing pip cache...")
        subprocess.run([venv_pip, "cache", "purge"], env=env, check=True)
        print("   ✓ Cleared pip cache")
    except subprocess.CalledProcessError as e:
        print(f"   Note: Could not clear cache: {e}")
    
    # 3. Install accelerate with force
    try:
        print("3. Installing accelerate (latest version)...")
        subprocess.run([venv_pip, "install", "--upgrade", "--force-reinstall", "accelerate"], env=env, check=True)
        print("   ✓ Installed accelerate")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install accelerate: {e}")
        return False
    
    # 4. Test accelerate import
    try:
        print("4. Testing accelerate import...")
        result = subprocess.run([
            venv_python, "-c", 
            "import accelerate; print(f'accelerate {accelerate.__version__} imported successfully')"
        ], capture_output=True, text=True, env=env, check=True)
        print(f"   ✓ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to import accelerate: {e}")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
        return False

if __name__ == "__main__":
    success = fix_accelerate()
    if success:
        print("\n✅ accelerate is now properly installed!")
        print("Run your evaluation again with:")
        print("/scratch/ps5218/qwen_venv/bin/python main_venv.py")
    else:
        print("\n❌ Failed to fix accelerate installation")