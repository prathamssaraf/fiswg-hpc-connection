#!/usr/bin/env python3
"""
Debug script to check accelerate installation and fix issues
"""

import subprocess
import sys
import os
from pathlib import Path

def debug_accelerate():
    """Debug accelerate installation in virtual environment"""
    
    venv_dir = "/scratch/ps5218/qwen_venv"
    venv_python = f"{venv_dir}/bin/python"
    venv_pip = f"{venv_dir}/bin/pip"
    
    print("🔍 Debugging accelerate installation...")
    
    # 1. Check if we're using the right python
    print(f"\n1. Current Python: {sys.executable}")
    print(f"   Expected Python: {venv_python}")
    
    # 2. Check sys.path for virtual environment
    print(f"\n2. Python path:")
    for path in sys.path:
        print(f"   {path}")
    
    # 3. List installed packages in venv
    try:
        print(f"\n3. Installed packages in virtual environment:")
        result = subprocess.run([venv_pip, "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'accelerate' in line.lower() or 'torch' in line.lower():
                print(f"   {line}")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to list packages: {e}")
    
    # 4. Try to import accelerate directly with venv python
    try:
        print(f"\n4. Testing accelerate import with venv python:")
        result = subprocess.run([
            venv_python, "-c", 
            "import sys; print('Python:', sys.executable); import accelerate; print(f'accelerate {accelerate.__version__} found at {accelerate.__file__}')"
        ], capture_output=True, text=True, check=True)
        print(f"   ✅ {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to import accelerate:")
        print(f"   stdout: {e.stdout}")
        print(f"   stderr: {e.stderr}")
    
    # 5. Check site-packages directory
    site_packages = Path(venv_dir) / "lib64/python3.9/site-packages"
    print(f"\n5. Checking site-packages directory: {site_packages}")
    if site_packages.exists():
        accelerate_dirs = list(site_packages.glob("*accelerate*"))
        if accelerate_dirs:
            print(f"   Found accelerate directories:")
            for d in accelerate_dirs:
                print(f"     {d}")
        else:
            print(f"   ❌ No accelerate directories found")
    else:
        print(f"   ❌ Site-packages directory not found")
    
    # 6. Check for lib vs lib64
    lib_site_packages = Path(venv_dir) / "lib/python3.9/site-packages"
    print(f"\n6. Checking alternative lib directory: {lib_site_packages}")
    if lib_site_packages.exists():
        accelerate_dirs = list(lib_site_packages.glob("*accelerate*"))
        if accelerate_dirs:
            print(f"   Found accelerate directories in lib:")
            for d in accelerate_dirs:
                print(f"     {d}")
    
    return False

def reinstall_accelerate():
    """Reinstall accelerate with verbose output"""
    
    venv_dir = "/scratch/ps5218/qwen_venv"
    venv_pip = f"{venv_dir}/bin/pip"
    venv_python = f"{venv_dir}/bin/python"
    
    pip_cache_dir = "/scratch/ps5218/pip_cache"
    env = os.environ.copy()
    env['PIP_CACHE_DIR'] = pip_cache_dir
    
    print(f"\n🔧 Reinstalling accelerate...")
    
    # Remove existing installations
    try:
        print("Removing existing accelerate...")
        subprocess.run([venv_pip, "uninstall", "-y", "accelerate"], env=env)
    except:
        pass
    
    # Install with verbose output
    try:
        print("Installing accelerate with verbose output...")
        result = subprocess.run([
            venv_pip, "install", "-v", "accelerate"
        ], env=env, capture_output=True, text=True, check=True)
        print("✅ Installation completed")
        print("Output:", result.stdout[-500:])  # Last 500 chars
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("stdout:", e.stdout[-500:])
        print("stderr:", e.stderr[-500:])
        return False
    
    # Test import again
    try:
        result = subprocess.run([
            venv_python, "-c", 
            "import accelerate; print(f'✅ accelerate {accelerate.__version__} successfully imported')"
        ], capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Still cannot import accelerate: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("ACCELERATE DEBUG AND FIX SCRIPT")
    print("=" * 70)
    
    # First debug the current state
    working = debug_accelerate()
    
    if not working:
        print("\n" + "=" * 50)
        print("ATTEMPTING REINSTALLATION")
        print("=" * 50)
        working = reinstall_accelerate()
    
    if working:
        print(f"\n✅ accelerate is working! You can now run:")
        print(f"/scratch/ps5218/qwen_venv/bin/python main_venv.py")
    else:
        print(f"\n❌ accelerate installation still has issues")
        print(f"You may need to recreate the virtual environment:")
        print(f"python3 setup_venv.py")