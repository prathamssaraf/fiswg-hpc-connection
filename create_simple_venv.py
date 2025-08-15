#!/usr/bin/env python3
"""
Simple virtual environment creation script
"""

import os
import subprocess
import sys

def create_simple_venv():
    """Create a simple virtual environment"""
    
    venv_dir = "/scratch/ps5218/qwen_venv"
    
    print("🐍 Creating simple virtual environment...")
    print(f"Python executable: {sys.executable}")
    print(f"Target directory: {venv_dir}")
    
    # Remove existing venv
    if os.path.exists(venv_dir):
        print("Removing existing venv...")
        import shutil
        shutil.rmtree(venv_dir, ignore_errors=True)
    
    # Create venv using python3
    try:
        print("Creating virtual environment...")
        result = subprocess.run([
            "python3", "-m", "venv", venv_dir
        ], capture_output=True, text=True, check=True)
        print("✓ Virtual environment created successfully")
        
        # Test if it was created
        venv_python = f"{venv_dir}/bin/python"
        if os.path.exists(venv_python):
            print(f"✓ Found venv python: {venv_python}")
        else:
            print(f"❌ Venv python not found: {venv_python}")
            return False
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment:")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError as e:
        print(f"❌ Python3 not found: {e}")
        print("Try using 'python' instead of 'python3'")
        
        # Try with 'python'
        try:
            print("Trying with 'python'...")
            subprocess.run([
                "python", "-m", "venv", venv_dir
            ], check=True)
            print("✓ Virtual environment created with 'python'")
            return True
        except Exception as e2:
            print(f"❌ Also failed with 'python': {e2}")
            return False

if __name__ == "__main__":
    success = create_simple_venv()
    if success:
        print("\n✓ Virtual environment ready!")
        print("Next: install packages manually")
    else:
        print("\n❌ Failed to create virtual environment")