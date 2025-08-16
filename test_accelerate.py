#!/usr/bin/env python3
"""
Simple test to verify accelerate is working in the current environment
"""

import sys
import os

def test_accelerate():
    print(f"Python executable: {sys.executable}")
    print(f"Current working directory: {os.getcwd()}")
    
    try:
        import accelerate
        print(f"✅ accelerate {accelerate.__version__} imported successfully")
        print(f"   Location: {accelerate.__file__}")
        
        # Test basic accelerate functionality
        from accelerate import Accelerator
        accelerator = Accelerator()
        print(f"✅ Accelerator initialized successfully")
        print(f"   Device: {accelerator.device}")
        print(f"   Process index: {accelerator.process_index}")
        
        return True
    except ImportError as e:
        print(f"❌ Failed to import accelerate: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing accelerate: {e}")
        return False

if __name__ == "__main__":
    success = test_accelerate()
    if success:
        print("\n✅ accelerate is fully functional!")
    else:
        print("\n❌ accelerate has issues")
    sys.exit(0 if success else 1)