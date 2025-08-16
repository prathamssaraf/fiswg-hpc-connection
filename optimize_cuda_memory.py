#!/usr/bin/env python3
"""
CUDA memory optimization script for multi-GPU Qwen2.5-VL
"""

import os
import sys

def set_cuda_optimizations():
    """Set CUDA environment variables for better memory management"""
    
    # Memory optimization settings
    cuda_settings = {
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:128',
        'CUDA_LAUNCH_BLOCKING': '0',  # Don't block CUDA launches
        'TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT': '32',  # Limit cuDNN cache
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',  # Consistent GPU ordering
    }
    
    print("🚀 Setting CUDA optimizations for multi-GPU setup...")
    
    for key, value in cuda_settings.items():
        os.environ[key] = value
        print(f"   {key}={value}")
    
    # Clear GPU memory if possible
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n📊 Available GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                if torch.cuda.memory_allocated(i) > 0:
                    print(f"   Clearing GPU {i} memory...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            print("   ✓ GPU memory cleared")
    except ImportError:
        print("   Note: PyTorch not available for memory clearing")
    
    print("\n✅ CUDA optimizations set!")
    print("These settings will help with:")
    print("  - Expandable memory segments to reduce fragmentation")
    print("  - Better memory allocation for large models")
    print("  - Consistent GPU device ordering")

if __name__ == "__main__":
    set_cuda_optimizations()