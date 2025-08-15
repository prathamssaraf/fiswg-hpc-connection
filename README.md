# FISWG HPC Connection

Complete LFW Dataset Evaluation with Qwen2.5-VL 72B for forensic facial comparison analysis using FISWG (Facial Identification Scientific Working Group) 19-component methodology.

## Overview

This project implements a comprehensive facial identification evaluation system that:
- Uses Qwen2.5-VL 72B vision-language model for forensic facial analysis
- Follows FISWG standardized 19-component facial comparison methodology
- Evaluates performance on the Labeled Faces in the Wild (LFW) dataset
- Provides detailed forensic analysis reports for each comparison

## Project Structure

```
├── main.py                 # Entry point for the evaluation
├── config.py              # Environment setup and constants
├── model_manager.py       # Qwen2.5-VL model loading and management
├── forensic_prompts.py    # FISWG 19-component analysis prompt
├── data_loader.py         # LFW dataset handling
├── image_analyzer.py      # Image analysis and facial comparison logic
├── evaluator.py           # Main evaluation orchestration
├── utils.py               # Helper functions and utilities
└── README.md              # This file
```

## Key Features

### FISWG 19-Component Analysis
The system performs detailed forensic facial comparison using all 19 FISWG standardized components:
1. Skin characteristics
2. Face/head outline
3. Face/head composition
4. Hair patterns
5. Forehead structure
6. Eyebrow characteristics
7. Eye features
8. Cheek structure
9. Nose anatomy
10. Ear morphology
11. Mouth features
12. Chin/jawline
13. Neck characteristics
14. Facial hair
15. Facial lines
16. Scars
17. Facial marks
18. Alterations
19. Other distinctive features

### Technical Features
- **High-performance computing optimized**: Designed for HPC environments with scratch directory caching
- **Memory-efficient**: Supports both 72B and 7B model variants with automatic fallback
- **Comprehensive evaluation**: Full LFW dataset support with detailed metrics
- **Forensic-grade analysis**: Professional forensic reporting format
- **Progress tracking**: Real-time evaluation progress with intermediate saves

## Installation and Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for 72B model)
- Sufficient disk space for model caching

### HPC Environment Setup
The system is configured for HPC environments with:
- Scratch directory caching (`/scratch/ps5218/huggingface_cache`)
- Custom package installation paths
- AWQ quantization for memory efficiency

### Installation for Qwen2.5-VL-72B-Instruct-AWQ
```bash
# Install transformers from source (REQUIRED)
pip install --target /scratch/ps5218/python_packages git+https://github.com/huggingface/transformers accelerate

# Install AutoAWQ for quantization support (REQUIRED)
pip install --target /scratch/ps5218/python_packages 'autoawq>=0.1.8'

# Install other dependencies
pip install --target /scratch/ps5218/python_packages 'qwen-vl-utils[decord]==0.0.8'
```

### Required Packages
- torch
- transformers (from source)
- autoawq>=0.1.8
- qwen-vl-utils[decord]==0.0.8
- scikit-learn
- PIL (Pillow)
- numpy
- tqdm
- requests
- accelerate

## Usage

### Basic Evaluation
```bash
python main.py
```

### Custom Configuration
Modify `config.py` to adjust:
- Model selection (72B vs 7B)
- Cache directories
- Evaluation parameters
- Batch sizes

### Example Output
```
LFW EVALUATION RESULTS
================================================================================
Model: Qwen/Qwen2.5-VL-72B-Instruct
Total pairs evaluated: 100
Accuracy: 0.8500 (85.00%)
Precision: 0.8200
Recall: 0.8800
F1 Score: 0.8491
Evaluation time: 3600.0 seconds
================================================================================
```

## Forensic Analysis Output

Each image pair receives a comprehensive forensic analysis report including:
- Detailed examination of all 19 FISWG components
- Morphological feature comparisons
- Expert forensic determination
- Confidence assessment
- Scientific reasoning for conclusions

## Performance Considerations

### Model Variants
- **Qwen2.5-VL-72B**: Highest accuracy, requires 80GB+ VRAM
- **Qwen2.5-VL-7B**: Faster processing, requires 16GB+ VRAM
- Automatic fallback from 72B to 7B if memory insufficient

### Optimization Features
- 8-bit quantization for memory efficiency
- Flash Attention 2 support
- Batch processing with progress tracking
- Intermediate result saving

## Research Applications

This system is designed for:
- Forensic facial identification research
- Biometric system evaluation
- Computer vision model benchmarking
- FISWG methodology validation
- Academic research in facial recognition

## Citation

If you use this code in your research, please cite:
```
FISWG HPC Connection: LFW Dataset Evaluation with Qwen2.5-VL
Forensic Facial Identification using FISWG 19-Component Methodology
```

## License

Public repository for research and educational purposes.

## Troubleshooting

### NumPy 2.0 Compatibility Issues
If you encounter errors like "numpy.core.multiarray failed to import" or "module compiled with NumPy 1.x cannot run with NumPy 2.0":

```bash
# Fix NumPy compatibility issues
python3 fix_numpy.py

# Or for complete package rebuild
python3 rebuild_all_packages.py
```

### Common Issues

#### AutoAWQ Installation Fails
```bash
# Run the cleanup script
python3 cleanup_packages.py
```

#### Disk Quota Exceeded
The system automatically uses scratch directories. Ensure you have:
- `/scratch/ps5218/huggingface_cache` for model cache
- `/scratch/ps5218/python_packages` for package installation
- `/scratch/ps5218/pip_cache` for pip cache

#### Model Loading Errors
1. Ensure transformers is installed from source
2. Verify AutoAWQ version >= 0.1.8
3. Check GPU memory (40GB+ recommended for 72B model)

#### Package Reinstalls Every Run
Run the updated package manager that checks scratch directory:
```bash
python3 main.py  # Now checks packages correctly
```

### Available Helper Scripts

- `fix_numpy.py` - Fixes NumPy 2.0 compatibility issues
- `rebuild_all_packages.py` - Complete package rebuild with correct NumPy
- `cleanup_packages.py` - Clean installation of core packages

## Contributing

Feel free to submit issues and enhancement requests!