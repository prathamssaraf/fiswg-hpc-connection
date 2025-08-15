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
├── main_venv.py            # Entry point for the evaluation (virtual environment)
├── setup_venv.py           # Virtual environment setup script
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

### Installation Steps
```bash
# Step 1: Create virtual environment
python3 setup_venv.py

# Step 2: Run the evaluation
/scratch/ps5218/qwen_venv/bin/python main_venv.py
```

### Required Packages (automatically installed)
- torch
- transformers (from source)
- qwen-vl-utils[decord]==0.0.8
- scikit-learn
- PIL (Pillow)
- numpy>=1.21.0,<2.0.0
- tqdm
- requests
- accelerate

## Usage

### Basic Evaluation
```bash
# First time setup
python3 setup_venv.py

# Run evaluation with interactive model selection
/scratch/ps5218/qwen_venv/bin/python main_venv.py
```

### Interactive Model Selection
The system now supports interactive model selection:
- Option 1: Qwen2.5-VL-7B-Instruct (16GB+ GPU memory)
- Option 2: Qwen2.5-VL-72B-Instruct (80GB+ GPU memory)

Modify `config.py` to adjust:
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
- Interactive selection at runtime

### Optimization Features
- Non-quantized models for best quality
- SDPA attention implementation
- Virtual environment isolation
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

### Virtual Environment Approach
The system now uses a virtual environment to avoid package conflicts:

```bash
# Recreate virtual environment if needed
python3 setup_venv.py
```

### Common Issues

#### Package Installation Issues
```bash
# Recreate the virtual environment
python3 setup_venv.py
```

#### Disk Quota Exceeded
The system automatically uses scratch directories. Ensure you have:
- `/scratch/ps5218/huggingface_cache` for model cache
- `/scratch/ps5218/python_packages` for package installation
- `/scratch/ps5218/pip_cache` for pip cache

#### Model Loading Errors
1. Ensure transformers is installed from source (handled by setup_venv.py)
2. Check GPU memory (16GB+ for 7B, 80GB+ for 72B model)
3. Verify virtual environment is active

#### Package Reinstalls Every Run
The virtual environment approach eliminates this issue:
```bash
/scratch/ps5218/qwen_venv/bin/python main_venv.py
```

### Available Scripts

- `setup_venv.py` - Creates virtual environment with all dependencies
- `main_venv.py` - Main entry point with interactive model selection

## Contributing

Feel free to submit issues and enhancement requests!