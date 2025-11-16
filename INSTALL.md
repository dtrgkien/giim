# Installation Guide

This guide provides detailed installation instructions for GIIM.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Dependency Installation](#dependency-installation)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **OS:** Linux, macOS, or Windows (Linux recommended)
- **Python:** 3.8 or higher
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 10GB free space for code, dependencies, and data

### Recommended Requirements

- **GPU:** CUDA-capable GPU with 8GB+ VRAM (NVIDIA)
- **CPU:** Multi-core processor (4+ cores)
- **RAM:** 32GB for large datasets
- **Storage:** SSD with 50GB+ free space

### Software Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- Git

## Installation Methods

### Method 1: Install from Source (Recommended)

This method is recommended for development and customization.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/giim.git
cd giim

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch (choose based on your CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch torchvision torchaudio

# 5. Install PyTorch Geometric
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 6. Install GIIM in editable mode
pip install -e .

# 7. (Optional) Install development dependencies
pip install -e ".[dev,docs]"
```

### Method 2: Install from PyPI (When Available)

```bash
pip install giim
```

### Method 3: Install with Conda

```bash
# Create conda environment
conda create -n giim python=3.9
conda activate giim

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install GIIM
pip install -e .
```

## Dependency Installation

### Core Dependencies

The following dependencies will be installed automatically:

- **torch** (>=1.12.0): Deep learning framework
- **torchvision** (>=0.13.0): Computer vision utilities
- **torch-geometric** (>=2.0.0): Graph neural network library
- **timm** (>=0.6.0): Image models (ConvNeXt)
- **numpy** (>=1.20.0): Numerical computing
- **pandas** (>=1.3.0): Data manipulation
- **scikit-learn** (>=1.0.0): Machine learning utilities
- **pillow** (>=9.0.0): Image processing
- **pyyaml** (>=6.0): Configuration parsing

### Optional Dependencies

For development:

```bash
pip install -e ".[dev]"
```

This installs:
- pytest: Testing framework
- black: Code formatter
- flake8: Linter
- mypy: Type checker
- isort: Import sorter

For documentation:

```bash
pip install -e ".[docs]"
```

This installs:
- sphinx: Documentation generator
- sphinx-rtd-theme: ReadTheDocs theme

### Manual Dependency Installation

If automatic installation fails, install dependencies manually:

```bash
pip install -r requirements.txt
```

## Verification

### 1. Import Test

```python
python -c "import giim; print(f'GIIM version: {giim.__version__}')"
```

### 2. Component Test

```python
python -c "
from giim import GIIMModel, DatasetLoader, FeatureExtractor, GraphBuilder
print('All components imported successfully!')
"
```

### 3. GPU Test

```python
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### 4. Run Example

```bash
python examples/quick_start.py
```

### 5. Run Tests

```bash
pytest tests/ -v
```

## Troubleshooting

### Common Issues

#### 1. PyTorch Installation Fails

**Problem:** PyTorch installation fails or is incompatible with CUDA

**Solution:**
- Check your CUDA version: `nvcc --version` or `nvidia-smi`
- Install PyTorch matching your CUDA version from [pytorch.org](https://pytorch.org)
- For CPU-only: `pip install torch torchvision torchaudio`

#### 2. PyTorch Geometric Installation Fails

**Problem:** Cannot install PyG or its dependencies

**Solution:**
```bash
# Method 1: Install from wheels
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Method 2: Use conda
conda install pyg -c pyg

# Method 3: Build from source
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

#### 3. CUDA Out of Memory

**Problem:** RuntimeError: CUDA out of memory

**Solution:**
- Reduce batch size in config: `batch_size: 4`
- Reduce model size: `hidden_dims: [256, 128, 64]`
- Enable gradient checkpointing
- Use CPU if GPU memory is insufficient

#### 4. Import Errors

**Problem:** ModuleNotFoundError or ImportError

**Solution:**
```bash
# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Ensure virtual environment is activated
which python  # Should point to venv/bin/python
```

#### 5. Version Conflicts

**Problem:** Dependency version conflicts

**Solution:**
```bash
# Create fresh environment
python -m venv fresh_venv
source fresh_venv/bin/activate

# Install with specific versions
pip install torch==2.0.0 torchvision==0.15.0
pip install torch-geometric==2.3.0

# Install GIIM
pip install -e .
```

### Platform-Specific Issues

#### Windows

- Use `venv\Scripts\activate` instead of `source venv/bin/activate`
- Install Microsoft Visual C++ Build Tools if needed
- Use PowerShell or Command Prompt (not Git Bash)

#### macOS

- PyTorch with CUDA is not available on macOS
- Use CPU version: `pip install torch torchvision torchaudio`
- Install Xcode Command Line Tools: `xcode-select --install`

#### Linux

- Ensure CUDA drivers are up to date
- Check library paths: `echo $LD_LIBRARY_PATH`
- May need to install system packages: `apt-get install python3-dev`

### Getting Help

If you encounter issues not covered here:

1. **Check existing issues:** [GitHub Issues](https://github.com/yourusername/giim/issues)
2. **Create new issue:** Include:
   - Python version: `python --version`
   - PyTorch version: `python -c "import torch; print(torch.__version__)"`
   - CUDA version: `nvcc --version`
   - Error message and stack trace
   - Steps to reproduce
3. **Community support:** [GitHub Discussions](https://github.com/yourusername/giim/discussions)

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](README.md#quick-start)
2. Prepare your data following [Dataset Documentation](docs/datasets.md)
3. Run example scripts in `examples/`
4. Train your first model: `python scripts/train.py --config configs/default.yaml`

## Uninstallation

To remove GIIM:

```bash
pip uninstall giim
```

To remove the entire environment:

```bash
# Deactivate first
deactivate

# Remove environment directory
rm -rf venv/
```

