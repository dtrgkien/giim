# Changelog

All notable changes to the GIIM project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.0] - 2025-11-16 (Pre-release)

### Added
- Complete documentation and repository structure
- Architecture documentation and API design
- Configuration files for all datasets
- Paper submitted for review (August 2025)
- Expected publication: January 2026
- Pre-release version pending institutional approval

### Status
- Documentation: Complete
- Implementation: Pending institutional review
- Expected full release: Q1 2026

## [1.0.0] - TBD (Planned for Q1 2026)

### Added
- Initial release of GIIM (Graph-based Learning of Inter- and Intra-view Dependencies)
- Heterogeneous graph neural network architecture for multi-view medical imaging
- Four missing view imputation methods: Constant, Learnable, RAG, Covariance
- Support for three medical imaging datasets: Liver CT, VinDr-Mammo, BreastDM
- ConvNeXt-Tiny feature extractor with ImageNet pretraining
- Comprehensive evaluation framework across missing-view rates
- Professional repository structure following AI research standards
- Complete documentation including:
  - Detailed README with installation and usage instructions
  - Architecture documentation
  - Dataset preparation guide
  - API documentation
- Example scripts and notebooks
- Unit tests for core components
- Configuration system using YAML files
- Training and evaluation scripts
- Data preprocessing utilities
- MIT License
- Citation file (BibTeX)
- Contributing guidelines

### Repository Structure
- Organized as Python package (`giim/`)
- Separate directories for configs, scripts, tests, docs, and examples
- Proper package setup with `setup.py` and `pyproject.toml`
- Standard files: `.gitignore`, `CONTRIBUTING.md`, `MANIFEST.in`, `Makefile`

### Dependencies
- PyTorch >= 1.12.0
- PyTorch Geometric >= 2.0.0
- timm >= 0.6.0
- NumPy >= 1.20.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

### Features
- Patient-level graph construction with multiple lesions
- Four types of edges: alpha (tumor-view), beta (inter-view), delta (intra-view across tumors), gamma (inter-tumor)
- Attention-based message passing
- Early stopping with validation monitoring
- Checkpoint saving and loading
- TensorBoard logging
- Comprehensive metrics: Accuracy, AUC, F1-score

## [Unreleased]

### Planned
- Multi-GPU training support
- 3D medical image support
- Pre-trained model weights
- Interactive visualization dashboard
- Integration with MONAI
- Additional imputation methods
- Hyperparameter optimization utilities

