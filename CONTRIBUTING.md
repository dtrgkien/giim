# Contributing to GIIM

Thank you for your interest in contributing to GIIM! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct that emphasizes respect, collaboration, and professionalism. By participating, you are expected to uphold this code.

### Our Standards

- **Be Respectful:** Treat all contributors with respect and consideration
- **Be Collaborative:** Work together constructively
- **Be Professional:** Focus on what is best for the project and the community
- **Be Patient:** Remember that everyone has different skill levels and backgrounds

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your changes
4. **Make your changes** following our guidelines
5. **Test your changes** thoroughly
6. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/yourusername/giim.git
cd giim

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

## Contribution Workflow

### 1. Create an Issue

Before starting work, create an issue to discuss your proposed changes:
- Bug reports: Describe the bug, steps to reproduce, and expected behavior
- Feature requests: Describe the feature and its motivation
- Documentation improvements: Specify what needs to be improved

### 2. Create a Branch

Create a descriptive branch name:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 3. Make Changes

- Write clear, concise commit messages
- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed

### 4. Commit Your Changes

```bash
git add .
git commit -m "feat: Add new feature X"
```

Follow [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length:** 100 characters maximum
- **Indentation:** 4 spaces (no tabs)
- **Quotes:** Use double quotes for strings
- **Imports:** Organize imports using `isort`

### Code Formatting

We use automated tools to maintain code quality:

```bash
# Format code with black
black giim/ scripts/ tests/

# Sort imports with isort
isort giim/ scripts/ tests/

# Check code style with flake8
flake8 giim/ scripts/ tests/

# Type checking with mypy (optional)
mypy giim/
```

### Documentation Standards

- **Docstrings:** Use Google-style docstrings
- **Type hints:** Include type hints for function arguments and returns
- **Comments:** Write clear, explanatory comments for complex logic

Example:

```python
def process_image(image: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """
    Process and convert an image to a tensor.
    
    Args:
        image: Input image as numpy array of shape (H, W, C)
        normalize: Whether to apply normalization (default: True)
    
    Returns:
        Processed image as PyTorch tensor of shape (C, H, W)
    
    Raises:
        ValueError: If image has invalid shape
    """
    # Implementation here
    pass
```

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names

```python
def test_graph_builder_creates_valid_hetero_graph():
    """Test that GraphBuilder creates a valid heterogeneous graph."""
    # Test implementation
    pass
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=giim --cov-report=html

# Run specific test file
pytest tests/test_graph_builder.py

# Run specific test
pytest tests/test_graph_builder.py::test_graph_builder_creates_valid_hetero_graph
```

## Documentation

### Code Documentation

- All public functions, classes, and methods must have docstrings
- Update README.md if adding new features
- Add examples for new functionality

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html
```

## Submitting Changes

### Pull Request Process

1. **Update your branch** with the latest main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request:**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

### Pull Request Guidelines

- **Title:** Clear, concise description of changes
- **Description:** Detailed explanation including:
  - What changes were made
  - Why they were made
  - How they were tested
  - Related issues (if any)
- **Tests:** Ensure all tests pass
- **Documentation:** Update relevant documentation
- **Code Review:** Address reviewer feedback promptly

### PR Template

```markdown
## Description
Brief description of changes

## Motivation
Why are these changes needed?

## Changes Made
- Change 1
- Change 2

## Testing
How were these changes tested?

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Commits follow conventional commits format

## Related Issues
Closes #123
```

## Areas for Contribution

We welcome contributions in the following areas:

### High Priority
- Bug fixes
- Performance improvements
- Documentation improvements
- Test coverage increases

### Feature Additions
- Support for additional datasets
- New imputation methods
- Improved visualization tools
- Alternative graph architectures

### Research Contributions
- Ablation studies
- Comparison with other methods
- Extension to new domains
- Theoretical analysis

## Getting Help

If you need help:

1. **Check the documentation:** README.md and code comments
2. **Search existing issues:** Someone may have had the same question
3. **Ask in discussions:** Use GitHub Discussions for questions
4. **Create an issue:** If you've found a bug or need clarification

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Significant contributions may warrant co-authorship on related publications (subject to discussion).

## License

By contributing to GIIM, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to GIIM! Your efforts help advance medical image analysis research.

