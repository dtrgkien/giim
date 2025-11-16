.PHONY: help install install-dev test lint format clean docs

help:
	@echo "GIIM Development Commands"
	@echo "========================="
	@echo "install         Install package"
	@echo "install-dev     Install package with development dependencies"
	@echo "test            Run tests"
	@echo "lint            Run linters (flake8, mypy)"
	@echo "format          Format code (black, isort)"
	@echo "clean           Clean build artifacts"
	@echo "docs            Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"
	pre-commit install

test:
	pytest tests/ -v --cov=giim --cov-report=html --cov-report=term

lint:
	flake8 giim/ scripts/ tests/
	mypy giim/

format:
	black giim/ scripts/ tests/ examples/
	isort giim/ scripts/ tests/ examples/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && make html

# Training shortcuts
train-liver:
	python scripts/train.py --config configs/liver_ct.yaml

train-mammo:
	python scripts/train.py --config configs/vin_dr_mammo.yaml

train-breast:
	python scripts/train.py --config configs/breastdm.yaml

# Evaluation shortcuts
eval:
	python scripts/evaluate.py --config $(CONFIG) --checkpoint $(CHECKPOINT)

