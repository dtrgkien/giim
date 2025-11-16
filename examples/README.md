# GIIM Examples

⚠️ **STATUS: PRE-RELEASE - API DEMONSTRATIONS**

This directory contains example scripts demonstrating the intended API usage for GIIM.

**Note:** These are API demonstrations showing the planned interface. Full implementation is pending institutional review and will be released in Q1 2027.

## Available Examples

### 1. Quick Start (`quick_start.py`)

Demonstrates the complete GIIM workflow:
- Loading configuration
- Initializing components
- Training a model
- Evaluating on test data

**Usage:**
```bash
python examples/quick_start.py
```

**Expected Output:**
- Shows intended API structure
- Displays workflow steps
- Indicates "Implementation pending" notices

### 2. Custom Dataset (`custom_dataset_example.py`)

Shows how to adapt GIIM for your own dataset:
- Creating custom configuration
- Data organization requirements
- Model initialization
- Training and evaluation workflow

**Usage:**
```bash
python examples/custom_dataset_example.py
```

**Expected Output:**
- Custom configuration template
- Dataset structure requirements
- Integration workflow steps

## Intended API Usage (Once Released)

### Basic Training Workflow

```python
from giim import DatasetLoader, FeatureExtractor, GraphBuilder, GIIMModel, Trainer, Utils
import yaml

# 1. Load configuration
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. Initialize components
dataset_loader = DatasetLoader(config)
feature_extractor = FeatureExtractor(config)
graph_builder = GraphBuilder(config['dataset']['name'])
utils = Utils(config, dataset_loader)

# 3. Load data
train_data = dataset_loader.load_data('liver_ct', 'train')
val_data = dataset_loader.load_data('liver_ct', 'val')
test_data = dataset_loader.load_data('liver_ct', 'test')

# 4. Create model
model = GIIMModel(
    in_dim=769,
    hidden_dims=[512, 256, 128, 64],
    num_classes=4,
    num_views=3
)

# 5. Train
trainer = Trainer(
    model=model,
    train_loader=train_data,
    val_loader=val_data,
    config=config,
    feature_extractor=feature_extractor,
    graph_builder=graph_builder,
    utils=utils
)

history = trainer.train()

# 6. Evaluate
from giim import Evaluation

evaluation = Evaluation(
    model=model,
    test_loader=test_data,
    config=config,
    feature_extractor=feature_extractor,
    graph_builder=graph_builder,
    utils=utils
)

results = evaluation.evaluate()
evaluation.print_summary(results)
```

### Custom Imputation (Once Released)

```python
from giim import Utils

# Initialize with custom database size
config['utils']['rag_db_size'] = 200
utils = Utils(config, dataset_loader)

# Use covariance-based imputation
imputed_features = utils.impute_missing_features(
    available_features,
    method='covariance',
    target_view='delay'
)
```

### Evaluation Protocol (Once Released)

```python
# Evaluate across multiple scenarios
results = {
    'full_view': {},    # Patients with all views
    'miss_view': {}     # Patients with one view missing
}

for test_set in ['full_view', 'miss_view']:
    for missing_rate in [0.0, 0.2, 0.5, 0.7, 1.0]:
        for method in ['constant', 'learnable', 'rag', 'covariance']:
            # Evaluate with specific imputation method
            metrics = evaluation.evaluate_scenario(
                test_set=test_set,
                missing_rate=missing_rate,
                imputation_method=method
            )
            results[test_set][f"{missing_rate}_{method}"] = metrics
```

## Data Structure Requirements

When the full implementation is released, organize your data as:

```
data/
├── liver_ct/
│   ├── train.csv          # patient_id,lesion_id,label,arterial_path,venous_path,delay_path
│   ├── val.csv
│   ├── test.csv
│   └── images/
│       ├── patient001_lesion01_arterial.png
│       ├── patient001_lesion01_venous.png
│       ├── patient001_lesion01_delay.png
│       └── ...
├── vin_dr_mammo/
│   └── ...
└── breastdm/
    └── ...
```

See [docs/datasets.md](../docs/datasets.md) for detailed data preparation instructions.

## Troubleshooting (For Future Reference)

### When Implementation is Released

**Data Not Found:**
- Check data path in configuration
- Verify CSV files exist (train.csv, val.csv, test.csv)
- Validate image paths in CSV are correct

**CUDA Out of Memory:**
- Reduce `batch_size` in config
- Reduce `hidden_dims` (e.g., [256, 128, 64, 32])
- Switch to CPU if GPU insufficient

**Poor Performance:**
- Check data quality and label distribution
- Increase training epochs
- Adjust learning rate (try 1e-4 or 1e-3)
- Try different imputation methods (RAG or covariance often work best)

**Missing View Handling:**
- Ensure missing_views are properly marked in PatientData
- Check imputation method is configured correctly
- Verify RAG/covariance databases are built (first run may be slow)

## Current Status

**What's Available Now:**
- ✅ API structure and documentation
- ✅ Configuration examples
- ✅ Usage demonstrations
- ✅ Data structure specifications

**What's Pending:**
- ❌ Actual implementations
- ❌ Runnable training code
- ❌ Working evaluation scripts
- ❌ Pre-trained weights
- ❌ Sample datasets

**Expected Release:** Q1 2027 (following institutional approval)

## Additional Resources

- [Architecture Documentation](../docs/architecture.md) - Detailed model architecture
- [Dataset Documentation](../docs/datasets.md) - Data preparation guide
- [Main README](../README.md) - Project overview and status
- [Installation Guide](../INSTALL.md) - Setup instructions
- [Configuration Files](../configs/) - Example configurations

## Questions?

For questions about:
- **API design**: See docstrings in `giim/` modules
- **Data preparation**: See `docs/datasets.md`
- **Architecture**: See `docs/architecture.md`
- **Release timeline**: See main README.md

Open an issue on GitHub or contact the authors.

---

**Note:** These examples demonstrate the intended API structure. Full implementation will be available following institutional review.
