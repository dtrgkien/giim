# Dataset Documentation

This document describes the datasets used with GIIM and how to prepare your own data.

## Table of Contents

1. [Supported Datasets](#supported-datasets)
2. [Data Format](#data-format)
3. [Dataset Preparation](#dataset-preparation)
4. [Adding Custom Datasets](#adding-custom-datasets)

## Supported Datasets

### 1. Liver CT Dataset

**Purpose:** Classification of liver lesions into four categories

**Classes:**
- 0: Benign
- 1: Ambiguous  
- 2: Malignant
- 3: HCC (Hepatocellular Carcinoma)

**Views:** Three CT phases
- Arterial phase
- Venous phase (portal venous)
- Delay phase

**Statistics:**
- ~1000 patients
- ~2500 lesions
- Image size: Variable, resized to 224×224
- Format: PNG or DICOM

**Download:** [Contact for access]

### 2. VinDr-Mammo

**Purpose:** Breast lesion classification from mammography

**Classes:**
- 0: Normal
- 1: Benign
- 2: Malignant

**Views:** Two standard mammographic projections
- CC (Craniocaudal)
- MLO (Mediolateral Oblique)

**Statistics:**
- ~20,000 mammograms
- Image size: High resolution, resized to 224×224
- Format: DICOM

**Download:** [VinDr Project](https://vindr.ai/datasets/mammo)

### 3. BreastDM

**Purpose:** Breast lesion detection and classification from dynamic contrast-enhanced MRI

**Classes:**
- 0: Benign
- 1: Malignant

**Views:** Three DCE-MRI sequences
- Pre-contrast
- Post-contrast
- Subtraction (post - pre)

**Statistics:**
- ~500 patients
- Image size: Variable slices, resized to 224×224
- Format: NIfTI or DICOM

**Download:** [Contact for access]

## Data Format

### Directory Structure

```
data/
├── liver_ct/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── images/
│       ├── patient001_lesion01_venous.png
│       ├── patient001_lesion01_arterial.png
│       ├── patient001_lesion01_delay.png
│       ├── patient001_lesion02_venous.png
│       └── ...
│
├── vin_dr_mammo/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── images/
│       ├── case001_cc.png
│       ├── case001_mlo.png
│       └── ...
│
└── breastdm/
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── images/
        ├── patient001_pre_contrast.png
        ├── patient001_post_contrast.png
        ├── patient001_subtraction.png
        └── ...
```

### CSV Format

Each CSV file (train.csv, val.csv, test.csv) should have the following columns:

#### For Liver CT

```csv
patient_id,lesion_id,label,venous_path,arterial_path,delay_path
patient001,lesion01,2,images/patient001_lesion01_venous.png,images/patient001_lesion01_arterial.png,images/patient001_lesion01_delay.png
patient001,lesion02,0,images/patient001_lesion02_venous.png,images/patient001_lesion02_arterial.png,images/patient001_lesion02_delay.png
patient002,lesion01,3,images/patient002_lesion01_venous.png,images/patient002_lesion01_arterial.png,images/patient002_lesion01_delay.png
```

#### For VinDr-Mammo

```csv
patient_id,lesion_id,label,cc_path,mlo_path
case001,lesion01,1,images/case001_cc.png,images/case001_mlo.png
case002,lesion01,2,images/case002_cc.png,images/case002_mlo.png
```

#### For BreastDM

```csv
patient_id,lesion_id,label,pre_contrast_path,post_contrast_path,subtraction_path
patient001,lesion01,1,images/patient001_pre_contrast.png,images/patient001_post_contrast.png,images/patient001_subtraction.png
```

**Required Columns:**
- `patient_id`: Unique identifier for patient
- `lesion_id`: Unique identifier for lesion within patient
- `label`: Integer class label (0-indexed)
- `{view}_path`: Relative path to image for each view

**Optional Columns:**
- `split`: train/val/test (if using single CSV)
- Additional metadata columns are allowed but ignored

## Dataset Preparation

### Step 1: Organize Images

1. Convert DICOM to PNG (if needed):

```python
import pydicom
from PIL import Image
import numpy as np

def dicom_to_png(dicom_path, output_path):
    ds = pydicom.dcmread(dicom_path)
    image = ds.pixel_array
    
    # Normalize to 0-255
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Save as PNG
    Image.fromarray(image).save(output_path)
```

2. Organize into the expected directory structure
3. Ensure consistent naming convention

### Step 2: Create CSV Files

```python
import pandas as pd
from pathlib import Path

data = []
for patient_dir in Path('raw_data').iterdir():
    patient_id = patient_dir.name
    
    for lesion_dir in patient_dir.iterdir():
        lesion_id = lesion_dir.name
        label = get_label(lesion_dir)  # Your label extraction logic
        
        # Get view paths
        views = {
            'venous': lesion_dir / 'venous.png',
            'arterial': lesion_dir / 'arterial.png',
            'delay': lesion_dir / 'delay.png'
        }
        
        data.append({
            'patient_id': patient_id,
            'lesion_id': lesion_id,
            'label': label,
            'venous_path': str(views['venous']),
            'arterial_path': str(views['arterial']),
            'delay_path': str(views['delay'])
        })

df = pd.DataFrame(data)

# Split into train/val/test
from sklearn.model_selection import train_test_split

# Split by patient to avoid data leakage
patient_ids = df['patient_id'].unique()
train_patients, test_patients = train_test_split(patient_ids, test_size=0.2, random_state=42)
train_patients, val_patients = train_test_split(train_patients, test_size=0.1, random_state=42)

train_df = df[df['patient_id'].isin(train_patients)]
val_df = df[df['patient_id'].isin(val_patients)]
test_df = df[df['patient_id'].isin(test_patients)]

# Save
train_df.to_csv('data/liver_ct/train.csv', index=False)
val_df.to_csv('data/liver_ct/val.csv', index=False)
test_df.to_csv('data/liver_ct/test.csv', index=False)
```

### Step 3: Verify Data

```python
from giim import DatasetLoader
import yaml

# Load config
with open('configs/liver_ct.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Test loading
loader = DatasetLoader(config)
train_data = loader.load_data('liver_ct', 'train')

print(f"Loaded {len(train_data)} patients")
print(f"First patient: {train_data[0]}")
```

## Adding Custom Datasets

### Step 1: Prepare Data in Standard Format

Follow the data format guidelines above to organize your dataset.

### Step 2: Create Configuration

Create a new config file `configs/your_dataset.yaml`:

```yaml
dataset:
  name: "your_dataset"
  data_path: "./data/"
  views:
    your_dataset: ["view1", "view2", "view3"]  # List your views
  label_type: "max_lesion"  # or "lesion" depending on your task

model:
  feature_extractor: "convnext_tiny"
  in_dim: 769
  hidden_dims: [512, 256, 128, 64]
  num_classes: 3  # Your number of classes
  num_views: 3    # Your number of views

training:
  learning_rate: 0.001
  batch_size: 8
  epochs: 100
  optimizer: "adam"
  weight_decay: 0.00001
  early_stopping_patience: 10
  device: "cuda"
  missing_view_rate: 0.0

evaluation:
  missing_view_rates: [0.0, 0.2, 0.5, 0.7, 1.0]
  imputation_methods: ["constant", "learnable", "rag", "covariance"]
  test_sets: ["full_view", "miss_view"]
  metrics: ["accuracy", "auc", "f1_macro"]

utils:
  rag_db_size: 100
  covariance_db_size: 100
```

### Step 3: Update Dataset Loader (if needed)

If your dataset requires special preprocessing, you can extend the `DatasetLoader`:

```python
from giim import DatasetLoader

class CustomDatasetLoader(DatasetLoader):
    def preprocess_image(self, image_path: str):
        # Your custom preprocessing
        image = super().preprocess_image(image_path)
        # Additional processing...
        return image
```

### Step 4: Run Training

```bash
python scripts/train.py --config configs/your_dataset.yaml
```

## Data Statistics and Splits

### Recommended Split Ratios

- **Training:** 70-80%
- **Validation:** 10-15%
- **Testing:** 10-20%

### Important Considerations

1. **Patient-level Splitting:** Always split by patient, not by lesion, to avoid data leakage
2. **Stratification:** Maintain class balance across splits
3. **Missing Views:** For evaluation, simulate missing views only at test time

### Class Balance

If your dataset is imbalanced, consider:

1. **Class Weights:**
```python
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
```

2. **Oversampling/Undersampling:**
```python
from imblearn.over_sampling import SMOTE
# Apply at the patient level
```

3. **Focal Loss:**
```python
# Modify loss function in trainer.py
from torch.nn import BCEWithLogitsLoss
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
```

## Image Preprocessing

### Standard Preprocessing Pipeline

1. **Resize:** 224×224 (ConvNeXt input size)
2. **Normalize:** ImageNet statistics
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
3. **Convert to RGB:** If grayscale, replicate to 3 channels

### Medical Image Specific

For medical images, you may want to:

1. **Window/Level Adjustment:**
```python
def apply_windowing(image, window_center, window_width):
    lower = window_center - window_width // 2
    upper = window_center + window_width // 2
    image = np.clip(image, lower, upper)
    return image
```

2. **Intensity Normalization:**
```python
def normalize_intensity(image):
    # Z-score normalization
    return (image - image.mean()) / image.std()
```

3. **ROI Extraction:**
```python
def extract_roi(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    return roi
```

## Troubleshooting

### Common Issues

1. **Images not found:**
   - Check paths in CSV are relative to dataset root
   - Verify image files exist

2. **Inconsistent image sizes:**
   - Ensure all images are resized to 224×224
   - Check for corrupted images

3. **Class imbalance:**
   - Use class weights or focal loss
   - Consider data augmentation for minority classes

4. **Memory issues:**
   - Reduce batch size
   - Use smaller images
   - Enable gradient checkpointing

### Validation Script

```python
def validate_dataset(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    
    # Check all images exist
    for idx, row in df.iterrows():
        for col in df.columns:
            if '_path' in col:
                image_path = Path(image_dir) / row[col]
                if not image_path.exists():
                    print(f"Missing: {image_path}")
    
    # Check class distribution
    print(df['label'].value_counts())
    
    # Check for duplicate entries
    duplicates = df.duplicated(subset=['patient_id', 'lesion_id'])
    if duplicates.any():
        print(f"Found {duplicates.sum()} duplicates")
```

## Contact

For dataset-specific questions or access requests, please contact:
- Email: [contact email]
- GitHub Issues: [link to issues]

