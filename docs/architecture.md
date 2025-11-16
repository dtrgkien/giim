# GIIM Architecture Documentation

This document provides detailed information about the GIIM (Graph-based Learning of Inter- and Intra-view Dependencies) architecture.

## Table of Contents

1. [Overview](#overview)
2. [Feature Extraction](#feature-extraction)
3. [Graph Construction](#graph-construction)
4. [Graph Neural Network](#graph-neural-network)
5. [Missing View Imputation](#missing-view-imputation)
6. [Classification Head](#classification-head)

## Overview

GIIM is designed to learn from multi-view medical images while robustly handling missing views. The architecture consists of several key components:

```
Input Images → Feature Extraction → Graph Construction → GNN → Classification
                                           ↓
                                  Missing View Imputation
```

## Feature Extraction

### ConvNeXt-Tiny Backbone

We use ConvNeXt-Tiny as the feature extractor for several reasons:

1. **Strong Performance:** ConvNeXt achieves competitive results with Vision Transformers
2. **Efficiency:** Pure convolutional architecture is efficient for medical images
3. **Pretrained Weights:** ImageNet pretraining provides good initialization

### Feature Processing Pipeline

```python
Image (H, W, 3) 
  → ConvNeXt-Tiny 
  → Features (768,) 
  → Add Bias (+1) 
  → Features (769,)
  → L2 Normalize
  → Final Features (769,)
```

**Key Design Decisions:**

- **Bias Term:** The additional bias dimension helps the model distinguish between actual features and imputed features
- **L2 Normalization:** Ensures stable training and better comparison between features from different views
- **No Fine-tuning:** We freeze ConvNeXt weights to prevent overfitting on small medical datasets

## Graph Construction

### Heterogeneous Graph Structure

GIIM constructs a heterogeneous graph for each patient with multiple lesions:

#### Node Types

1. **Tumor Nodes** (`tumor`)
   - Represent each lesion in the patient
   - Features: Concatenation of all view features (769 × num_views)
   - One node per lesion

2. **View Nodes** (`venous`, `arterial`, `delay`, etc.)
   - Represent individual imaging views for each lesion
   - Features: View-specific features (769 dims)
   - num_views nodes per lesion

#### Edge Types

The graph includes four types of edges to capture different dependencies:

1. **Alpha Edges** (`alpha`)
   - Connect: Tumor nodes ↔ View nodes (bidirectional)
   - Purpose: Aggregate view information into tumor representation
   - Structure: Each tumor connects to its corresponding views
   
2. **Beta Edges** (`beta`)
   - Connect: View nodes ↔ View nodes (within same tumor)
   - Purpose: Capture inter-view dependencies for same lesion
   - Structure: Fully connected among views of same tumor

3. **Delta Edges** (`delta`)
   - Connect: Same view type across different tumors
   - Purpose: Learn view-specific patterns across lesions
   - Structure: Fully connected among same view type

4. **Gamma Edges** (`gamma`)
   - Connect: Tumor nodes ↔ Tumor nodes
   - Purpose: Capture patient-level patterns across lesions
   - Structure: Fully connected tumor graph

### Mathematical Formulation

For a patient with N lesions and V views:

- **Total Nodes:** N × (1 + V) = N tumor nodes + N × V view nodes
- **Alpha Edges:** N × V × 2 (bidirectional)
- **Beta Edges:** N × V × (V-1) (complete graph per tumor)
- **Delta Edges:** V × N × (N-1) (complete graph per view type)
- **Gamma Edges:** N × (N-1) (complete tumor graph)

### Example: 2 Tumors, 3 Views

```
Tumor 1 (T1)         Tumor 2 (T2)
    |                    |
  alpha               alpha
    |                    |
+---+---+            +---+---+
|   |   |            |   |   |
V1  V2  V3          V1  V2  V3
|   |   |            |   |   |
+---+---+            +---+---+
    beta                beta

     delta connects V1-V1, V2-V2, V3-V3
     gamma connects T1-T2
```

## Graph Neural Network

### Message Passing Architecture

GIIM uses heterogeneous graph attention networks with separate processing for each edge type:

```python
for layer in layers:
    # Update for each edge type
    for edge_type in ['alpha', 'beta', 'delta', 'gamma']:
        messages = compute_messages(edge_type)
        aggregated = aggregate_messages(messages)
        features = update_features(features, aggregated)
    
    features = layer_norm(features)
    features = dropout(features)
```

### Layer-wise Architecture

**Input:** Node features (769 dims for views, 769×V for tumors)

**Hidden Layers:** [512, 256, 128, 64]

Each layer consists of:
1. Heterogeneous graph convolution
2. Layer normalization
3. ReLU activation
4. Dropout (0.5)

**Output:** 64-dim node embeddings

### Attention Mechanism

For each edge type:

```python
# Compute attention weights
e_ij = LeakyReLU(a^T [W h_i || W h_j])
alpha_ij = softmax_j(e_ij)

# Aggregate with attention
h_i' = sum_j(alpha_ij * W h_j)
```

## Missing View Imputation

GIIM supports four imputation strategies:

### 1. Constant Imputation

**Method:** Replace missing features with zero vector

```python
if view is missing:
    features[view] = zeros(769)
```

**Pros:** Simple, no additional parameters
**Cons:** May lose information about which views are missing

### 2. Learnable Imputation

**Method:** Learn a view-specific embedding vector

```python
# Training
learnable_embeddings = nn.Parameter(torch.randn(num_views, 769))

# Inference
if view is missing:
    features[view] = F.normalize(learnable_embeddings[view_idx])
```

**Pros:** Learns view-specific patterns, lightweight
**Cons:** Same imputation for all patients

### 3. RAG (Retrieval-Augmented Generation)

**Method:** Retrieve similar complete cases from training set

```python
# Build database
database = complete_training_samples

# Imputation
available_features = [features[v] for v in available_views]
query = mean(available_features)

# Find most similar case
similarities = cosine_similarity(query, database)
best_match = database[argmax(similarities)]

# Use matched features
features[missing_view] = best_match[missing_view]
```

**Pros:** Leverages actual patient data, personalized
**Cons:** Requires database storage, slower inference

### 4. Covariance-based Imputation

**Method:** Use covariance structure from training data

```python
# Compute covariance from training set
Sigma = compute_covariance(complete_training_features)

# Imputation (simplified)
available_idx, missing_idx = get_indices()
Sigma_aa = Sigma[available_idx, available_idx]
Sigma_am = Sigma[available_idx, missing_idx]

features[missing] = Sigma_am^T @ Sigma_aa^-1 @ features[available]
```

**Pros:** Statistical approach, principled
**Cons:** Assumes linear relationships, requires matrix operations

## Classification Head

### Final Prediction

```python
# Extract tumor node embeddings
tumor_embeddings = graph['tumor'].x  # (N_tumors, 64)

# Classification
logits = classifier(tumor_embeddings)  # (N_tumors, num_classes)
probabilities = softmax(logits)
```

### Loss Function

**Cross-Entropy Loss** averaged over all tumors:

```python
loss = CrossEntropyLoss(logits, labels)
```

### Multi-Lesion Handling

For patients with multiple lesions:
- Each lesion gets independent prediction
- Patient-level aggregation (if needed): max, mean, or voting

## Training Details

### Optimization

- **Optimizer:** Adam
- **Learning Rate:** 0.001 with cosine annealing
- **Weight Decay:** 1e-5
- **Batch Size:** 8 patients (variable number of lesions per patient)

### Data Augmentation

During training:
- Random horizontal flip (0.5)
- Random rotation (±10°)
- Color jitter (brightness, contrast)

### Training Protocol

1. **Phase 1:** Train on complete views only
   - Epochs: 100
   - No missing view simulation

2. **Phase 2:** Evaluation on missing views
   - Missing rates: [0.0, 0.2, 0.5, 0.7, 1.0]
   - All four imputation methods

### Regularization

- Dropout: 0.5 after each GNN layer
- Weight decay: 1e-5
- Early stopping: patience=10
- Gradient clipping: max_norm=1.0

## Implementation Notes

### PyTorch Geometric

GIIM uses PyTorch Geometric for heterogeneous graphs:

```python
from torch_geometric.data import HeteroData

# Create heterogeneous graph
data = HeteroData()
data['tumor'].x = tumor_features
data['venous'].x = venous_features
data['arterial'].x = arterial_features
data['delay'].x = delay_features

data['tumor', 'alpha', 'venous'].edge_index = alpha_edges
data['tumor', 'gamma', 'tumor'].edge_index = gamma_edges
# ... more edge types
```

### Computational Complexity

For N lesions, V views:

- **Nodes:** O(N × V)
- **Edges:** O(N² × V) (dominated by delta and gamma edges)
- **Forward Pass:** O(N² × V × D) where D is hidden dimension
- **Memory:** O(N² × V) for adjacency matrices

## References

1. ConvNeXt: Liu et al., "A ConvNet for the 2020s", CVPR 2022
2. PyTorch Geometric: Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric", ICLR 2019
3. Graph Attention Networks: Veličković et al., "Graph Attention Networks", ICLR 2018

