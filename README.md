# Predicting Poverty in India from Satellite Imagery

Replication and adaptation of Jean et al. (2016) "Combining satellite imagery and machine learning to predict poverty" for the Indian context.

## Table of Contents

1. [Overview](#overview)
2. [Original Paper Summary](#original-paper-summary)
3. [Our Adaptation](#our-adaptation)
4. [Data Sources and Extraction](#data-sources-and-extraction)
5. [Methodology](#methodology)
6. [Key Differences from Original Paper](#key-differences-from-original-paper)
7. [Project Structure](#project-structure)
8. [Installation](#installation)
9. [Usage](#usage)
10. [Results](#results)
11. [Limitations and Ethical Considerations](#limitations-and-ethical-considerations)
12. [References](#references)

---

## Overview

This project replicates the transfer learning methodology from Jean et al. (2016) to predict village-level poverty in India using freely available satellite imagery. The approach addresses **SDG 1 (No Poverty)** by enabling continuous poverty monitoring without expensive ground surveys.

### Problem Statement

Traditional poverty measurement relies on household surveys (e.g., DHS, LSMS) which are:
- Expensive to conduct (USD 1-2 million per country)
- Infrequent (every 3-5 years)
- Limited in geographic coverage

Satellite imagery offers a scalable, low-cost alternative for poverty estimation.

---

## Original Paper Summary

**Paper:** Jean, N., Burke, M., Xie, M., Davis, W. M., Lobell, D. B., and Ermon, S. (2016). [Combining satellite imagery and machine learning to predict poverty](https://www.science.org/doi/abs/10.1126/science.aaf7894). Science, 353(6301), 790-794.

### Key Innovation

The paper introduced a transfer learning approach that uses nighttime lights as a proxy for economic activity:

1. **Phase 1 - Nightlight Classification:** Train a CNN to classify satellite images by nightlight intensity (dim, medium, bright)
2. **Phase 2 - Feature Extraction:** Use the trained CNN as a feature extractor
3. **Phase 3 - Wealth Prediction:** Apply Ridge regression on extracted features to predict DHS wealth index

### Original Results

| Metric | Value |
|--------|-------|
| Countries | Nigeria, Tanzania, Uganda, Malawi, Rwanda |
| Training Images | ~500,000 (random locations) |
| Survey Clusters | ~5,000 (DHS locations) |
| Nightlight Accuracy | ~75% |
| Wealth R-squared | 0.50-0.55 |
| Improvement over NL-only | 12x |

### Why This Works

Nighttime lights correlate with economic activity but cannot directly predict poverty. The CNN learns visual patterns (roads, building density, agricultural development) associated with different nightlight levels. These patterns transfer to poverty prediction because the same features that predict nightlight intensity also predict economic well-being.

---

## Our Adaptation

### Context: India

India presents a compelling case for poverty prediction from satellite imagery:

- **Population:** 1.4 billion people
- **Poverty:** Home to approximately one-third of the global extreme poor
- **Inequality:** Significant regional disparities between urban and rural areas
- **Survey Coverage:** Limited DHS coverage in remote northeastern states

### SDG Alignment

- **SDG 1 (No Poverty):** Enable targeted welfare interventions
- **SDG 10 (Reduced Inequalities):** Identify spatial disparities for policy action
- **SDG 11 (Sustainable Cities):** Monitor urban-rural development gaps

---

## Data Sources and Extraction

### 1. Survey Data

**Demographic and Health Surveys (DHS) India 2019-2021**

- Source: https://dhsprogram.com/
- Coverage: 30,052 geo-located cluster centroids
- Variables: Wealth index (composite score based on assets)
- GPS: Coordinates with random displacement for privacy (urban: 0-2km, rural: 0-5km)

The wealth index is computed by DHS using principal component analysis on household assets including:
- Housing materials (floor, roof, walls)
- Utility access (water, electricity, sanitation)
- Asset ownership (TV, radio, refrigerator, vehicle)

### 2. Satellite Imagery

All imagery was accessed through Google Earth Engine (GEE) and processed into multi-spectral composites.

| Band | Source | Resolution | Description |
|------|--------|------------|-------------|
| 1-3 | Sentinel-2 | 10m | Red, Green, Blue (optical) |
| 4 | Dynamic World | 10m | Built area probability |
| 5 | VIIRS | 500m | Nighttime lights radiance |
| 6-7 | SRTM | 30m | Elevation and slope |

**Note:** ERA5 temperature band was excluded from the final model as it did not improve performance.

### 3. Data Extraction Pipeline

```
DHS Cluster Locations (GPS)
         |
         v
Google Earth Engine
    - Query Sentinel-2, VIIRS, Dynamic World, SRTM
    - Create annual composites (median)
    - Clip to 25.6km x 25.6km tiles
         |
         v
Local Processing
    - Crop 256x256 patches centered on each cluster
    - Normalize per-band (2-98 percentile)
    - Create nightlight labels (tertile binning)
         |
         v
Final Dataset
    - 28,880 valid images
    - 7 bands x 256 x 256 pixels
    - Train/Val/Test split (70/15/15 by cluster)
```

### 4. Nightlight Label Generation

Labels are created from the VIIRS nightlight band:

1. Extract mean nightlight value for each image
2. Compute 33rd and 66th percentiles across all images
3. Assign labels:
   - Class 0 (Dim): Below 33rd percentile
   - Class 1 (Medium): 33rd to 66th percentile
   - Class 2 (Bright): Above 66th percentile

This ensures balanced classes for training.

---

## Methodology

### Phase 1: Nightlight Classification

**Objective:** Train a CNN to predict nightlight intensity class from satellite imagery.

**Architecture:** LightCNN256 - a custom VGG-style network designed for multi-spectral input.

```
Input: (7 channels, 256, 256)
    |
Block 1: Conv(5x5, stride 2) -> Conv(3x3, stride 2)  [256 -> 64]
    |
Block 2: Conv(3x3) -> Pool -> Conv(3x3) -> Pool     [64 -> 16]
    |
Block 3: Conv(3x3) -> Pool -> Conv(3x3) -> Pool     [16 -> 4]
    |
Global Average Pooling
    |
Feature Projection (512 dim)
    |
Classification Head (3 classes)
```

**Training Configuration:**
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Loss: Cross-entropy
- Scheduler: Cosine annealing
- Epochs: 20 (with early stopping, patience=7)
- Batch size: 16

### Phase 2: Feature Extraction

After Phase 1, the trained CNN (without classification head) is used as a feature extractor:

1. Pass each image through the CNN backbone
2. Extract 512-dimensional feature vector from the projection layer
3. These features encode the visual patterns learned during nightlight classification

### Phase 3: Wealth Prediction

**Ridge Regression** is applied to predict wealth scores from extracted features:

```python
ridge = RidgeCV(alphas=np.logspace(-4, 2, 20), cv=5, scoring='r2')
ridge.fit(train_features, train_wealth_labels)
```

Ridge regression is preferred over neural network regression because:
- Fewer parameters reduces overfitting risk
- Works well with limited samples (~20k training images)
- Computationally efficient

---

## Key Differences from Original Paper

| Aspect | Jean et al. (2016) | Our Implementation |
|--------|-------------------|-------------------|
| **Geographic Context** | 5 African countries | India only |
| **Input Bands** | 3 (RGB from Google Static Maps) | 7 (multi-spectral from Sentinel-2, VIIRS, DEM) |
| **Architecture** | VGG-F pretrained on ImageNet (~60M params) | LightCNN256 trained from scratch (~4M params) |
| **Image Source** | Google Static Maps | Google Earth Engine composites |
| **Resolution** | 400x400 pixels | 256x256 pixels |
| **Training Data** | 500k random images for nightlight | 29k DHS cluster images |
| **Survey Data** | ~5k clusters across 5 countries | ~30k clusters in India |

### Why These Modifications?

1. **Multi-spectral input (7 bands):** Unlike the original paper which used only RGB, we include nightlights, built area, and terrain features directly in the input. This provides richer information but prevents using ImageNet pretrained weights.

2. **Lighter architecture:** With 7 input channels, we cannot leverage ImageNet weights. A smaller custom network (4M vs 60M parameters) was designed to reduce overfitting risk given our training data size.

3. **From-scratch training:** Without ImageNet pretraining, we train the entire network on the nightlight classification task. This is suboptimal but necessary for multi-spectral input.

4. **Single country focus:** Focusing on India allows for higher density of training samples and country-specific model optimization.

---

## Project Structure

```
poverty-prediction-india/
|
|-- src/
|   |-- crop_and_process.py       # Extract and preprocess satellite imagery
|   |-- train.py                  # Train CNN and Ridge regression
|   |-- resume_from_checkpoint.py # Resume training from saved checkpoint
|   |-- generate_visualizations.py # Create poster figures
|
|-- data/
|   |-- README.md                 # Data download instructions
|
|-- results/
|   |-- figures/                  # Generated plots
|   |-- models/                   # Saved model weights
|   |-- metrics/                  # Evaluation results
|
|-- docs/
|   |-- poster_content.md         # Poster text content
|
|-- requirements.txt
|-- README.md
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy, Pandas, Matplotlib
- scikit-learn
- rasterio (for GeoTIFF processing)
- tqdm

### Setup

```bash
git clone https://github.com/[your-username]/poverty-prediction-india.git
cd poverty-prediction-india
pip install -r requirements.txt
```

---

## Usage

### 1. Data Preparation

Download DHS India data from https://dhsprogram.com/ (requires registration).

Download satellite composites via Google Earth Engine or use the provided scripts.

### 2. Preprocessing

```bash
python src/crop_and_process.py
```

This script:
- Reads satellite composites and DHS cluster locations
- Crops 256x256 patches for each cluster
- Normalizes bands and creates labels
- Saves processed data as memory-mapped arrays

### 3. Training

```bash
python src/train.py
```

This runs both phases:
- Phase 1: Nightlight classification (CNN training)
- Phase 2: Feature extraction and Ridge regression

### 4. Generate Visualizations

```bash
python src/generate_visualizations.py
```

Creates figures for the poster and analysis.

---

## Results

### Phase 1: Nightlight Classification

| Metric | Value |
|--------|-------|
| Training Accuracy | 92.5% |
| Validation Accuracy | 90.7% |

### Phase 2: Wealth Prediction

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| R-squared | 0.4916 | 0.4595 | 0.4754 |
| MAE | 0.1555 | 0.1590 | 0.1574 |

### Baseline Comparison

| Method | Test R-squared |
|--------|----------------|
| Nightlights only | 0.0155 |
| All bands mean | 0.2104 |
| CNN Transfer (ours) | **0.4754** |

**Improvement: 30.7x over nightlight-only baseline**

---

## Limitations and Ethical Considerations

### Technical Limitations

1. **Urban-rural performance gap:** The model may perform differently in dense urban areas where visual patterns differ significantly from typical DHS sample locations.

2. **Temporal mismatch:** Satellite imagery and survey data may be from different time periods, introducing noise.

3. **GPS displacement:** DHS applies random displacement to protect privacy, which may cause imagery to capture adjacent rather than target areas.

4. **Informal economy:** Economic activity not visible from satellites (informal labor, remittances) cannot be captured.

### Ethical Considerations

1. **Privacy:** High-resolution satellite imagery combined with predictions could enable surveillance or discrimination against identified poor areas.

2. **Bias:** The model learns from existing DHS data which may underrepresent certain populations. Predictions may reinforce existing biases in how poverty is measured.

3. **Dual use:** While intended for welfare targeting, predictions could be misused for discriminatory practices (e.g., redlining for financial services).

4. **Ground truth validity:** The DHS wealth index is a proxy for poverty, not a direct measure. Model errors may be due to limitations in the target variable itself.

### Recommendations

- Validate predictions with ground-truth field surveys before policy implementation
- Use predictions as one input among many for policy decisions, not as sole criterion
- Ensure transparent reporting of model limitations and uncertainty
- Engage local communities in how predictions are used

---

## References

1. Jean, N., Burke, M., Xie, M., Davis, W. M., Lobell, D. B., and Ermon, S. (2016). Combining satellite imagery and machine learning to predict poverty. Science, 353(6301), 790-794.

2. Demographic and Health Surveys Program. https://dhsprogram.com/

3. Google Earth Engine. https://earthengine.google.com/

4. Sentinel-2 Mission. European Space Agency. https://sentinel.esa.int/

5. VIIRS Nighttime Lights. NOAA/NASA. https://ngdc.noaa.gov/eog/viirs/

---

## License

This project is released under the MIT License.

## Acknowledgments

- Original methodology: Jean et al. (2016)
- Survey data: DHS Program
- Satellite imagery: Google Earth Engine, ESA Sentinel, NASA/NOAA VIIRS
- UCL AI for Sustainable Development course
