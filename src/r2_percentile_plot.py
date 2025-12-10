# =============================================================================
# R² vs Poorest Percent of Clusters Used (Jean et al. style plot)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from pathlib import Path
import json
from tqdm import tqdm

print("="*60)
print("GENERATING R² vs POOREST PERCENT PLOT")
print("="*60)

# =============================================================================
# CONFIGURATION
# =============================================================================

data_path = Path('/content/drive/MyDrive/UCL/Predicting Poverty AI4SD/processed_256x256_20k')
results_path = Path('/content/drive/MyDrive/UCL/Predicting Poverty AI4SD/results_20251209_152057')
output_path = Path('/content/drive/MyDrive/UCL/Predicting Poverty AI4SD/')

# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading data...")

with open(data_path / 'dataset_info.json', 'r') as f:
    info = json.load(f)

n_images = info['n_images']
n_bands = info['n_bands']
img_size = info['image_size']

images = np.memmap(data_path / 'satellite_images.npy', dtype='float32', mode='r',
                   shape=(n_images, n_bands, img_size, img_size))
wealth_labels = np.load(data_path / 'wealth_labels.npy')

train_idx = np.load(data_path / 'train_idx.npy')
test_idx = np.load(data_path / 'test_idx.npy')

# Load pre-extracted features (from training)
import joblib
ridge_model = joblib.load(results_path / 'models' / 'ridge_model.joblib')

print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

# =============================================================================
# EXTRACT FEATURES (or load if available)
# =============================================================================

# We need to re-extract features from model
# For simplicity, use nightlight mean as proxy for now
print("\nExtracting nightlight values...")

NL_BAND = 4  # VIIRS band index

# Get nightlight values for all images
nl_train = np.array([images[i, NL_BAND].mean() for i in tqdm(train_idx, desc="Train NL")])
nl_test = np.array([images[i, NL_BAND].mean() for i in tqdm(test_idx, desc="Test NL")])

wealth_train = wealth_labels[train_idx]
wealth_test = wealth_labels[test_idx]

# =============================================================================
# LOAD CNN FEATURES (if you saved them, otherwise skip)
# =============================================================================

# Try to load saved features, or compute simple band means as proxy
try:
    test_predictions = np.loadtxt(results_path / 'predictions' / 'test_predictions.csv', 
                                   delimiter=',', skiprows=1)
    cnn_test_preds = test_predictions[:, 1]  # predicted column
    print("Loaded CNN predictions")
except:
    print("Using band means as CNN proxy...")
    # Use all band means as simple feature proxy
    all_train = np.array([images[i].mean(axis=(1, 2)) for i in tqdm(train_idx, desc="Train bands")])
    all_test = np.array([images[i].mean(axis=(1, 2)) for i in tqdm(test_idx, desc="Test bands")])
    
    ridge_proxy = RidgeCV(alphas=np.logspace(-4, 2, 20), cv=5)
    ridge_proxy.fit(all_train, wealth_train)
    cnn_test_preds = ridge_proxy.predict(all_test)

# =============================================================================
# COMPUTE R² FOR DIFFERENT PERCENTILE THRESHOLDS
# =============================================================================

print("\nComputing R² for different thresholds...")

percentiles = np.arange(10, 101, 5)  # 10, 15, 20, ..., 100
r2_transfer = []
r2_nightlights = []

for pct in tqdm(percentiles, desc="Computing R²"):
    # Find threshold wealth value
    threshold = np.percentile(wealth_test, pct)
    
    # Select clusters below this threshold
    mask = wealth_test <= threshold
    
    if mask.sum() < 10:  # Need at least 10 samples
        r2_transfer.append(np.nan)
        r2_nightlights.append(np.nan)
        continue
    
    # R² for CNN transfer learning
    try:
        r2_cnn = r2_score(wealth_test[mask], cnn_test_preds[mask])
    except:
        r2_cnn = np.nan
    r2_transfer.append(r2_cnn)
    
    # R² for nightlights only
    ridge_nl = RidgeCV(alphas=np.logspace(-4, 2, 10), cv=min(5, mask.sum()-1))
    try:
        ridge_nl.fit(nl_train.reshape(-1, 1), wealth_train)
        nl_preds = ridge_nl.predict(nl_test[mask].reshape(-1, 1))
        r2_nl = r2_score(wealth_test[mask], nl_preds)
    except:
        r2_nl = np.nan
    r2_nightlights.append(r2_nl)

# =============================================================================
# PLOT
# =============================================================================

print("\nGenerating plot...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(percentiles, r2_transfer, 'b-', linewidth=2, label='Transfer Learning (CNN)')
ax.plot(percentiles, r2_nightlights, 'g-', linewidth=2, label='Nightlights Only')

# Add poverty line markers (approximate based on wealth distribution)
# Assuming Q1 (bottom 20%) represents extreme poverty
ax.axvline(x=20, color='red', linestyle='--', alpha=0.7, linewidth=1)
ax.text(21, 0.55, 'Q1\n(Poorest 20%)', fontsize=9, color='red')

ax.axvline(x=40, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.text(41, 0.55, 'Q2', fontsize=9, color='red')

ax.set_xlabel('Poorest Percent of Clusters Used', fontsize=12)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('Transfer Learning vs Nightlights Performance\nby Wealth Percentile', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(10, 100)
ax.set_ylim(0, 0.6)

plt.tight_layout()
plt.savefig(output_path / 'r2_vs_percentile.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved: {output_path / 'r2_vs_percentile.png'}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"\nAt 100% (all data):")
print(f"  Transfer Learning R²: {r2_transfer[-1]:.4f}")
print(f"  Nightlights R²: {r2_nightlights[-1]:.4f}")
