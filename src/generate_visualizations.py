# =============================================================================
# GENERATE POSTER VISUALIZATIONS
# Creates figures for academic poster: nightlight comparison, bands, etc.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

print("="*60)
print("GENERATING POSTER VISUALIZATIONS")
print("="*60)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Update these paths for your system
data_path = Path('./data/processed')
output_path = Path('./results/figures')
output_path.mkdir(exist_ok=True, parents=True)

# Load data info
with open(data_path / 'dataset_info.json', 'r') as f:
    info = json.load(f)

n_images = info['n_images']
n_bands = info['n_bands']
img_size = info['image_size']

print(f"Images: {n_images}, Bands: {n_bands}, Size: {img_size}x{img_size}")

# Load as memmap
images = np.memmap(data_path / 'satellite_images.npy', dtype='float32', mode='r',
                   shape=(n_images, n_bands, img_size, img_size))
wealth_labels = np.load(data_path / 'wealth_labels.npy')
nightlight_labels = np.load(data_path / 'nightlight_labels.npy')

print("Data loaded")

BAND_NAMES = ['Red', 'Green', 'Blue', 'Built Area', 'Nightlights', 'Elevation', 'Slope']

# =============================================================================
# 1. NIGHTLIGHT COMPARISON
# =============================================================================

print("\nGenerating nightlight comparison...")

dim_idx = np.where(nightlight_labels == 0)[0]
medium_idx = np.where(nightlight_labels == 1)[0]
bright_idx = np.where(nightlight_labels == 2)[0]

np.random.seed(42)
examples = {
    'Dim (Class 0)': np.random.choice(dim_idx, 3, replace=False),
    'Medium (Class 1)': np.random.choice(medium_idx, 3, replace=False),
    'Bright (Class 2)': np.random.choice(bright_idx, 3, replace=False),
}

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Nightlight Classification: Example Images', fontsize=16, fontweight='bold')

for row, (label, indices) in enumerate(examples.items()):
    for col, idx in enumerate(indices):
        img = images[idx]
        rgb = np.stack([img[0], img[1], img[2]], axis=-1)
        rgb = np.clip(rgb, 0, 1)
        axes[row, col].imshow(rgb)
        axes[row, col].set_title(f'{label}\nWealth: {wealth_labels[idx]:.2f}', fontsize=10)
        axes[row, col].axis('off')
    
    nl_band = images[indices[0], 4]
    im = axes[row, 3].imshow(nl_band, cmap='hot')
    axes[row, 3].set_title('Nightlight Band', fontsize=10)
    axes[row, 3].axis('off')
    plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(output_path / 'nightlight_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  nightlight_comparison.png saved")

# =============================================================================
# 2. ALL BANDS VISUALIZATION
# =============================================================================

print("\nGenerating band visualization...")

mid_wealth_idx = np.argsort(wealth_labels)[n_images // 2]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Multi-Spectral Satellite Bands (256x256 pixels)', fontsize=16, fontweight='bold')

img = images[mid_wealth_idx]

rgb = np.stack([img[0], img[1], img[2]], axis=-1)
rgb = np.clip(rgb, 0, 1)
axes[0, 0].imshow(rgb)
axes[0, 0].set_title('RGB Composite', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

cmaps = ['Reds', 'Greens', 'Blues', 'Oranges', 'hot', 'terrain', 'YlOrBr']
positions = [(0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)]

for i, (pos, name, cmap) in enumerate(zip(positions, BAND_NAMES, cmaps)):
    im = axes[pos].imshow(img[i], cmap=cmap)
    axes[pos].set_title(f'Band {i+1}: {name}', fontsize=11)
    axes[pos].axis('off')
    plt.colorbar(im, ax=axes[pos], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(output_path / 'all_bands.png', dpi=150, bbox_inches='tight')
plt.close()
print("  all_bands.png saved")

# =============================================================================
# 3. WEALTH QUINTILE EXAMPLES
# =============================================================================

print("\nGenerating wealth quintile examples...")

sorted_idx = np.argsort(wealth_labels)
quintile_size = len(sorted_idx) // 5

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
fig.suptitle('Satellite Images by Wealth Quintile', fontsize=16, fontweight='bold')

quintile_names = ['Q1 (Poorest)', 'Q2', 'Q3', 'Q4', 'Q5 (Richest)']
quintile_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']

for q in range(5):
    start = q * quintile_size
    end = (q + 1) * quintile_size if q < 4 else len(sorted_idx)
    q_indices = sorted_idx[start:end]
    
    example_idx = q_indices[len(q_indices) // 2]
    
    img = images[example_idx]
    
    rgb = np.stack([img[0], img[1], img[2]], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    axes[0, q].imshow(rgb)
    axes[0, q].set_title(f'{quintile_names[q]}\nWealth: {wealth_labels[example_idx]:.2f}', 
                         fontsize=11, color=quintile_colors[q], fontweight='bold')
    axes[0, q].axis('off')
    
    axes[1, q].imshow(img[4], cmap='hot')
    axes[1, q].set_title('Nightlight Band', fontsize=10)
    axes[1, q].axis('off')

plt.tight_layout()
plt.savefig(output_path / 'wealth_quintiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("  wealth_quintiles.png saved")

# =============================================================================
# 4. DISTRIBUTIONS
# =============================================================================

print("\nGenerating distribution plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(wealth_labels, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Wealth Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Wealth Score Distribution', fontsize=14, fontweight='bold')
axes[0].axvline(np.median(wealth_labels), color='red', linestyle='--', lw=2, 
                label=f'Median: {np.median(wealth_labels):.2f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

class_counts = [np.sum(nightlight_labels == i) for i in range(3)]
class_names = ['Dim (0)', 'Medium (1)', 'Bright (2)']
colors = ['#2c3e50', '#f39c12', '#e74c3c']
bars = axes[1].bar(class_names, class_counts, color=colors, edgecolor='black')
axes[1].set_ylabel('Number of Images', fontsize=12)
axes[1].set_title('Nightlight Class Distribution', fontsize=14, fontweight='bold')
for bar, count in zip(bars, class_counts):
    axes[1].text(bar.get_x() + bar.get_width()/2, count + 100, f'{count:,}', 
                 ha='center', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_path / 'distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  distributions.png saved")

# =============================================================================
# DONE
# =============================================================================

print("\n" + "="*60)
print("ALL VISUALIZATIONS SAVED")
print("="*60)
print(f"\nOutput folder: {output_path}")
print("  nightlight_comparison.png")
print("  all_bands.png")
print("  wealth_quintiles.png")
print("  distributions.png")
