# =============================================================================
# WEALTH QUINTILE COMPARISON - Circular cluster images (Cloud-free selection)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
from pathlib import Path

print("="*60)
print("WEALTH QUINTILE COMPARISON - CLOUD-FREE CLUSTERS")
print("="*60)

# =============================================================================
# CONFIGURATION - Update for Colab
# =============================================================================

data_path = Path('/content/drive/MyDrive/UCL/Predicting Poverty AI4SD/processed_256x256_20k')
output_path = Path('/content/drive/MyDrive/UCL/Predicting Poverty AI4SD/quintile_images')
output_path.mkdir(exist_ok=True, parents=True)

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

print(f"Images: {n_images}, Bands: {n_bands}")

# =============================================================================
# IMAGE QUALITY SCORING (to avoid clouds)
# =============================================================================

def score_image_quality(img):
    """
    Score image quality - higher = clearer/less cloudy
    Cloud-covered images tend to have:
    - Low variance (white everywhere)
    - High blue band values
    - Low contrast
    """
    rgb = img[:3]
    
    # Variance in RGB (cloudy = low variance)
    rgb_var = np.var(rgb, axis=(1, 2)).mean()
    
    # Check for cloud-like pixels (high bright values in all bands)
    bright_pixels = np.mean((rgb > 0.8).all(axis=0))  # % of very bright pixels
    
    # Edge strength (clear images have more edges)
    gray = rgb.mean(axis=0)
    dx = np.abs(np.diff(gray, axis=1)).mean()
    dy = np.abs(np.diff(gray, axis=0)).mean()
    edge_strength = dx + dy
    
    # Combine scores (higher = better)
    score = rgb_var * 10 + edge_strength * 5 - bright_pixels * 3
    
    return score

# =============================================================================
# CREATE CIRCULAR MASK
# =============================================================================

def apply_circular_mask(img_data, radius_ratio=0.48):
    """Apply circular mask to image, making outside transparent"""
    h, w = img_data.shape[:2]
    center = (h // 2, w // 2)
    radius = int(min(h, w) * radius_ratio)
    
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = dist <= radius
    
    if len(img_data.shape) == 3:
        rgba = np.zeros((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = img_data
        rgba[:, :, 3] = mask.astype(np.float32)
        return rgba
    else:
        masked = np.copy(img_data)
        masked[~mask] = np.nan
        return masked, mask

# =============================================================================
# SELECT BEST (CLOUD-FREE) CLUSTER PER QUINTILE
# =============================================================================

print("\nSelecting clearest clusters from each quintile...")

sorted_idx = np.argsort(wealth_labels)
quintile_size = len(sorted_idx) // 5

quintiles = ['Q1_Poorest', 'Q2', 'Q3', 'Q4', 'Q5_Richest']
quintile_labels = ['Q1 (Poorest)', 'Q2', 'Q3', 'Q4', 'Q5 (Richest)']
selected = []

for q in range(5):
    start = q * quintile_size
    end = (q + 1) * quintile_size if q < 4 else len(sorted_idx)
    q_indices = sorted_idx[start:end]
    
    # Score all images in this quintile for quality
    print(f"\n  Scoring {quintile_labels[q]}...")
    
    # Sample up to 200 images for scoring (for speed)
    sample_size = min(200, len(q_indices))
    sample_indices = np.random.choice(q_indices, sample_size, replace=False)
    
    scores = []
    for idx in sample_indices:
        score = score_image_quality(images[idx])
        scores.append((idx, score))
    
    # Sort by quality score (highest = clearest)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select the best one
    best_idx = scores[0][0]
    best_score = scores[0][1]
    wealth = wealth_labels[best_idx]
    
    selected.append({
        'idx': best_idx, 
        'wealth': wealth, 
        'quintile': quintiles[q], 
        'label': quintile_labels[q],
        'quality_score': best_score
    })
    
    print(f"    Selected idx={best_idx}, wealth={wealth:.4f}, quality={best_score:.3f}")
    print(f"    (Best of {sample_size} samples)")

colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#1a9850']

# =============================================================================
# SAVE INDIVIDUAL IMAGES
# =============================================================================

print("\n\nSaving individual images...")

bands_to_show = {
    'RGB': None,
    'Built': 3,
    'Nightlights': 4,
    'Elevation': 5,
    'Slope': 6
}
cmaps = {'Built': 'Oranges', 'Nightlights': 'hot', 'Elevation': 'terrain', 'Slope': 'YlOrBr'}

for sample in selected:
    img = images[sample['idx']]
    q_name = sample['quintile']
    wealth = sample['wealth']
    
    # Save RGB
    rgb = np.stack([img[0], img[1], img[2]], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    rgba = apply_circular_mask(rgb)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(rgba)
    ax.axis('off')
    ax.set_title(f"{sample['label']}\nWealth: {wealth:.3f}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / f'{q_name}_RGB.png', dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    
    # Save each band
    for band_name, band_idx in bands_to_show.items():
        if band_idx is None:
            continue
            
        band_data = img[band_idx]
        masked_data, mask = apply_circular_mask(band_data)
        
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(masked_data, cmap=cmaps[band_name])
        ax.axis('off')
        ax.set_title(f"{sample['label']} - {band_name}\nWealth: {wealth:.3f}", fontsize=10, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / f'{q_name}_{band_name}.png', dpi=150, bbox_inches='tight', transparent=True)
        plt.close()

print(f"  Saved to: {output_path}")

# =============================================================================
# COMBINED PLOT: RGB ROW
# =============================================================================

print("\nGenerating RGB comparison row...")

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('Satellite Clusters by Wealth Quintile (Cloud-Free)', fontsize=16, fontweight='bold', y=1.05)

for i, (ax, sample) in enumerate(zip(axes, selected)):
    img = images[sample['idx']]
    rgb = np.stack([img[0], img[1], img[2]], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    rgba = apply_circular_mask(rgb)
    
    ax.imshow(rgba)
    ax.set_title(f"{sample['label']}\nWealth: {sample['wealth']:.3f}", 
                 fontsize=12, fontweight='bold', color=colors[i])
    ax.axis('off')

plt.tight_layout()
plt.savefig(output_path / 'quintile_rgb_row.png', dpi=150, bbox_inches='tight', transparent=True)
plt.show()

# =============================================================================
# COMBINED PLOT: ALL BANDS GRID
# =============================================================================

print("\nGenerating all bands comparison grid...")

band_names = ['RGB', 'Built', 'Nightlights', 'Elevation', 'Slope']

fig, axes = plt.subplots(5, 5, figsize=(18, 18))
fig.suptitle('Cluster Images by Wealth Quintile (Cloud-Free)', fontsize=18, fontweight='bold', y=1.01)

for row, sample in enumerate(selected):
    img = images[sample['idx']]
    
    # RGB
    rgb = np.stack([img[0], img[1], img[2]], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    rgba = apply_circular_mask(rgb)
    axes[row, 0].imshow(rgba)
    axes[row, 0].set_ylabel(f"{sample['label']}\n(W={sample['wealth']:.2f})", 
                            fontsize=11, fontweight='bold', color=colors[row])
    if row == 0:
        axes[row, 0].set_title('RGB', fontsize=12, fontweight='bold')
    axes[row, 0].axis('off')
    
    # Other bands
    for col, band_name in enumerate(['Built', 'Nightlights', 'Elevation', 'Slope']):
        band_idx = bands_to_show[band_name]
        band_data = img[band_idx]
        masked_data, _ = apply_circular_mask(band_data)
        
        axes[row, col+1].imshow(masked_data, cmap=cmaps[band_name])
        if row == 0:
            axes[row, col+1].set_title(band_name, fontsize=12, fontweight='bold')
        axes[row, col+1].axis('off')

plt.tight_layout()
plt.savefig(output_path / 'quintile_all_bands_grid.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"\nSelected clusters (best quality from each quintile):")
for sample in selected:
    print(f"  {sample['label']}: idx={sample['idx']}, wealth={sample['wealth']:.3f}, quality={sample['quality_score']:.2f}")
print(f"\nSaved to: {output_path}")
