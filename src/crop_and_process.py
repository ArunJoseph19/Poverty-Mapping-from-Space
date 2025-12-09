# =============================================================================
# CROP AND PROCESS SATELLITE IMAGERY
# Extracts 256x256 patches from composites and prepares training data
# =============================================================================

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
import re
import gc
import shutil
from collections import defaultdict

print("="*70)
print("CROP AND PROCESS SATELLITE IMAGERY")
print("="*70)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Update these paths for your system
BASE_PATH = Path('./data')
COMPOSITES_FOLDER = BASE_PATH / 'DHS_India_Full_Composites'
CLUSTER_CSV = BASE_PATH / 'cluster_wealth_gps_valid.csv'
OUTPUT_PATH = BASE_PATH / 'processed'

OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

# Processing settings
CROP_SIZE = 256
PIXEL_RESOLUTION = 100  # meters per pixel
YEARS = [2021]  # Years to process
VIIRS_BAND_IDX = 4  # Index of nightlight band
N_BANDS = 7  # Number of bands (excluding ERA5 temp)

# Checkpoint
CHECKPOINT_FILE = OUTPUT_PATH / 'progress.json'
BATCH_SIZE = 1000

print(f"\nComposites: {COMPOSITES_FOLDER}")
print(f"Output: {OUTPUT_PATH}")
print(f"Crop size: {CROP_SIZE}x{CROP_SIZE}")

# =============================================================================
# LOAD OR CREATE CHECKPOINT
# =============================================================================

def load_progress():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {
        'processed': [],
        'current_idx': 0,
        'metadata': [],
        'nightlight_values': [],
        'n_bands': None
    }

def save_progress(prog):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(prog, f)

progress = load_progress()
print(f"Checkpoint: {len(progress['processed'])} already done")

# =============================================================================
# LOAD CLUSTERS
# =============================================================================

print("\nLoading clusters...")

df = pd.read_csv(CLUSTER_CSV)
df['wealth_quintile'] = pd.qcut(df['wealth_score'], q=5, labels=['Q1','Q2','Q3','Q4','Q5'])
df['cluster_idx'] = range(len(df))

print(f"Total clusters: {len(df):,}")
for q in ['Q1','Q2','Q3','Q4','Q5']:
    print(f"  {q}: {len(df[df['wealth_quintile']==q])}")

df.to_csv(OUTPUT_PATH / 'all_clusters.csv', index=False)

# =============================================================================
# BUILD TASK LIST
# =============================================================================

print("\nBuilding task list...")

all_tasks = []
for _, row in df.iterrows():
    for year in YEARS:
        task_key = f"{row['cluster_idx']}_{year}"
        if task_key not in progress['processed']:
            all_tasks.append({
                'cluster_idx': row['cluster_idx'],
                'year': year,
                'lat': row['LATNUM'],
                'lon': row['LONGNUM'],
                'wealth_score': row['wealth_score'],
                'wealth_quintile': row['wealth_quintile'],
                'task_key': task_key
            })

total_expected = len(df) * len(YEARS)
print(f"  Total expected: {total_expected:,}")
print(f"  Already done: {len(progress['processed']):,}")
print(f"  Remaining: {len(all_tasks):,}")

if len(all_tasks) == 0:
    print("\nAll tasks already completed!")
else:
    # =============================================================================
    # INDEX COMPOSITE TILES
    # =============================================================================
    
    print("\nIndexing composite tiles...")
    
    tile_files = list(COMPOSITES_FOLDER.glob('india_full_*.tif'))
    print(f"  Found {len(tile_files)} tiles")
    
    tiles_by_year = defaultdict(list)
    for f in tile_files:
        match = re.search(r'india_full_(\d{4})', f.name)
        if match:
            year = int(match.group(1))
            tiles_by_year[year].append(f)
    
    tile_index = {}
    for year in YEARS:
        if year in tiles_by_year:
            tile_index[year] = []
            for tile_path in tiles_by_year[year]:
                with rasterio.open(tile_path) as src:
                    bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                    n_bands = src.count
                    tile_index[year].append((tile_path, bounds))
            print(f"  {year}: {len(tile_index[year])} tiles, {n_bands} bands")
    
    n_bands = N_BANDS
    progress['n_bands'] = n_bands
    save_progress(progress)
    
    # =============================================================================
    # CREATE MEMMAP
    # =============================================================================
    
    total_images = total_expected
    MEMMAP_PATH = OUTPUT_PATH / 'images_temp.npy'
    
    if MEMMAP_PATH.exists():
        print(f"\nOpening existing memmap...")
        images = np.memmap(MEMMAP_PATH, dtype='float32', mode='r+',
                          shape=(total_images, n_bands, CROP_SIZE, CROP_SIZE))
    else:
        print(f"\nCreating new memmap...")
        images = np.memmap(MEMMAP_PATH, dtype='float32', mode='w+',
                          shape=(total_images, n_bands, CROP_SIZE, CROP_SIZE))
    
    print(f"  Shape: {images.shape}")
    print(f"  Size: {images.nbytes / (1024**3):.1f} GB")
    
    # =============================================================================
    # HELPER FUNCTIONS
    # =============================================================================
    
    def point_in_bounds(lon, lat, bounds):
        left, bottom, right, top = bounds
        return left <= lon <= right and bottom <= lat <= top
    
    def crop_and_process(tile_path, lat, lon):
        """Crop from tile and normalize"""
        try:
            with rasterio.open(tile_path) as src:
                if not point_in_bounds(lon, lat, (src.bounds.left, src.bounds.bottom, 
                                                   src.bounds.right, src.bounds.top)):
                    return None, None
                
                row, col = src.index(lon, lat)
                half = CROP_SIZE // 2
                
                if (row - half < 0 or col - half < 0 or 
                    row + half > src.height or col + half > src.width):
                    return None, None
                
                window = Window(col - half, row - half, CROP_SIZE, CROP_SIZE)
                data = src.read(window=window).astype(np.float32)
                
                # Keep only first N_BANDS
                if data.shape[0] > N_BANDS:
                    data = data[:N_BANDS]
                
                if np.all(data == 0) or np.all(np.isnan(data)):
                    return None, None
                
                # Get raw nightlight value
                nl_value = float(np.nanmean(data[VIIRS_BAND_IDX]))
                
                # Handle NaN
                for b in range(data.shape[0]):
                    band = data[b]
                    if np.isnan(band).any():
                        median = np.nanmedian(band)
                        if np.isnan(median):
                            median = 0.0
                        band[np.isnan(band)] = median
                        data[b] = band
                
                # Normalize (2-98 percentile)
                for b in range(data.shape[0]):
                    band = data[b]
                    p2, p98 = np.percentile(band, [2, 98])
                    if p98 - p2 > 1e-6:
                        data[b] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)
                    else:
                        data[b] = 0.5
                
                return data, nl_value
                
        except Exception as e:
            return None, None
    
    # =============================================================================
    # PROCESS ALL TASKS
    # =============================================================================
    
    print(f"\nProcessing {len(all_tasks):,} images...")
    
    success = 0
    failed = 0
    current_idx = progress['current_idx']
    
    for i, task in enumerate(tqdm(all_tasks, desc="Processing")):
        year = task['year']
        
        if year not in tile_index:
            failed += 1
            continue
        
        data = None
        for tile_path, bounds in tile_index[year]:
            if point_in_bounds(task['lon'], task['lat'], bounds):
                data, nl_value = crop_and_process(tile_path, task['lat'], task['lon'])
                if data is not None:
                    break
        
        if data is not None:
            images[current_idx] = data
            
            progress['metadata'].append({
                'idx': current_idx,
                'cluster_idx': task['cluster_idx'],
                'year': year,
                'wealth_score': task['wealth_score'],
                'wealth_quintile': task['wealth_quintile']
            })
            progress['nightlight_values'].append(nl_value)
            progress['processed'].append(task['task_key'])
            progress['current_idx'] = current_idx + 1
            
            current_idx += 1
            success += 1
        else:
            failed += 1
        
        if (i + 1) % BATCH_SIZE == 0:
            images.flush()
            save_progress(progress)
            print(f"\n  Checkpoint saved ({success:,} done)...")
    
    images.flush()
    save_progress(progress)
    
    print(f"\nProcessing complete!")
    print(f"  Success: {success:,}")
    print(f"  Failed: {failed:,}")

# =============================================================================
# FINALIZE
# =============================================================================

print("\n" + "="*70)
print("FINALIZING")
print("="*70)

progress = load_progress()
n_images = len(progress['metadata'])
n_bands = progress['n_bands']

print(f"\nTotal images: {n_images:,}")

if n_images > 0:
    # Create nightlight labels
    print("\nCreating nightlight labels...")
    nl_raw = np.array(progress['nightlight_values'])
    p33 = np.percentile(nl_raw, 33)
    p66 = np.percentile(nl_raw, 66)
    
    nl_labels = np.zeros(n_images, dtype=np.int64)
    nl_labels[nl_raw >= p33] = 1
    nl_labels[nl_raw >= p66] = 2
    
    class_counts = np.bincount(nl_labels, minlength=3)
    print(f"  Dim: {class_counts[0]:,}, Medium: {class_counts[1]:,}, Bright: {class_counts[2]:,}")
    
    # Create wealth labels
    print("\nCreating wealth labels...")
    meta_df = pd.DataFrame(progress['metadata'])
    wealth_labels = meta_df['wealth_score'].values.astype(np.float32)
    
    # Train/val/test split
    print("\nCreating splits...")
    unique_clusters = meta_df.groupby('cluster_idx').first().reset_index()
    
    train_c, temp_c = train_test_split(unique_clusters, test_size=0.3, 
                                        stratify=unique_clusters['wealth_quintile'], random_state=42)
    val_c, test_c = train_test_split(temp_c, test_size=0.5,
                                      stratify=temp_c['wealth_quintile'], random_state=42)
    
    train_idx = meta_df[meta_df['cluster_idx'].isin(train_c['cluster_idx'])]['idx'].values
    val_idx = meta_df[meta_df['cluster_idx'].isin(val_c['cluster_idx'])]['idx'].values
    test_idx = meta_df[meta_df['cluster_idx'].isin(test_c['cluster_idx'])]['idx'].values
    
    print(f"  Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")
    
    # Save files
    print("\nSaving final files...")
    
    MEMMAP_PATH = OUTPUT_PATH / 'images_temp.npy'
    final_images = OUTPUT_PATH / 'satellite_images.npy'
    if MEMMAP_PATH.exists():
        shutil.move(str(MEMMAP_PATH), str(final_images))
    print(f"  satellite_images.npy")
    
    np.save(OUTPUT_PATH / 'wealth_labels.npy', wealth_labels)
    np.save(OUTPUT_PATH / 'nightlight_labels.npy', nl_labels)
    np.save(OUTPUT_PATH / 'train_idx.npy', train_idx)
    np.save(OUTPUT_PATH / 'val_idx.npy', val_idx)
    np.save(OUTPUT_PATH / 'test_idx.npy', test_idx)
    
    info = {
        'n_images': n_images,
        'n_bands': n_bands,
        'image_size': CROP_SIZE,
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx)
    }
    with open(OUTPUT_PATH / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    nl_info = {
        'p33': float(p33),
        'p66': float(p66),
        'class_0': int(class_counts[0]),
        'class_1': int(class_counts[1]),
        'class_2': int(class_counts[2])
    }
    with open(OUTPUT_PATH / 'nightlight_thresholds.json', 'w') as f:
        json.dump(nl_info, f, indent=2)
    
    meta_df.to_csv(OUTPUT_PATH / 'metadata.csv', index=False)
    
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"  satellite_images.npy ({n_images:,} x {n_bands} x {CROP_SIZE} x {CROP_SIZE})")
    print(f"  wealth_labels.npy")
    print(f"  nightlight_labels.npy")
    print(f"  train_idx.npy, val_idx.npy, test_idx.npy")
