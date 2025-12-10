# =============================================================================
# INTERACTIVE MAP: DHS Clusters + ALL Satellite TIF Tiles
# Renders all 9 GeoTIFF tiles for 2021 as map overlays
# =============================================================================

# Install if needed
# !pip install folium rasterio

import folium
from folium import raster_layers
import pandas as pd
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

print("="*60)
print("GENERATING INTERACTIVE MAP WITH ALL TIF TILES")
print("="*60)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH = Path('/content/drive/MyDrive/UCL/Predicting Poverty AI4SD/')
COMPOSITES_FOLDER = Path('/content/drive/MyDrive/DHS_India_Full_Composites')
CLUSTER_CSV = BASE_PATH / 'cluster_wealth_gps_valid.csv'
OUTPUT_HTML = BASE_PATH / 'satellite_bands_map.html'

# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading cluster data...")
df = pd.read_csv(CLUSTER_CSV)
df['wealth_quintile'] = pd.qcut(df['wealth_score'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
print(f"Total clusters: {len(df):,}")

# Find ALL TIF files for 2021
print("\nFinding GeoTIFF files...")
tif_files = sorted(list(COMPOSITES_FOLDER.glob('*2021*.tif')))
print(f"Found {len(tif_files)} TIF files for 2021:")
for f in tif_files:
    print(f"  - {f.name}")

# =============================================================================
# HELPER: Create RGB overlay from a single TIF
# =============================================================================

def create_rgb_overlay(tif_path, max_size=1500):
    """Create RGB overlay from a TIF file"""
    try:
        with rasterio.open(tif_path) as src:
            # Get bounds in WGS84
            bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            
            # Downsample for performance
            scale = min(1, max_size / max(src.width, src.height))
            out_shape = (max(1, int(src.height * scale)), max(1, int(src.width * scale)))
            
            # Read RGB bands
            r = src.read(1, out_shape=out_shape).astype(np.float32)
            g = src.read(2, out_shape=out_shape).astype(np.float32)
            b = src.read(3, out_shape=out_shape).astype(np.float32)
            
            # Normalize each band
            for band in [r, g, b]:
                valid = band[band > 0]
                if len(valid) > 0:
                    p2, p98 = np.percentile(valid, [2, 98])
                    if p98 - p2 > 0:
                        band[:] = np.clip((band - p2) / (p98 - p2), 0, 1)
                    else:
                        band[:] = 0
            
            # Create RGBA
            rgb = np.stack([r, g, b], axis=-1)
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            
            # Make no-data areas transparent
            alpha = np.where((r == 0) & (g == 0) & (b == 0), 0, 255).astype(np.uint8)
            rgba = np.dstack([rgb, alpha])
            
            # Convert to base64 PNG
            img = Image.fromarray(rgba, 'RGBA')
            buffer = BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode()
            
            # Folium bounds: [[south, west], [north, east]]
            folium_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
            
            return folium_bounds, f"data:image/png;base64,{b64}"
    
    except Exception as e:
        print(f"  Error: {e}")
        return None, None

# =============================================================================
# HELPER: Create single band overlay
# =============================================================================

def create_band_overlay(tif_path, band_idx, cmap_name, max_size=1500):
    """Create colored overlay for a single band"""
    try:
        with rasterio.open(tif_path) as src:
            bounds = transform_bounds(src.crs, 'EPSG:4326', *src.bounds)
            
            scale = min(1, max_size / max(src.width, src.height))
            out_shape = (max(1, int(src.height * scale)), max(1, int(src.width * scale)))
            
            data = src.read(band_idx, out_shape=out_shape).astype(np.float32)
            data = np.nan_to_num(data, nan=0)
            
            valid = data[data > 0]
            if len(valid) > 0:
                p2, p98 = np.percentile(valid, [2, 98])
                if p98 - p2 > 0:
                    data = np.clip((data - p2) / (p98 - p2), 0, 1)
            
            # Apply colormap
            cmap = plt.get_cmap(cmap_name)
            colored = (cmap(data) * 255).astype(np.uint8)
            colored[data == 0, 3] = 0  # Transparent where no data
            
            img = Image.fromarray(colored, 'RGBA')
            buffer = BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode()
            
            folium_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
            return folium_bounds, f"data:image/png;base64,{b64}"
    
    except Exception as e:
        print(f"  Error: {e}")
        return None, None

# =============================================================================
# CREATE MAP
# =============================================================================

print("\nCreating map...")

center_lat = df['LATNUM'].mean()
center_lon = df['LONGNUM'].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=5,
    tiles='cartodbpositron'
)

# Base layers
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Esri Satellite'
).add_to(m)

folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)

# =============================================================================
# ADD ALL RGB TILES AS ONE LAYER
# =============================================================================

print("\nProcessing all TIF files for RGB layer...")

rgb_group = folium.FeatureGroup(name='RGB Composites (All Tiles)', show=True)

for i, tif_path in enumerate(tif_files):
    print(f"  [{i+1}/{len(tif_files)}] {tif_path.name}...")
    bounds, b64_img = create_rgb_overlay(tif_path)
    
    if bounds and b64_img:
        folium.raster_layers.ImageOverlay(
            image=b64_img,
            bounds=bounds,
            opacity=0.85,
            interactive=True,
            cross_origin=False
        ).add_to(rgb_group)
        print(f"       Added to map")

rgb_group.add_to(m)

# =============================================================================
# ADD NIGHTLIGHT TILES AS ONE LAYER
# =============================================================================

print("\nProcessing all TIF files for Nightlight layer...")

nl_group = folium.FeatureGroup(name='Nightlights (All Tiles)', show=False)

for i, tif_path in enumerate(tif_files):
    print(f"  [{i+1}/{len(tif_files)}] {tif_path.name}...")
    bounds, b64_img = create_band_overlay(tif_path, band_idx=5, cmap_name='hot')  # Band 5 = VIIRS
    
    if bounds and b64_img:
        folium.raster_layers.ImageOverlay(
            image=b64_img,
            bounds=bounds,
            opacity=0.85,
            interactive=True,
            cross_origin=False
        ).add_to(nl_group)

nl_group.add_to(m)

# =============================================================================
# ADD CLUSTER MARKERS
# =============================================================================

print("\nAdding cluster markers...")

colors = {'Q1': '#d73027', 'Q2': '#fc8d59', 'Q3': '#fee090', 'Q4': '#91cf60', 'Q5': '#1a9850'}

cluster_group = folium.FeatureGroup(name='DHS Clusters')

for _, row in df.iterrows():
    quintile = row['wealth_quintile']
    folium.CircleMarker(
        location=[row['LATNUM'], row['LONGNUM']],
        radius=3,
        color=colors[quintile],
        fill=True,
        fill_color=colors[quintile],
        fill_opacity=0.7,
        popup=f"Wealth: {row['wealth_score']:.3f}<br>Quintile: {quintile}",
    ).add_to(cluster_group)

cluster_group.add_to(m)
print(f"  Added {len(df):,} markers")

# =============================================================================
# ADD LEGEND
# =============================================================================

legend_html = """
<div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
     background: white; padding: 10px; border: 2px solid gray; border-radius: 5px;">
    <b>Wealth Quintiles</b><br>
    <i style="background:#d73027;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Q1 (Poorest)<br>
    <i style="background:#fc8d59;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Q2<br>
    <i style="background:#fee090;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Q3<br>
    <i style="background:#91cf60;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Q4<br>
    <i style="background:#1a9850;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Q5 (Richest)
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

folium.LayerControl(collapsed=False).add_to(m)

# =============================================================================
# SAVE
# =============================================================================

print(f"\nSaving map to: {OUTPUT_HTML}")
m.save(str(OUTPUT_HTML))

print("\n" + "="*60)
print("MAP GENERATED!")
print("="*60)
print(f"\nOutput: {OUTPUT_HTML}")
print("\nLayers:")
print("  - RGB Composites (All 9 Tiles)")
print("  - Nightlights (All 9 Tiles)")
print("  - DHS Clusters")
print("  - Base maps")
