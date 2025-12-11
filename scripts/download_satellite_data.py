#!/usr/bin/env python3
"""
Download Satellite Data via Google Earth Engine

This script exports multi-spectral satellite composites for India via GEE.
Run in Google Colab with Earth Engine credentials.

Output: Full India composites saved to Google Drive, which can then be
cropped using src/crop_and_process.py
"""

import ee
import pandas as pd
import numpy as np

# Initialize Earth Engine
ee.Initialize()

print("=" * 60)
print("DOWNLOAD INDIA SATELLITE COMPOSITES")
print("=" * 60)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Update this path for your Google Drive
CSV_PATH = '/content/drive/MyDrive/Predicting Poverty AI4SD/cluster_wealth_gps.csv'
OUTPUT_FOLDER = 'DHS_India_Full_Composites'
YEARS = [2019, 2020, 2021]
RESOLUTION = 100  # meters


# =============================================================================
# LOAD AND VALIDATE GPS DATA
# =============================================================================

print("\nLoading cluster coordinates...")
df = pd.read_csv(CSV_PATH)

# Remove invalid coordinates
df_valid = df[(df['LATNUM'] > 0) & (df['LONGNUM'] > 0) &
              (df['LATNUM'].notna()) & (df['LONGNUM'].notna())].copy()

print(f"Total clusters: {len(df)}")
print(f"Valid GPS: {len(df_valid)}")

# Get bounding box
min_lat = df_valid['LATNUM'].min() - 0.5
max_lat = df_valid['LATNUM'].max() + 0.5
min_lon = df_valid['LONGNUM'].min() - 0.5
max_lon = df_valid['LONGNUM'].max() + 0.5

# Clip to reasonable India bounds
min_lat = max(min_lat, -1)
max_lat = min(max_lat, 40)
min_lon = max(min_lon, 60)
max_lon = min(max_lon, 100)

india_region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

print(f"\nExport region:")
print(f"  Latitude:  {min_lat:.2f} to {max_lat:.2f}")
print(f"  Longitude: {min_lon:.2f} to {max_lon:.2f}")

# Save valid clusters
valid_csv = CSV_PATH.replace('.csv', '_valid.csv')
df_valid.to_csv(valid_csv, index=False)
print(f"\nSaved valid clusters to: {valid_csv}")


# =============================================================================
# CREATE COMPOSITE FUNCTION
# =============================================================================

def create_stack(year):
    """
    Create a multi-band composite for a given year.
    
    Bands:
        0-2: Sentinel-2 RGB (Red, Green, Blue)
        3:   Dynamic World Built Area Probability
        4:   VIIRS Nighttime Lights
        5:   SRTM Elevation
        6:   SRTM Slope
        7:   ERA5 Temperature (optional, can be excluded)
    """
    start = f'{year}-01-01'
    end = f'{year}-12-31'
    
    print(f"  Building {year} composite...")
    
    # Sentinel-2 Optical
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start, end).filterBounds(india_region) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .select(['B4', 'B3', 'B2'])
    
    # Dynamic World Land Use
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        .filterDate(start, end).filterBounds(india_region) \
        .select(['built'])
    
    # VIIRS Nighttime Lights
    viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') \
        .filterDate(start, end).select(['avg_rad'])
    
    # DEM (static)
    dem = ee.Image('USGS/SRTMGL1_003')
    elevation = dem.select('elevation')
    slope = ee.Terrain.slope(dem)
    
    # ERA5 Temperature
    era5 = ee.ImageCollection('ECMWF/ERA5/MONTHLY') \
        .filterDate(start, end).select(['mean_2m_air_temperature'])
    
    # Create composites with explicit band names
    s2_composite = s2.median().rename(['S2_Red', 'S2_Green', 'S2_Blue']).toFloat()
    dw_composite = dw.mean().rename(['DW_Built']).toFloat()
    viirs_composite = viirs.median().rename(['VIIRS_NL']).toFloat()
    elevation_band = elevation.rename(['DEM_Elevation']).toFloat()
    slope_band = slope.rename(['DEM_Slope']).toFloat()
    temp_composite = era5.mean().rename(['ERA5_Temp']).toFloat()
    
    # Stack all bands
    stack = s2_composite \
        .addBands(dw_composite) \
        .addBands(viirs_composite) \
        .addBands(elevation_band) \
        .addBands(slope_band) \
        .addBands(temp_composite)
    
    return stack


# =============================================================================
# EXPORT COMPOSITES
# =============================================================================

print("\nStarting exports...")

for year in YEARS:
    stack = create_stack(year)
    
    task = ee.batch.Export.image.toDrive(
        image=stack,
        description=f'india_full_{year}',
        fileNamePrefix=f'india_full_{year}',
        folder=OUTPUT_FOLDER,
        region=india_region,
        scale=RESOLUTION,
        crs='EPSG:4326',
        maxPixels=1e13,
        fileFormat='GeoTIFF'
    )
    task.start()
    print(f"  Queued: india_full_{year}.tif")

print("\n" + "=" * 60)
print("EXPORTS QUEUED!")
print("=" * 60)

print(f"""
Summary:
  Valid clusters: {len(df_valid)}
  Files to export: {len(YEARS)} (one per year)
  Resolution: {RESOLUTION}m
  Bands: 8
  Folder: {OUTPUT_FOLDER}/

Bands:
  0-2: S2_Red, S2_Green, S2_Blue
  3:   DW_Built
  4:   VIIRS_NL
  5:   DEM_Elevation
  6:   DEM_Slope
  7:   ERA5_Temp

Estimated time: 2-4 hours
Monitor: https://code.earthengine.google.com/tasks

Next: Use src/crop_and_process.py to extract 256x256 patches
""")
