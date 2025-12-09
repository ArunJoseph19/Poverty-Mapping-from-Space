# Data Directory

This directory should contain the following data:

## Required Files

### 1. Cluster Data
`cluster_wealth_gps_valid.csv` - CSV file with DHS cluster information

Required columns:
- `LATNUM`: Latitude of cluster centroid
- `LONGNUM`: Longitude of cluster centroid
- `wealth_score`: DHS wealth index value

### 2. Satellite Composites
`DHS_India_Full_Composites/` - Directory containing GeoTIFF files

Files should be named: `india_full_YYYY_*.tif` where YYYY is the year.

Each GeoTIFF should have 8 bands (we use 7, excluding ERA5 temperature):
1. Sentinel-2 Red
2. Sentinel-2 Green
3. Sentinel-2 Blue
4. Dynamic World Built Area
5. VIIRS Nighttime Lights
6. SRTM Elevation
7. SRTM Slope
8. ERA5 Temperature (excluded)

## Data Sources

### DHS Data
- Website: https://dhsprogram.com/
- Registration required
- Request access to India GPS dataset

### Satellite Imagery (via Google Earth Engine)
- Sentinel-2: Surface reflectance
- VIIRS: Monthly nighttime lights
- Dynamic World: Land cover classification
- SRTM: Digital elevation model

## Output Files

After running `crop_and_process.py`, the following files will be created in `processed/`:

- `satellite_images.npy` - Memory-mapped array of images
- `wealth_labels.npy` - Wealth scores for each image
- `nightlight_labels.npy` - Nightlight class labels (0, 1, 2)
- `train_idx.npy`, `val_idx.npy`, `test_idx.npy` - Split indices
- `dataset_info.json` - Dataset metadata
- `nightlight_thresholds.json` - Class thresholds
- `metadata.csv` - Per-image metadata

## Notes

- GPS coordinates in DHS data have random displacement for privacy
- Urban clusters: 0-2km displacement
- Rural clusters: 0-5km displacement (with 1% displaced up to 10km)
