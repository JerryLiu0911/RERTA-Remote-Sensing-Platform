import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from rasterstats import zonal_stats
import os
from rasterio.warp import reproject, Resampling # Import necessary functions

# --- Paths ---
gpkg_path = " Rerta_koordinate_corrected_result_data.gpkg"     # Input point file
# dtm_path = "/content/drive/MyDrive/UROP/UROP RERTA Palapa June2019 DTM.tif"          # DTM raster
# dem_path = "/content/drive/MyDrive/UROP/UROP RERTA Palapa June2019 DEM.tif"          # DEM raster
chm_path = "/content/drive/MyDrive/UROP/UROP Rerta Palapa June2019 CHM.tif"          # CHM raster
output_buffer_gpkg = "buffered_points.gpkg"
output_zonal_gpkg = "zonal_stats_result.gpkg"


# --- Load and reproject point data ---
points = gpd.read_file(gpkg_path)
print(points.crs)
if points.crs.is_geographic:
    points = points.to_crs(points.estimate_utm_crs())

# --- Create buffer around each point ---
buffered = points.copy()
buffered['geometry'] = buffered.geometry.buffer(5)

# Optional: Save buffered layer
buffered.to_file(output_buffer_gpkg, driver="GPKG")

points = gpd.read_file(gpkg_path)
print("Vector CRS:", points.crs)

# --- Zonal Statistics on CHM ---
stats = zonal_stats(
    vectors=buffered,
    raster=chm_path,
    stats=['min', 'max', 'mean', 'median', 'std'],
    geojson_out=True,
    nodata=-9999
)

# --- Convert stats back into GeoDataFrame ---
zonal_gdf = gpd.GeoDataFrame.from_features(stats)
zonal_gdf.set_crs(buffered.crs, inplace=True)

# --- Save result ---
zonal_gdf.to_file(output_zonal_gpkg, driver="GPKG")
print(f"Saved zonal statistics to: {output_zonal_gpkg}")