import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from rasterstats import zonal_stats
import os
#from rasterio.warp import reproject, Resampling # Import necessary functions

# # --- Paths ---
# gpkg_path = "canopy_openness_result.gpkg"     # Input point file
# # dtm_path = "/content/drive/MyDrive/UROP/UROP RERTA Palapa June2019 DTM.tif"          # DTM raster
# # dem_path = "/content/drive/MyDrive/UROP/UROP RERTA Palapa June2019 DEM.tif"          # DEM raster
# chm_path = "/content/drive/MyDrive/UROP/UROP Rerta  Palapa June2019 CHM.tif"          # CHM raster
# output_buffer_gpkg = "buffered_points.gpkg"
# output_zonal_gpkg = "zonal_stats_result.gpkg"

def zonal_statistics(gpkg_path, raster_path, output_buffer_path, output_zonal_gpkg, buffer_geom_path = None):
    # --- Load and reproject point data ---
    points = gpd.read_file(gpkg_path)
    if points.crs.is_geographic:
        points = points.to_crs(points.estimate_utm_crs())

    points = gpd.read_file(gpkg_path)
    print("Vector CRS:", points.crs)

    # --- Create buffer around each point ---
    buffered = create_buffer(points, output_buffer_path, buffer_geom=gpd.read_file(buffer_geom_path) if buffer_geom_path else None)

    # --- Zonal Statistics on CHM ---
    stats = zonal_stats(
        vectors=buffered,
        raster= raster_path,
        stats=["mean", "min", "max", "sum","std","median","majority","minority","unique","range", "count"],
        geojson_out=True,
        nodata= None
    )

    # --- Convert stats back into GeoDataFrame ---
    zonal_gdf = gpd.GeoDataFrame.from_features(stats)
    zonal_gdf.set_crs(buffered.crs, inplace=True)

    # --- Save result ---
    zonal_gdf.to_file(output_zonal_gpkg, driver="GPKG")
    print(f"Saved zonal statistics to: {output_zonal_gpkg}")


def create_buffer(gpkg_vector, output_buffer_gpkg, buffer_geom = None, buffer_distance=12.5):
    """
    Creates a buffer around each point in the GeoPackage file, or uses pre-defined buffer geometries if provided with outer convex hulls.

    Args:
        gpkg_vector (gpd.GeoDataFrame): GeoDataFrame containing the input points.
        buffer_distance (float): Distance to buffer around each point in meters.
        output_buffer_gpkg (str): Path to save the buffered geometries.
        buffer_points (gpd.GeoDataFrame): Optional GeoDataFrame containing pre-defined buffer geometries.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the buffered geometries.
        
    """
    points = gpkg_vector
    if points.crs.is_geographic:
        points = points.to_crs(points.estimate_utm_crs())


    if buffer_geom is not None:
        gpkg_vector = gpkg_vector.drop(columns='geometry')
        buffer_geom = buffer_geom.merge(gpkg_vector, left_on='name', right_on='point.label', how='inner')
        # --- Use pre-defined buffer geometries ---
        print(buffer_geom.head())
        buffer_df = []
        
        for name, group in buffer_geom.groupby("name"):
            convex_hull = group.geometry.unary_union.convex_hull
            results_row = group.iloc[0].copy()  # Copy the geometry to avoid SettingWithCopyWarning
            results_row['geometry'] = convex_hull
            buffer_df.append(results_row)

        # Create GeoDataFrame with both name and geometry
        buffered = gpd.GeoDataFrame(buffer_df, crs=buffer_geom.crs)
        print(buffered.head())

    else:
        # --- Create buffer around each point ---
        buffered = points.copy()
        buffered['geometry'] = buffered.geometry.buffer(12.5)

    # Saving buffer
    buffered.to_file(output_buffer_gpkg, driver="GPKG")
    return buffered