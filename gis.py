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

def zonal_statistics(gpkg_path, raster_path, output_buffer_gpkg, output_zonal_gpkg, buffer_points = None):
    # --- Load and reproject point data ---
    points = gpd.read_file(gpkg_path)
    print(points.crs)
    if points.crs.is_geographic:
        points = points.to_crs(points.estimate_utm_crs())


    if buffer_points is not None:
        # --- Create buffer around each point ---
        # Group by 'zone_id' and make convex hulls
        buffer_points = gpd.read_file(buffer_points)
        polygons = buffer_points.groupby("name")["geometry"].apply(lambda x: x.unary_union.convex_hull)

        # Convert to GeoDataFrame
        buffered = gpd.GeoDataFrame(polygons, geometry=polygons)
        buffered = buffered.set_crs(buffer_points.crs)
        #multi = polygons_gdf.union_all() # This is a shapely MultiPolygon object
    else:
        # --- Create buffer around each point ---
        buffered = points.copy()
        buffered['geometry'] = buffered.geometry.buffer(12.5)

    # Saving buffer
    buffered.to_file(output_buffer_gpkg, driver="GPKG")

    points = gpd.read_file(gpkg_path)
    print("Vector CRS:", points.crs)

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