import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from rasterstats import zonal_stats
import os


def zonal_statistics(gpkg_path, raster_path, output_buffer_path, output_zonal_gpkg, buffer_geom_path = None): 
    # --- Load and reproject point data ---
    points = gpd.read_file(gpkg_path)
    if points.crs.is_geographic:
        points = points.to_crs(points.estimate_utm_crs())

    print("Vector CRS:", points.crs)

    # --- Create buffer around each point ---
    buffered = create_buffer(points, output_buffer_path, buffer_geom=gpd.read_file(buffer_geom_path) if buffer_geom_path else None)

    # --- Zonal Statistics on CHM ---
    # stats = zonal_stats(
    #     nodata=0,
    #     vectors=buffered,
    #     raster= raster_path,
    #     stats=['mean', 'min', 'max', 'std', 'median', 'range', 'count'],
    #     geojson_out=True,
    # )

    # # --- Convert stats back into GeoDataFrame ---
    # zonal_gdf = gpd.GeoDataFrame.from_features(stats)
    # zonal_gdf.set_crs(buffered.crs, inplace=True)


    # --- Zonal Statistics on CHM with clipping ---
    results = []
        
    with rasterio.open(raster_path) as src:
        print(f"Raster CRS: {src.crs}")
        
        for idx, row in buffered.iterrows():
            try:
                # Clip raster to just an individual buffer
                masked_data, masked_transform = rasterio.mask.mask(
                    src, [row.geometry], crop=True, nodata=src.nodata
                )
                
                #print(masked_data)
                # Flatten the array and remove nodata
                valid_data = masked_data[masked_data != src.nodata] if src.nodata is not None else masked_data.flatten()
                
                # Apply your custom clipping (remove values < 0)
                clipped_data = valid_data[valid_data >= 0]
                print(clipped_data)

                negative_count = (clipped_data < 0).sum()
                print(f"Filtering check: {negative_count} buffers with negative mins (should be 0)")
                if negative_count == 0:
                    print("✅ Filtering working correctly!")
                else:
                    print("❌ Filtering failed!")
                
                if len(clipped_data) > 0:
                    stats = {
                        'mean': float(np.mean(clipped_data)),
                        'min': float(np.min(clipped_data)),
                        'max': float(np.max(clipped_data)),
                        'std': float(np.std(clipped_data)),
                        'median': float(np.median(clipped_data)),
                        'range': float(np.max(clipped_data) - np.min(clipped_data)),
                        'count': len(clipped_data)
                    }
                else:
                    # No valid data in this buffer
                    stats = {
                        'mean': np.nan, 'min': np.nan, 'max': np.nan,
                        'std': np.nan, 'median': np.nan, 'range': np.nan, 'count': 0
                    }
                
                # Combine with original row data
                result_row = row.to_dict()
                result_row.update(stats)
                results.append(result_row)
                
                print(f"Processed buffer {int(idx/16)+1}/{len(buffered)}")
                
            except Exception as e:
                print(f"Error processing buffer {idx}: {e}")
                continue
            
    zonal_gdf = gpd.GeoDataFrame(results, crs=buffered.crs)

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