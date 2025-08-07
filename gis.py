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
                        'count': len(clipped_data),
                        'canopy_coverage': float(len(clipped_data[clipped_data > 0.5])) / len(clipped_data) * 100  # Percentage of positive values
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
    
    #buffered = buffered[buffered['name'].str.contains("A|B|C|D", case=True, na=False)]
    # Saving buffer
    buffered.to_file(output_buffer_gpkg, driver="GPKG")
    return buffered

### BE CAUTIOUS FUNCTIONS BELOW STILL BEING TESTED AND NOT USED IN MAIN WORKFLOW YET ###

# def raster_calculation_zonal(gpkg_path, raster_path, output_buffer_path, output_zonal_gpkg, 
#                            calculation_func, calculation_name, buffer_geom_path=None):
#     """
#     Performs custom raster calculations on each buffer zone instead of basic statistics.
    
#     Args:
#         gpkg_path (str): Path to the point data GeoPackage
#         raster_path (str): Path to the multi-band raster file
#         output_buffer_path (str): Path to save buffer geometries
#         output_zonal_gpkg (str): Path to save results
#         calculation_func (function): Function that takes raster bands and returns calculated values
#         calculation_name (str): Name for the calculation (e.g., 'NDVI', 'Red_minus_Green')
#         buffer_geom_path (str, optional): Path to predefined buffer geometries
    
#     Returns:
#         gpd.GeoDataFrame: Results with calculation statistics per buffer
    
#     Example calculation functions:
#         # NDVI calculation
#         def ndvi_calc(bands):
#             red = bands[2]    # Band 3 (0-indexed)
#             nir = bands[3]    # Band 4 (0-indexed)
#             return (nir - red) / (nir + red + 1e-8)  # Add small value to avoid division by zero
        
#         # Simple band difference
#         def red_minus_green(bands):
#             red = bands[2]    # Band 3
#             green = bands[1]  # Band 2
#             return red - green
        
#         # Custom vegetation index
#         def custom_vi(bands):
#             blue = bands[0]
#             green = bands[1]
#             red = bands[2]
#             return (green - blue) / (red + green + blue + 1e-8)
#     """
    
#     # --- Load and reproject point data ---
#     points = gpd.read_file(gpkg_path)
#     if points.crs.is_geographic:
#         points = points.to_crs(points.estimate_utm_crs())

#     print("Vector CRS:", points.crs)

#     # --- Create buffer around each point ---
#     buffered = create_buffer(points, output_buffer_path, 
#                            buffer_geom=gpd.read_file(buffer_geom_path) if buffer_geom_path else None)

#     # --- Perform raster calculations on each buffer ---
#     results = []
        
#     with rasterio.open(raster_path) as src:
#         print(f"Raster CRS: {src.crs}")
#         print(f"Raster bands: {src.count}")
#         print(f"Raster shape: {src.width} x {src.height}")
        
#         for idx, row in buffered.iterrows():
#             try:
#                 # Clip raster to individual buffer - get all bands
#                 masked_data, masked_transform = rasterio.mask.mask(
#                     src, [row.geometry], crop=True, nodata=src.nodata, all_touched=True
#                 )
                
#                 print(f"Processing buffer {idx + 1}/{len(buffered)}")
#                 print(f"Masked data shape: {masked_data.shape}")  # Should be (bands, height, width)
                
#                 # Remove nodata values from each band
#                 valid_mask = masked_data[0] != src.nodata if src.nodata is not None else np.ones_like(masked_data[0], dtype=bool)
                
#                 # Apply nodata mask to all bands
#                 for band_idx in range(masked_data.shape[0]):
#                     if src.nodata is not None:
#                         band_valid_mask = masked_data[band_idx] != src.nodata
#                         valid_mask = valid_mask & band_valid_mask
                
#                 # Extract valid pixels for all bands
#                 valid_bands = []
#                 for band_idx in range(masked_data.shape[0]):
#                     valid_band_data = masked_data[band_idx][valid_mask]
#                     valid_bands.append(valid_band_data)
                
#                 if len(valid_bands[0]) > 0:
#                     # Perform the custom calculation
#                     calculated_values = calculation_func(valid_bands)
                    
#                     # Remove invalid results (NaN, inf)
#                     calculated_values = calculated_values[np.isfinite(calculated_values)]
                    
#                     if len(calculated_values) > 0:
#                         # Calculate statistics on the result
#                         stats = {
#                             f'{calculation_name}_mean': float(np.mean(calculated_values)),
#                             f'{calculation_name}_min': float(np.min(calculated_values)),
#                             f'{calculation_name}_max': float(np.max(calculated_values)),
#                             f'{calculation_name}_std': float(np.std(calculated_values)),
#                             f'{calculation_name}_median': float(np.median(calculated_values)),
#                             f'{calculation_name}_range': float(np.max(calculated_values) - np.min(calculated_values)),
#                             f'{calculation_name}_count': len(calculated_values),
#                             f'{calculation_name}_percentile_25': float(np.percentile(calculated_values, 25)),
#                             f'{calculation_name}_percentile_75': float(np.percentile(calculated_values, 75))
#                         }
                        
#                         print(f"  {calculation_name} range: {stats[f'{calculation_name}_min']:.3f} to {stats[f'{calculation_name}_max']:.3f}")
#                         print(f"  Valid pixels: {stats[f'{calculation_name}_count']}")
                        
#                     else:
#                         print(f"  No valid calculated values for buffer {idx}")
#                         stats = {f'{calculation_name}_{stat}': np.nan 
#                                 for stat in ['mean', 'min', 'max', 'std', 'median', 'range', 'percentile_25', 'percentile_75']}
#                         stats[f'{calculation_name}_count'] = 0
#                 else:
#                     print(f"  No valid pixels in buffer {idx}")
#                     stats = {f'{calculation_name}_{stat}': np.nan 
#                             for stat in ['mean', 'min', 'max', 'std', 'median', 'range', 'percentile_25', 'percentile_75']}
#                     stats[f'{calculation_name}_count'] = 0
                
#                 # Combine with original row data
#                 result_row = row.to_dict()
#                 result_row.update(stats)
#                 results.append(result_row)
                
#             except Exception as e:
#                 print(f"Error processing buffer {idx}: {e}")
#                 # Add row with NaN values for failed processing
#                 result_row = row.to_dict()
#                 error_stats = {f'{calculation_name}_{stat}': np.nan 
#                               for stat in ['mean', 'min', 'max', 'std', 'median', 'range', 'percentile_25', 'percentile_75']}
#                 error_stats[f'{calculation_name}_count'] = 0
#                 result_row.update(error_stats)
#                 results.append(result_row)
#                 continue
            
#     zonal_gdf = gpd.GeoDataFrame(results, crs=buffered.crs)

#     # --- Save result ---
#     zonal_gdf.to_file(output_zonal_gpkg, driver="GPKG")
#     print(f"Saved raster calculation results to: {output_zonal_gpkg}")
    
#     return zonal_gdf


# def create_calculation_functions():
#     """
#     Predefined calculation functions for common raster operations
    
#     Returns:
#         dict: Dictionary of calculation functions
#     """
    
#     def ndvi_calculation(bands):
#         """NDVI = (NIR - Red) / (NIR + Red)"""
#         if len(bands) < 4:
#             raise ValueError("NDVI requires at least 4 bands (assuming Red=band3, NIR=band4)")
#         red = bands[2].astype(float)   # Band 3 (0-indexed)
#         nir = bands[3].astype(float)   # Band 4 (0-indexed)
#         denominator = nir + red
#         # Avoid division by zero
#         ndvi = np.where(denominator != 0, (nir - red) / denominator, 0)
#         return ndvi
    
#     def red_minus_green(bands):
#         """Simple band difference: Red - Green"""
#         if len(bands) < 3:
#             raise ValueError("Red minus Green requires at least 3 bands")
#         red = bands[2].astype(float)    # Band 3
#         green = bands[1].astype(float)  # Band 2
#         return red - green
    
#     def green_leaf_index(bands):
#         """Green/Red ratio"""
#         if len(bands) < 3:
#             raise ValueError("Green/Red ratio requires at least 3 bands")
#         red = bands[2].astype(float)
#         green = bands[1].astype(float)
#         return np.where(red != 0, green / red, 0)
    
#     def enhanced_vegetation_index(bands):
#         """EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)"""
#         if len(bands) < 4:
#             raise ValueError("EVI requires at least 4 bands")
#         blue = bands[0].astype(float)
#         red = bands[2].astype(float)
#         nir = bands[3].astype(float)
        
#         denominator = nir + 6*red - 7.5*blue + 1
#         evi = np.where(denominator != 0, 2.5 * (nir - red) / denominator, 0)
#         return evi
    
#     def brightness_index(bands):
#         """Simple brightness: mean of all bands"""
#         all_bands = np.stack(bands, axis=0)
#         return np.mean(all_bands, axis=0)
    
#     def band_ratio(bands, numerator_idx=3, denominator_idx=2):
#         """Generic band ratio (default NIR/Red)"""
#         if len(bands) <= max(numerator_idx, denominator_idx):
#             raise ValueError(f"Not enough bands for ratio {numerator_idx}/{denominator_idx}")
        
#         numerator = bands[numerator_idx].astype(float)
#         denominator = bands[denominator_idx].astype(float)
#         return np.where(denominator != 0, numerator / denominator, 0)
    
#     return {
#         'NDVI': ndvi_calculation,
#         'Red_minus_Green': red_minus_green,
#         'Green_Leaf_Index': green_leaf_index,
#         'EVI': enhanced_vegetation_index,
#         'Brightness': brightness_index,
#         'NIR_Red_Ratio': lambda bands: band_ratio(bands, 3, 2),
#         'Blue_Green_Ratio': lambda bands: band_ratio(bands, 0, 1)
#     }


# # Example usage function
# def example_usage():
#     """
#     Example of how to use the raster calculation function
#     """
    
#     # Get predefined calculation functions
#     calc_functions = create_calculation_functions()
    
#     # Example 1: Calculate NDVI
#     raster_calculation_zonal(
#         gpkg_path="points.gpkg",
#         raster_path="multispectral_image.tif",
#         output_buffer_path="buffers_ndvi.gpkg",
#         output_zonal_gpkg="ndvi_results.gpkg",
#         calculation_func=calc_functions['NDVI'],
#         calculation_name='NDVI'
#     )
    
#     # Example 2: Custom calculation function
#     def custom_vegetation_index(bands):
#         """Custom calculation: (Green + NIR) / (Red + Blue)"""
#         if len(bands) < 4:
#             raise ValueError("Custom VI requires 4 bands")
#         blue = bands[0].astype(float)
#         green = bands[1].astype(float)
#         red = bands[2].astype(float)
#         nir = bands[3].astype(float)
        
#         numerator = green + nir
#         denominator = red + blue
#         return np.where(denominator != 0, numerator / denominator, 0)
    
#     raster_calculation_zonal(
#         gpkg_path="points.gpkg",
#         raster_path="multispectral_image.tif",
#         output_buffer_path="buffers_custom.gpkg",
#         output_zonal_gpkg="custom_vi_results.gpkg",
#         calculation_func=custom_vegetation_index,
#         calculation_name='Custom_VI'
#     )