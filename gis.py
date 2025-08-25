import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from rasterstats import zonal_stats
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd
from skimage.feature import graycomatrix, graycoprops


def clip_below_zero(data):
    """
    Clips the input data array to remove negative values.
    """
    clipped_data = data[data >= 0]
    negative_count = (clipped_data < 0).sum()
    print(f"Filtering check: {negative_count} buffers with negative mins (should be 0)")
    if negative_count == 0:
        print(" Filtering working correctly!")
    else:
        print(" Filtering failed!")
    return data[data >= 0]

def clip_above_num(data, upper_bound=20):
    """
    Clips the input data array to remove values above a specified upper bound.
    """
    clipped_data = data[data <= upper_bound]
    above_count = (clipped_data > upper_bound).sum()
    # print(f"Filtering check: {above_count} buffers with values above {upper_bound} (should be 0)")
    # if above_count == 0:
    #     print(" Filtering working correctly!")
    # else:
    #     print(" Filtering failed!")
    return data[data <= upper_bound]

def remove_outliers(data, thresh=3):
    """
    Clips the input data array to remove negative values.
    """
    
    # Remove outliers using 2*IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - thresh * IQR
    upper_bound = Q3 + thresh * IQR
    print(f"Filtering check: IQR lower bound = {lower_bound}, upper bound = {upper_bound}")
    print(" Filtering working correctly!")
    return data[(data >= lower_bound) & (data <= upper_bound)]

def combine_filters(filters):
    """
    Combines multiple filtering functions into a single function.
    """
    def combined_filter(data):
        for f in filters:
            data = f(data)
        return data
    return combined_filter

def create_buffer(points, output_buffer_gpkg, buffer_geom = None, buffer_distance=12.5):
    """
    Creates a buffer around each point in the GeoPackage file, or uses pre-defined buffer geometries if provided with outer convex hulls.

    Args:
        gpkg_vector (gpd.GeoDataFrame): GeoDataFrame containing the input points.
        output_buffer_gpkg (str): Path to save the buffered geometries.
        buffer_geom (gpd.GeoDataFrame): Optional GeoDataFrame containing pre-defined buffer geometries.
        buffer_distance (float): Distance to buffer around each point in meters.
        
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the buffered geometries.
        
    """
    if points.crs.is_geographic:
        points = points.to_crs(points.estimate_utm_crs())


    if buffer_geom is not None:
        points = points.drop(columns='geometry')
        buffer_geom = buffer_geom.merge(points, left_on='name', right_on='point.label', how='inner') #Merges columns in vector geopackage if contains data (e.g. Canopy openness)
        buffer_geom.drop(columns='name', inplace=True)
        # --- Use pre-defined buffer geometries ---
        print("Applying convex hull to buffer geometries...")

        buffer_df = []
        
        for name, group in buffer_geom.groupby("point.label"):
            convex_hull = group.geometry.unary_union.convex_hull
            results_row = group.iloc[0].copy()  # Copy the geometry to avoid SettingWithCopyWarning
            results_row['geometry'] = convex_hull
            buffer_df.append(results_row)

        # Create GeoDataFrame with both name and geometry
        buffered = gpd.GeoDataFrame(buffer_df, crs=buffer_geom.crs)

    else:
        # --- Create buffer around each point ---
        print(f"Creating buffer around each point with distance {buffer_distance}...")
        buffered = points.copy()
        buffered['geometry'] = buffered.geometry.buffer(12.5)
    
    #buffered = buffered[buffered['name'].str.contains("A|B|C|D", case=True, na=False)]
    # Saving buffer
    buffered.to_file(output_buffer_gpkg, driver="GPKG")
    print(f"Buffer geometries created and saved to {output_buffer_gpkg}")

    return buffered

def canopy_openness_proxy(data, thresh=30):
    data = np.asarray(data, dtype=float).flatten()
    canopy_openness = float(len(data[data < thresh]) / len(data) * 100)
    print(len(data[data < thresh]), "canopy openness pixels out of", len(data), "total pixels")
    if canopy_openness <= 100 :
        print(" Filtering working correctly!")
    else:
        print(" Canopy openness filtering failed!")
    return {
        'canopy_openness': canopy_openness # Percentage of positive values
    }

def GLCM(data, levels = 16):
    if data.dtype != np.uint16:
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        if vmin == vmax:
            return None  # constant image
        bins = np.linspace(vmin, vmax + 1e-12, levels + 1)
        q = np.clip(np.digitize(data, bins) - 1, 0, levels - 1)
        dtype = np.uint8 if levels <= 256 else np.uint16
        data = q.astype(dtype, copy=False)
    print("Using GLCM for texture analysis")
    glcm = graycomatrix(data, distances=[1], angles=[0], levels=levels, symmetric=True, normed=True)
    return {
        'contrast': graycoprops(glcm, 'contrast')[0, 0],
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0, 0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0, 0],
        'energy': graycoprops(glcm, 'energy')[0, 0],
        'correlation': graycoprops(glcm, 'correlation')[0, 0],
        'ASM': graycoprops(glcm, 'ASM')[0, 0]
    }

def zonal_statistics(gpkg_path, raster_path, output_buffer_path, output_zonal_gpkg, filtering_logic = None, proxies = None, buffer_geom_path = None, show_plots=False, value='y', save_plots=False): 
    '''
    
    Performs zonal statistics on a raster file using buffered geometries from a GeoPackage.
    There are three ways of creating the buffer geometries: 
    1. Create a buffer around each point in the GeoPackage.
    2. Use a pre-defined points outlining the buffers from another GeoPackage (Convex Hulls).
    3. Import a GeoPackage with pre-defined buffer geometries.

    Args:
        gpkg_path (str): Path to the GeoPackage containing point data.
        raster_path (str): Path to the raster file for which statistics are calculated.
        output_buffer_path (str): Path to save the buffered geometries.
        output_zonal_gpkg (str): Path to save the zonal statistics results.
        filtering_logic (function) : A function that takes a DataFrame and returns a filtered DataFrame.
        proxies (func): A function which returns a dictionary of proxy functions to apply to the data, e.g. gis.canopy_openness_proxy
        buffer_geom_path (str, optional): Path to a GeoPackage containing pre-defined buffer geometries.
        value (str) : Name of the variable being plotted.
        create_plots (bool): Whether to create distribution plots by treatment region.
        save_plots (bool): Whether to save the distribution plots.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the zonal statistics results.

    '''
    #--- Load and reproject point data ---
    points = gpd.read_file(gpkg_path)
    if points.crs.is_geographic:
        points = points.to_crs(points.estimate_utm_crs())


    print(f"\n \n Calculating Zonal Statistics for {output_zonal_gpkg}")
    #--- Create buffer around each point ---
    try:
        if buffer_geom_path and (gpd.read_file(buffer_geom_path).geometry.type=="MultiPolygon").all():
            points = points.drop(columns='geometry')
            buffered = gpd.read_file(buffer_geom_path)
            buffered = buffered.merge(points, left_on='name', right_on='point.label', how='inner') #Merges columns in vector geopackage if contains data (e.g. Canopy openness)
            buffered.drop(columns='name', inplace=True)
        else:
            print("WARNING No buffer geometry entered, creating buffer around each point...")
            buffered = create_buffer(points, output_buffer_path, buffer_geom=gpd.read_file(buffer_geom_path) if buffer_geom_path else None)

    except Exception as e:
        print(f"Error creating buffer geometries: {e}")
        return None

    results = [] # Store results for each buffer, each element being a dictionary
    region_data = defaultdict(list) # Stores all pixels which belong to a region
    
    
    with rasterio.open(raster_path) as src:

        for idx, row in buffered.iterrows():
            try:
                region_name = row.get('treatment', f'Treatment {idx}')

                # Clip raster to just an individual buffer
                masked_data, masked_transform = rasterio.mask.mask(
                    src, [row.geometry], crop=True, nodata=src.nodata
                )

                # # Flatten the array and remove nodata, vectorising for better performance
                valid_data = masked_data[masked_data != src.nodata] if src.nodata is not None else masked_data.flatten()
                

                valid_data = np.asarray(valid_data, dtype=float)  # Ensure data is 1D
                finite_mask = np.isfinite(valid_data)
                valid_data = valid_data[finite_mask]
        

                # Apply your custom clipping (remove values < 0)
                clipped_data = filtering_logic(valid_data) if filtering_logic else valid_data


                if len(clipped_data) > 0:
                    region_data[region_name].extend(clipped_data)

                    stats = {
                        'mean': float(np.mean(clipped_data)),
                        #'min': float(np.min(clipped_data)),
                        #'max': float(np.max(clipped_data)),
                        #'std': float(np.std(clipped_data)),
                        #'median': float(np.median(clipped_data)),
                        'range': float(np.max(clipped_data) - np.min(clipped_data)),
                        #'count': len(clipped_data),
                        'cv': float(np.std(clipped_data) / np.mean(clipped_data)) if np.mean(clipped_data) != 0 else 0
                    }
                    if proxies:
                        try:
                            stats = stats | proxies(masked_data[0])
                        except Exception as e:
                            print(f"Error applying proxies: {e}")
                            try:
                                stats = stats | proxies(clipped_data)
                            except Exception as e:
                                print(f"Error applying proxies: {e}")
                else:
                    # No valid data in this buffer
                    stats = {
                        'mean': np.nan, 'min': np.nan, 'max': np.nan,
                        'std': np.nan, 'median': np.nan, 'range': np.nan, 'count': 0
                    }

                    print(f"No valid data in buffer {idx} {row.get('point.label')}, skipping...")

                # Combine with original row data
                result_row = row.to_dict()
                result_row.update(stats)
                results.append(result_row)
                
                print(f"Processed buffer {int(idx/16)+1}    {idx+1}/{len(buffered)}")
                
            except Exception as e:
                print(f"Error processing buffer {idx}, at {row.get('point.label')} : {e}")
                continue

    zonal_gdf = gpd.GeoDataFrame(results, crs=buffered.crs)
    region_data = {k: region_data[k] for k in sorted(region_data.keys())}

    # --- Save result ---
    zonal_gdf.to_file(output_zonal_gpkg, driver="GPKG", overwrite=True)
    print(f"Saved zonal statistics to: {output_zonal_gpkg}")

    # Create plots if requested
    figures = []
    if (show_plots or save_plots) and region_data:
        print("\nCreating distribution plots...")
        
        # Create plots using pre-processed data
        fig1 = create_distribution_plots_from_data(region_data, value, output_path=output_zonal_gpkg, save_plots=save_plots)
        if fig1:
            figures.append(fig1)

        fig2 = create_boxplot_from_data(region_data, value, output_path=output_zonal_gpkg, save_plots=save_plots)
        if fig2:
            figures.append(fig2)

        if show_plots:
            plt.show()

    return zonal_gdf

def get_region_data(gpkg_path, value_column):
    """
    Extract region data from the zonal GeoDataFrame.

    Args:
        zonal_gdf (GeoDataFrame): The zonal GeoDataFrame containing the results.
        value_column (str): The name of the column containing the values to extract.
    Returns:
        dict: A dictionary with region names as keys and their data as values.
    """
    gdf = gpd.read_file(gpkg_path)


    region_data = defaultdict(list)
    for index, row in gdf.iterrows():
        region_name = row.get('treatment', f'Treatment {index}')
        region_data[region_name].append(row[value_column])

    region_data = {k: region_data[k] for k in sorted(region_data.keys())}
    return region_data

def create_distribution_plots_from_data(region_data, value, figsize=(15, 10), output_path=None, save_plots=False):
    """
    Create distribution plots from pre-processed data.

    Args:
        region_data (dict): A dictionary containing region names as keys and their data as values.
        value (str): The name of the variable being plotted.
        figsize (tuple): The size of the figure to create.
        output_path (str): The path to save the figure.
        save_plots (bool): Whether to save the plots.

    Returns:
        matplotlib.figure.Figure: The created figure.

    """
    output_path = output_path.replace('Data', 'Results') if output_path else None
    n_regions = len(region_data)
    if n_regions == 0:
        return None
    
    cols = min(4, n_regions)
    rows = (n_regions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_regions == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, (region_name, data) in enumerate(region_data.items()):
        ax = axes[idx]
        
        # Ensure numeric and remove non-finite values (NaN, +inf, -inf)
        data_array = np.asarray(data, dtype=float)
        finite_mask = np.isfinite(data_array)
        data_to_plot = data_array[finite_mask]

        # If nothing left to plot, annotate and skip plotting to avoid histogram errors
        if data_to_plot.size == 0:
            mean_val = np.nan
            std_val = np.nan
            count_val = 0
            ax.text(0.02, 0.98, "No finite data", transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.set_title(f'{region_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel(value)
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            continue

        # Create histogram with only finite values
        try:
            ax.hist(data_to_plot, bins=50 if len(data_to_plot) >= 2500 else 10, alpha=0.7, color=plt.cm.Set3(idx),
                    edgecolor='black', linewidth=0.5)
        except Exception as e:
            print(f"Histogram error for region {region_name}: {e}")
            # Fallback: simple density plot
            sns.kdeplot(data_to_plot, ax=ax, fill=True, color=plt.cm.Set3(idx))

        # Statistics (on finite values)
        mean_val = float(np.mean(data_to_plot))
        std_val = float(np.std(data_to_plot))
        count_val = int(data_to_plot.size)

        stats_text = f"Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nCount: {count_val}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                label=f"Mean: {mean_val:.2f}")
        
        ax.set_title(f'{region_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel(value)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(n_regions, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    if output_path:
        plt.suptitle(f'{os.path.basename(output_path).replace(".gpkg", "")} by Treatment Region', fontsize=16, fontweight='bold', y=1.02)
        if save_plots:
            plt.savefig(output_path.replace('.gpkg', '_distributions.png'), dpi=300, bbox_inches='tight')
            print(f"Distribution plots saved to: {output_path.replace('.gpkg', '_distributions.png')}")

    return fig

def create_boxplot_from_data(region_data, value,figsize=(12, 8), output_path=None, save_plots=False):
    """
    Create boxplot from pre-processed data.

    Args:
        region_data (dict): A dictionary containing region names as keys and their data as values.
        value (str): The name of the variable being plotted.
        figsize (tuple): The size of the figure to create.
        output_path (str): The path to save the figure.
        save_plots (bool): Whether to save the plots.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """

    output_path = output_path.replace('Data', 'Results') if output_path else None
    if not region_data:
        return None
    
    regions = list(region_data.keys())
    data_lists = [region_data[region] for region in regions]
    
    fig, ax = plt.subplots(figsize=figsize)

    box_plot = ax.boxplot(data_lists, labels=[f"Treatment {region}" for region in regions], patch_artist=True)

    # Customize colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Treatment Region', fontsize=12)
    ax.set_ylabel(value, fontsize=12)
    ax.grid(True, alpha=0.3)
 
    ax.tick_params(axis='x', labelsize=13)
    
    # Add sample size annotations
    for i, (region, data) in enumerate(region_data.items()):
        ax.text(i+1, ax.get_ylim()[0]+0.5, f'number of pixels (n)={len(data)}', ha='center', va='top', fontsize=10)
    
    plt.tight_layout()
    if output_path:
        ax.set_title(f'{os.path.basename(output_path).replace(".gpkg", "")} Comparison Across Treatment Regions', 
                fontsize=14, fontweight='bold')
        if save_plots:
            plt.savefig(output_path.replace('.gpkg', '_boxplot.png'), dpi=300, bbox_inches='tight')
            print(f"Boxplot saved to: {output_path.replace('.gpkg', '_boxplot.png')}")

    return fig


# Example usage - Comment out when not testing
# Test the plotting functions
# print("Creating CHM distribution plots...")

# # Use optimized functions for better performance
# zonal_gdf, figures = zonal_statistics(
#     gpkg_path="Frogs_result.gpkg",
#     raster_path="D:/Jerry/Palapa July2025 Clre.tif", 
#     output_buffer_path=0, 
#     filtering_logic=remove_outliers,
#     output_zonal_gpkg="Data/Palapa June2025 Clre Statistics.gpkg", 
#     buffer_geom_path="Data/Palapa_transects_buffer.gpkg",
#     #proxies=canopy_openness_proxy,
#     show_plots=True
#     #save_plots=True
# )

# print("Plots created and saved successfully!")