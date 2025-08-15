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


def clip_below_zero(data):
    """
    Clips the input data array to remove negative values.
    """
    clipped_data = data[data >= 0]
    negative_count = (clipped_data < 0).sum()
    print(f"Filtering check: {negative_count} buffers with negative mins (should be 0)")
    if negative_count == 0:
        print("✅ Filtering working correctly!")
    else:
        print("❌ Filtering failed!")
    return data[data >= 0]

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

def zonal_statistics(gpkg_path, raster_path, output_buffer_path, output_zonal_gpkg, filtering_logic = None, buffer_geom_path = None, show_plots=False, save_plots=False): 
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
        buffer_geom_path (str, optional): Path to a GeoPackage containing pre-defined buffer geometries.
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
        buffered = gpd.read_file(buffer_geom_path) if (gpd.read_file(buffer_geom_path).geometry.type=="MultiPolygon").all() else create_buffer(points, output_buffer_path, buffer_geom=gpd.read_file(buffer_geom_path) if buffer_geom_path else None)

    except Exception as e:
        print(f"Error creating buffer geometries: {e}")
        return None

    results = [] # Store results for each buffer, each element being a dictionary
    region_data = defaultdict(list) # Stores all pixels which belong to a region
    
    
    with rasterio.open(raster_path) as src:
        print(f"Point CRS: f{points.crs}")
        print(f"Raster CRS: {src.crs}")
        if points.crs != src.crs:
            print("⚠️ Warning: CRS of points and raster do not match!")
        else:
            print("✅ CRS of points and raster match.")

        for idx, row in buffered.iterrows():
            try:
                region_name = row.get('treatment', f'Treatment {idx}')

                # Clip raster to just an individual buffer
                masked_data, masked_transform = rasterio.mask.mask(
                    src, [row.geometry], crop=True, nodata=src.nodata
                )
                

                # Flatten the array and remove nodata, vectorising for better performance
                valid_data = masked_data[masked_data != src.nodata] if src.nodata is not None else masked_data.flatten()
                
                # Apply your custom clipping (remove values < 0)
                clipped_data = filtering_logic(valid_data) if filtering_logic else valid_data

                if len(clipped_data) > 0:
                    region_data[region_name].extend(clipped_data)

                    stats = {
                        'mean': float(np.mean(clipped_data)),
                        'min': float(np.min(clipped_data)),
                        'max': float(np.max(clipped_data)),
                        'std': float(np.std(clipped_data)),
                        'median': float(np.median(clipped_data)),
                        'range': float(np.max(clipped_data) - np.min(clipped_data)),
                        'count': len(clipped_data),
                        'canopy_openness': float(len(clipped_data[clipped_data < 0.8])) / len(clipped_data) * 100  # Percentage of positive values
                    }

                else:
                    # No valid data in this buffer
                    stats = {
                        'mean': np.nan, 'min': np.nan, 'max': np.nan,
                        'std': np.nan, 'median': np.nan, 'range': np.nan, 'count': 0
                    }

                    print(f"No valid data in buffer {idx}, skipping...")
                
                # Combine with original row data
                result_row = row.to_dict()
                result_row.update(stats)
                results.append(result_row)
                
                print(f"Processed buffer {int(idx/16)+1}    {idx+1}/{len(buffered)}")
                
            except Exception as e:
                print(f"Error processing buffer {idx}: {e}")
                continue
            
    zonal_gdf = gpd.GeoDataFrame(results, crs=buffered.crs)

    # --- Save result ---
    zonal_gdf.to_file(output_zonal_gpkg, driver="GPKG")
    print(f"Saved zonal statistics to: {output_zonal_gpkg}")

    # Create plots if requested
    figures = []
    if (show_plots or save_plots) and region_data:
        print("\nCreating distribution plots...")
        
        # Create plots using pre-processed data
        fig1 = create_distribution_plots_from_data(region_data, output_path=output_zonal_gpkg, save_plots=save_plots)
        if fig1:
            figures.append(fig1)
            if save_plots:
                fig1.savefig(output_zonal_gpkg.replace('.gpkg', '_distributions.png'), 
                           dpi=300, bbox_inches='tight')
        
        fig2 = create_boxplot_from_data(region_data, output_path=output_zonal_gpkg, save_plots=save_plots)
        if fig2:
            figures.append(fig2)
            if save_plots:
                fig2.savefig(output_zonal_gpkg.replace('.gpkg', '_boxplot.png'), 
                           dpi=300, bbox_inches='tight')
        
        plt.show()
    
    return zonal_gdf, figures

def create_distribution_plots_from_data(region_data, figsize=(15, 10), output_path=None, save_plots=False):
    """
    Create distribution plots from pre-processed data.

    Args:
        region_data (dict): A dictionary containing region names as keys and their data as values.
        figsize (tuple): The size of the figure to create.
        output_path (str): The path to save the figure.
        save_plots (bool): Whether to save the plots.

    Returns:
        matplotlib.figure.Figure: The created figure.

    """
    output_path = output_path.replace('Data', 'Results')
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
        data_array = np.array(data)
        
        # Create histogram
        ax.hist(data_array, bins=50, alpha=0.7, color=plt.cm.Set3(idx), 
               edgecolor='black', linewidth=0.5)
        
        # Statistics
        mean_val = np.mean(data_array)
        std_val = np.std(data_array)
        count_val = len(data_array)
        
        stats_text = f"Mean: {mean_val:.2f}m\nStd: {std_val:.2f}m\nCount: {count_val}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f"Mean: {mean_val:.2f}m")
        
        ax.set_title(f'{region_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Canopy Height (m)')
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

def create_boxplot_from_data(region_data, figsize=(12, 8), output_path=None, save_plots=False):
    """
    Create boxplot from pre-processed data.

    Args:
        region_data (dict): A dictionary containing region names as keys and their data as values.
        figsize (tuple): The size of the figure to create.
        output_path (str): The path to save the figure.
        save_plots (bool): Whether to save the plots.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """

    output_path = output_path.replace('Data', 'Results')
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
    ax.set_ylabel('Elevation (m)', fontsize=12)
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
#     gpkg_path=1,
#     raster_path="D:/Jerry/UROP Rerta Palapa June2019 CHM.tif", 
#     output_buffer_path=0, 
#     filtering_logic=clip_below_zero,
#     output_zonal_gpkg="Results/Palapa June2019 CHM Statistics.gpkg", 
#     buffer_geom_path="G:/My Drive/UROP/TreatmentRegions.gpkg",
#     create_plots=True,
#     save_plots=True
# )

# print("Plots created and saved successfully!")