import os

from scipy import stats
from shapely import buffer
import load_field_data
import coordinate_extraction
import statistical_modelling
import gis as gis
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm


def plot_relations(df, target, column, geo = False):
    """
    Generic function to plot different features in selected points.

    Args:
        df: The DataFrame containing the data.
        target: The target variable (dependent variable).
        column: The independent variable (feature).
        geo: Whether to plot the data with respect to geographical positions (default: False).

    Returns:
        None
    """
    if geo:
        plt.scatter(df[target], df['point.label']
                    , label=target
                    , color='blue', alpha=0.5)
        plt.scatter(df[column], df['point.label']
                    , label= 'CHM'
                    , color='red', alpha=0.5)
        plt.xlabel(column + ' and ' + target)
        plt.ylabel('point.label')
        plt.legend()
        plt.show()
        
    else:
        # Example: frog_abundance vs. canopy_openness
        x = df[column]
        y = df[target]

        # Scatterplot
        sns.scatterplot(x=x, y=y, alpha=0.7)

        # Fit LOESS smoother
        lowess = sm.nonparametric.lowess
        smoothed = lowess(y, x, frac=0.4)  # frac controls smoothing (0.2–0.5 typical)

        # Plot LOESS line
        plt.plot(smoothed[:,0], smoothed[:,1], color='red', linewidth=2)
        plt.xlabel("Canopy openness (CHM)")
        plt.ylabel("Frog abundance")
        plt.title("Frog abundance vs Canopy openness with LOESS smooth")
        plt.show()
        # plt.scatter(df[target], df[column]
        #             , label=target
        #             , color='blue', alpha=0.5
        #             , s=10)
        # plt.xlabel(target)
        # plt.ylabel(column)
        # plt.legend()
        # plt.show()

def preprocess_dataset(paths, raster_name, buffer, filtering_logic = None, proxies = None, show_plots = False):
    """
    To improve efficiency of process different datasets with different requirements, this function automatically calls the zonal_statistics
    modules to streamline the preprocessing steps.

    Args:
        paths (dict): The library paths for the data files.
        raster_name (str): The name of the TYPE of raster to be used, used to call the file through paths. e.g. "CHM"
        buffer (str): The name of the target geopackage dataframe which contains both the geometry and field data.(see align_coords.align_coords()) e.g. "veg_plots_corner_coordinates"
        filtering_logic (function): A function to apply filtering to the data, e.g. gis.clip_below_zero
        proxies (dict): A dictionary of proxy functions to apply to the data, e.g. gis.canopy_openness_proxy

    Returns:
        The updated statistic dataframe.
    """
    zonal_gdf,_ = gis.zonal_statistics(gpkg_path=paths[f'{buffer}_result'],
                         raster_path=paths[f'{raster_name}_tif'],
                         filtering_logic=filtering_logic,
                         output_zonal_gpkg=paths[f'{raster_name}'],
                         proxies=proxies,
                         buffer_geom_path=paths[buffer],
                         value=raster_name.split(' ')[-1] if raster_name.split(' ')[-1] != 'ortho' else 'intensity',
                         save_plots=True,
                         show_plots=show_plots)
    return zonal_gdf

def plot_features(df, features, target, group=None):
    """
    Plots resulting histogram of feature importances
    """
    n_features = len(features)
    n_cols = min(2, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for i, feature in enumerate(features):
        if feature in df.columns:
            ax = axes[i]
            if group and group in df.columns:
                region_data = gis.get_region_data(df, feature)
                gis.create_boxplot_from_data(region_data, feature, ax=ax)
            else:
                sns.histplot(df[feature], kde=True, ax=ax)
                ax.set_title(f'Distribution of {feature}')
            ax.set_xlabel(feature)
            ax.set_ylabel('Value')
        else:
            axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'Results/feature_distributions_{target}.png', dpi=300)
    # plt.show()

def main(paths):

    ### VARIABLES ###
    buffer = "veg_plots_corner_coordinates"
    timepoint = ("2021-1-1", "2024-12-31") # (start_date, end_date) for filtering field data based on date. Alernatives are 2019 as an int for a year, or "post2" by timepoint.


    buffer_types = {
    'veg_plots_corner_coordinates' : ['canopy_openness','erosion_sticks','seed_removal'],
    '100m_transects' : ['frogs']
    }


    ###=================== Only need to run this once when processing a new dataset :  ========================= ###

    # dataframes = []

    # for target_name in buffer_types[buffer]:
    #     df = getattr(load_field_data, f'load_{target_name}')(paths[f'{target_name}_csv'], timepoint=timepoint)
    #     print(df)
    #     dataframes.append(df)

    # # Extracts coordinates for mapping ecological features
    # getattr(coordinate_extraction, f'extract_{buffer}')(paths['veg_plots_coordinates'], paths[buffer]) # Source path for geopackage needs to be explicitly defined

    # # Align coordinates of field data with extracted coordinates  
    # load_field_data.align_coords(dataframes, paths[buffer], paths[f'{buffer}_result'], filter=None)  # filter is a regex pattern to match specific point labels e.g. r"OPC|BC" excludes OPE

    # # Preprocess each dataset according the geometries and merges with field data 
    # preprocess_dataset(paths, 'Palapa July2025 DEM', buffer, filtering_logic=gis.clip_below_zero, proxies=gis.canopy_openness_proxy)
    # preprocess_dataset(paths, 'Palapa July2025 GLI', buffer, filtering_logic=gis.remove_outliers, proxies=gis.GLCM)
    # preprocess_dataset(paths, 'Palapa July2025 ReNDVI', buffer, filtering_logic=gis.remove_outliers, proxies=gis.GLCM)
    # preprocess_dataset(paths, 'Palapa July2025 Clre', buffer, filtering_logic=gis.remove_outliers, proxies=gis.GLCM)
    # preprocess_dataset(paths, 'Palapa July2025 GNDVI', buffer, filtering_logic=gis.remove_outliers, proxies=gis.GLCM)
    # preprocess_dataset(paths, 'Palapa July2025 NDVI', buffer, filtering_logic=gis.remove_outliers, proxies=gis.GLCM)
    # preprocess_dataset(paths, 'Palapa July2025 ortho', buffer, proxies=gis.GLCM)

    ###=================== Preprocessing finished ==================== ###



    ## Display ecological data by treatment ###
    # region_data = gis.get_region_data(paths['Palapa July2025 DEM'], 'canopy_openness')
    # gis.create_boxplot_from_data(region_data, 'canopy_openness')
    # gis.create_distribution_plots_from_data(region_data, 'canopy_openness')
    # plt.show()




    targets = [col for col in load_field_data.gpd.read_file(paths[f'{buffer}_result']).columns if col not in ['geometry', 'point.label', 'treatment']] # Removes repeated columns when merging datasets. 

    print(targets)

    # Option 1: Use only UAV derived data / vegetation indexes
    option = 'vi'

    merged_df = statistical_modelling.load_data( # Merging each index  
            [
            ('DEM', paths['Palapa July2025 DEM']),
            ('GLI', paths['Palapa July2025 GLI']),
            ('Clre', paths['Palapa July2025 Clre']),
            ('ReNDVI', paths['Palapa July2025 ReNDVI']),
            ('GNDVI', paths['Palapa July2025 GNDVI']),
            ('NDVI', paths['Palapa July2025 NDVI']),
            # ('band1', paths['band1']),
            # ('band2', paths['band2']),
            # ('band3', paths['band3']),
            # ('band4', paths['band4']),
            # ('band5', paths['band5']),
            # ('band6', paths['band6']),
            # ('band7', paths['band7']),
            ], targets,
            filter = None)
    
    all_features = [col for col in merged_df.columns if any(x in col for x in ['DEM', 'GLI', 'Clre', 'ReNDVI', 'GNDVI', 'NDVI', 'band'])]
    # print(f"Identified target variables: {targets}")
    # print(f"Finished merging columns : {merged_df.columns}")
    # print("Final merged dataframe :")
    # print(merged_df[:26])
    # print(len(merged_df))




    ###================ Post-processing after merging, renaming for conventions and transformations ====================

    transformations = {
        'proportion': statistical_modelling.arcsinc_sqrt_transform,
        'canopy_openness': statistical_modelling.arcsinc_sqrt_transform,
        'abundance': statistical_modelling.log_transform,
        'richness' : statistical_modelling.log_transform
    }

    print(f"Sample size before removing missing values: {len(merged_df)}")
    for col in merged_df.columns:
        if col in targets + all_features:   # Chooses all meaningful columns (aka excluding any geometric parameters or identifiers)
            merged_df[col] = load_field_data.pd.to_numeric(merged_df[col], errors='coerce')

            if merged_df[col].isna().all():
                print(f"Warning: Column '{col}' contains only NaN values after conversion to numeric.")

            merged_df = merged_df.rename(columns={col: col.replace(' ', '_').replace('.', '_')}) # Convert column names to snake_case as to not break any downstream processing from parsing
        if any([x in col for x in transformations.keys()]):
            for key in transformations.keys():
                if key in col:
                    # print(merged_df[col])
                    # print(f"Applying {transformations[key].__name__} to {col}") # Apply transformation
                    merged_df[col] = transformations[key](merged_df[col])

    print(f"Sample size after removing missing values: {len(merged_df)}")

    ###======================= Post-processing finished =======================



    # Refreshes all_features and targets after post-processing to match column names (prevents key errors)
    all_features = [col for col in merged_df.columns if any(x in col for x in ['DEM', 'GLI', 'Clre', 'ReNDVI', 'GNDVI', 'NDVI', 'band'])]
    targets = [col.replace(' ', '_').replace('.', '_') for col in load_field_data.gpd.read_file(paths[f'{buffer}_result']).columns if col not in ['geometry', 'point.label', 'treatment']]

    # Statistical modelling for each target/response variable
    for target in targets:
        features = statistical_modelling.smart_feature_selection_pipeline(merged_df, target, all_features)
#        plot_features(merged_df, features, target, group='treatment')
#        statistical_modelling.multi_linear_regression_display(merged_df, target, features, display=True, option=option)
#        statistical_modelling.linear_mixed_model(merged_df, target, features, display=True, option=option)
        statistical_modelling.random_forest_regression(merged_df, target, features=features, display=True, option=option)







    print("OPTION 2")
    option = 'bands'
    targets = [col for col in load_field_data.gpd.read_file(paths[f'{buffer}_result']).columns if col not in ['geometry', 'point.label', 'treatment']] # Removes repeated columns when merging datasets. 
    # Option 2: Using only raw band data
    merged_df = statistical_modelling.load_data(
        [
        # ('DEM', paths['Palapa July2025 DEM']),
        # ('GLI', paths['Palapa July2025 GLI']),
        # ('Clre', paths['Palapa July2025 Clre']),
        # ('ReNDVI', paths['Palapa July2025 ReNDVI']),
        # ('GNDVI', paths['Palapa July2025 GNDVI']),
        # ('NDVI', paths['Palapa July2025 NDVI']),
        ('band1', paths['band1']),
        ('band2', paths['band2']),
        ('band3', paths['band3']),
        ('band4', paths['band4']),
        ('band5', paths['band5']),
        ('band6', paths['band6']),
        ('band7', paths['band7']),
        ], targets,
            filter = None)
    
    
    ###================ Post-processing after merging, renaming for conventions and transformations ====================


    all_features = [col for col in merged_df.columns if any(x in col for x in ['DEM', 'GLI', 'Clre', 'ReNDVI', 'GNDVI', 'NDVI', 'band'])]
    

    transformations = {
        'proportion': statistical_modelling.arcsinc_sqrt_transform,
        'canopy_openness': statistical_modelling.arcsinc_sqrt_transform,
        'abundance': statistical_modelling.log_transform,
        'richness' : statistical_modelling.log_transform
    }

    print(f"Sample size before removing missing values: {len(merged_df)}")
    for col in merged_df.columns:
        if col in targets + all_features:   # Chooses all meaningful columns (aka excluding any geometric parameters or identifiers)
            merged_df[col] = load_field_data.pd.to_numeric(merged_df[col], errors='coerce')

            if merged_df[col].isna().all():
                print(f"Warning: Column '{col}' contains only NaN values after conversion to numeric.")

            merged_df = merged_df.rename(columns={col: col.replace(' ', '_').replace('.', '_')}) # Convert column names to snake_case as to not break any downstream processing from parsing
        if any([x in col for x in transformations.keys()]):
            for key in transformations.keys():
                if key in col:
                    # print(merged_df[col])
                    # print(f"Applying {transformations[key].__name__} to {col}") # Apply transformation
                    merged_df[col] = transformations[key](merged_df[col])

    print(f"Sample size after removing missing values: {len(merged_df)}")

    targets = [col.replace(' ', '_').replace('.', '_') for col in load_field_data.gpd.read_file(paths[f'{buffer}_result']).columns if col not in ['geometry', 'point.label', 'treatment']]

    ###======================= Post-processing finished =======================
    
    pca_results, interpretation, X_pca, treatments = statistical_modelling.comprehensive_PCA_analysis(merged_df, target_columns=all_features, display=False)
    print(f'Number of PCA components: {X_pca.shape[1]}')

    for i in range(4):             # Add to merged dataframe
        merged_df[f'PC{i+1}'] = X_pca[:, i]
        print(f"merging PC{i+1} to merged_df")

    all_features = [col for col in merged_df.columns if 'PC' in col] # Using PCA components as features

        # Statistical modelling for each target/response variable
    for target in targets:
        # features = statistical_modelling.smart_feature_selection_pipeline(merged_df, target, all_features)
        # plot_features(merged_df, all_features, target, group='treatment')
        # statistical_modelling.multi_linear_regression_display(merged_df, target, all_features, display=True, option=option)
        # statistical_modelling.linear_mixed_model(merged_df, target, all_features, display=True, option=option)
        statistical_modelling.random_forest_regression(merged_df, target, features=all_features, display=True, option=option)

    # You can also access specific results:
    # print(f"\nKey findings:")
    # print(f"PC1 explains {pca_results['explained_variance_ratio'][0]*100:.1f}% of variance")
    # print(f"Most important variables for PC1: {interpretation['pc1_key_variables'][:3]}")
    # print(f"Treatment separation quality: {interpretation['separation_quality']}")
    


    print("OPTION 3")
    option = 'bands+vi'
    targets = [col for col in load_field_data.gpd.read_file(paths[f'{buffer}_result']).columns if col not in ['geometry', 'point.label', 'treatment']] # Removes repeated columns when merging datasets. 
    # Option 3: Use both band data and VIs
    merged_df = statistical_modelling.load_data(
        [
        ('DEM', paths['Palapa July2025 DEM']),
        ('GLI', paths['Palapa July2025 GLI']),
        ('Clre', paths['Palapa July2025 Clre']),
        ('ReNDVI', paths['Palapa July2025 ReNDVI']),
        ('GNDVI', paths['Palapa July2025 GNDVI']),
        ('NDVI', paths['Palapa July2025 NDVI']),
        ('band1', paths['band1']),
        ('band2', paths['band2']),
        ('band3', paths['band3']),
        ('band4', paths['band4']),
        ('band5', paths['band5']),
        ('band6', paths['band6']),
        ('band7', paths['band7']),
        ], targets,
            filter = None)
    
    
    ###================ Post-processing after merging, renaming for conventions and transformations ====================


    all_features = [col for col in merged_df.columns if any(x in col for x in ['DEM', 'GLI', 'Clre', 'ReNDVI', 'GNDVI', 'NDVI', 'band'])]
    

    transformations = {
        'proportion': statistical_modelling.arcsinc_sqrt_transform,
        'canopy_openness': statistical_modelling.arcsinc_sqrt_transform,
        'abundance': statistical_modelling.log_transform,
        'richness' : statistical_modelling.log_transform
    }

    print(f"Sample size before removing missing values: {len(merged_df)}")
    for col in merged_df.columns:
        if col in targets + all_features:   # Chooses all meaningful columns (aka excluding any geometric parameters or identifiers)
            merged_df[col] = load_field_data.pd.to_numeric(merged_df[col], errors='coerce')

            if merged_df[col].isna().all():
                print(f"Warning: Column '{col}' contains only NaN values after conversion to numeric.")

            merged_df = merged_df.rename(columns={col: col.replace(' ', '_').replace('.', '_')}) # Convert column names to snake_case as to not break any downstream processing from parsing
        if any([x in col for x in transformations.keys()]):
            for key in transformations.keys():
                if key in col:
                    # print(merged_df[col])
                    # print(f"Applying {transformations[key].__name__} to {col}") # Apply transformation
                    merged_df[col] = transformations[key](merged_df[col])

    print(f"Sample size after removing missing values: {len(merged_df)}")

    targets = [col.replace(' ', '_').replace('.', '_') for col in load_field_data.gpd.read_file(paths[f'{buffer}_result']).columns if col not in ['geometry', 'point.label', 'treatment']]

    ###======================= Post-processing finished =======================
    

    all_features = [col for col in merged_df.columns if any(x in col for x in ['DEM', 'GLI', 'Clre', 'ReNDVI', 'GNDVI', 'NDVI', 'band'])]

        # Statistical modelling for each target/response variable
    for target in targets:
        features = statistical_modelling.smart_feature_selection_pipeline(merged_df, target, all_features)
        # plot_features(merged_df, features, target, group='treatment')
        # statistical_modelling.multi_linear_regression_display(merged_df, target, features, display=True, option=option)
        # statistical_modelling.linear_mixed_model(merged_df, target, features, display=True, option=option)
        statistical_modelling.random_forest_regression(merged_df, target, features=features, display=True, option=option)


paths = {
# Raw files
    ## Field data (csv)
    'canopy_openness_csv': "Data/3.4-canopy.openness.csv", # USE WITH buffered_points !! According to protocol
    'frogs_csv': "Data/4.3_Frogs.csv", # USE WITH 100m_transects !! According to protocol
    'erosion_sticks_csv': "Data/1.2_Erosion-sticks.csv", # USE WITH buffered_points !! According to protocol
    'seed_removal_csv': "Data/6.5_Seed-removal.csv", # USE WITH buffered_points !! According to protocol

    ## point coordinates
    'veg_plots_coordinates': "Data/Palapa_veg_plots.gpkg", #"Data/Rerta koordinate 2018_09_24.gpkg",
    'abcd_coordinates': "Data/Palapa_ABCD.gpkg",
    '100m_transects_coordinates': "Data/Palapa_transects.gpkg",

    ## Palapa 2019
    'Palapa June2019 CHM_tif' : "G:/My Drive/UROP/UROP Rerta Palapa June2019 CHM.tif",
    'Palapa June2019 ExG_tif' : "G:/My Drive/UROP/UROP Rerta Palapa June2019 ExG.tif",
    'Palapa June2019 GLI_tif' : "G:/My Drive/UROP/UROP Rerta Palapa June2019 GLI.tif",
    'Palapa June2019 DTM_tif' : "G:/My Drive/UROP/UROP Rerta Palapa June2019 DTM.tif",

    ## Palapa 2025
    'Palapa July2025 GLI_tif' : "D:/Jerry/Palapa July2025 GLI.tif",
    'Palapa July2025 NDVI_tif' : "D:/Jerry/Palapa July2025 NDVI.tif",
    'Palapa July2025 DEM_tif' : "D:/Jerry/Palapa July2025 DEM.tif",
    'Palapa July2025 Clre_tif' : "D:/Jerry/Palapa July2025 Clre.tif",
    'Palapa July2025 ReNDVI_tif' : "D:/Jerry/Palapa July2025 ReNDVI.tif",
    'Palapa July2025 GNDVI_tif' : "D:/Jerry/Palapa July2025 GNDVI.tif",
    'Palapa July2025 ortho_tif' : "D:/Jerry/UROP Rerta Palapa July2025 orthomosaic.tif",

    ## Kandista 2025
    'Kandista July2025 GLI_tif' : "D:/Jerry/Kandista July2025 GLI.tif",
    'Kandista July2025 NDVI_tif' : "D:/Jerry/Kandista July2025 NDVI.tif",
    'Kandista July2025 DEM_tif' : "D:/Jerry/Kandista July2025 DEM.tif",
    'Kandista July2025 Clre_tif' : "D:/Jerry/Kandista July2025 Clre.tif",
    'Kandista July2025 ReNDVI_tif' : "D:/Jerry/Kandista July2025 ReNDVI.tif",
    'Kandista July2025 GNDVI_tif' : "D:/Jerry/Kandista July2025 GNDVI.tif",
    'Kandista July2025 ortho_tif' : "D:/Jerry/Kandista July2025 ortho.tif",

# Buffer geometries
'veg_plots_centre_coordinates': "Data/Palapa_veg_plots_centres.gpkg",
'veg_plots_corner_coordinates': "Data/Palapa_veg_plots_corners.gpkg",
'treatment_buffers' : "Data/TreatmentRegions.gpkg",
'100m_transects' : "Data/Palapa_transects_buffer.gpkg",

# Zonal statistics

    ## Palapa 2019
    'Palapa June2019 CHM': "Data/Palapa June2019 CHM statistics.gpkg",
    'Palapa June2019 ExG': "Data/Palapa June2019 ExG statistics.gpkg",
    'Palapa June2019 GLI': "Data/Palapa June2019 GLI statistics.gpkg",
    'Palapa June2019 DTM': "Data/Palapa June2019 DTM statistics.gpkg",
    ## Palapa 2025
    'Palapa July2025 CHM': "Data/Palapa July2025 CHM statistics.gpkg",
    'Palapa July2025 GLI': "Data/Palapa July2025 GLI statistics.gpkg",
    'Palapa July2025 ExG': "Data/Palapa July2025 ExG statistics.gpkg",
    'Palapa July2025 DEM': "Data/Palapa July2025 DEM statistics.gpkg",
    'Palapa July2025 ReNDVI': "Data/Palapa July2025 ReNDVI statistics.gpkg",
    'Palapa July2025 GNDVI': "Data/Palapa July2025 GNDVI statistics.gpkg",
    'Palapa July2025 NDVI': "Data/Palapa July2025 NDVI statistics.gpkg",
    'Palapa July2025 ortho': "Data/Palapa July2025 ortho statistics.gpkg",
    'Palapa July2025 Clre': "Data/Palapa July2025 Clre statistics.gpkg",

# Processed files
'result_data': "result_data.gpkg",
'veg_plots_corner_coordinates_result': "Data/Palapa_veg_plots_result.gpkg",
'100m_transects_result': "Data/Palapa_100m_transects_result.gpkg",
# 'canopy_openness_result': "canopy_openness_result.gpkg",
# 'frogs_result': "Frogs_result.gpkg",
'GLI': "Data/Palapa June2019 GLI statistics.gpkg", # FORMATTING TO BE DISCUSSED
'ExG': "Data/Palapa June2019 ExG statistics.gpkg",
'DEM': "Data/Palapa June2019 DEM statistics.gpkg",
'CHM': "Data/Palapa June2019 CHM statistics.gpkg",
'band1': "Data/Palapa July2025 ortho statistics_band1.gpkg",
'band2': "Data/Palapa July2025 ortho statistics_band2.gpkg",
'band3': "Data/Palapa July2025 ortho statistics_band3.gpkg",
'band4': "Data/Palapa July2025 ortho statistics_band4.gpkg",
'band5': "Data/Palapa July2025 ortho statistics_band5.gpkg",
'band6': "Data/Palapa July2025 ortho statistics_band6.gpkg",
'band7': "Data/Palapa July2025 ortho statistics_band7.gpkg"

}

# rasters = ['Palapa July2025 NDVI', 'Palapa July2025 GNDVI', 'Palapa July2025 GLI', 'Palapa July2025 Clre', 'Palapa July2025 ReNDVI']

# for raster_name in rasters:
#     gis.plot_index_kde_sampled(paths[f'{raster_name}_tif'], value=raster_name.split(' ')[-1], output_path=f"D:/Jerry/{raster_name} Raster Histogram.png")

main(paths)
