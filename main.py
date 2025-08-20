import align_coords
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
    Plots average canopy openness against CHM for the BC points.

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

def preprocess_dataset(paths, raster_name, target_name, timepoint, filtering_logic):
    """
    To improve efficiency of process different datasets with different requirements, this function automatically calls the align_coord and zonal_statistics
    modules to streamline the preprocessing steps.

    Args:
        paths (dict): The library paths for the data files.
        raster_name (str): The name of the TYPE of raster to be used, used to call the file through paths. e.g. "CHM"
        target_name (str): The name of the target variable (dependent variable).
        timepoint (str or int): The timepoint for the analysis. string for timepoint e.g. "post1", string for date e.g. 2019

    Returns:
        The updated statistic dataframe.
    """
    buffer_types = {
        'canopy_openness' : paths['veg_plots_corner_coordinates'],
        'frogs': paths['100m_transects'],
    }
    getattr(align_coords, target_name)(paths[f'{target_name}_csv'], buffer_types[target_name], paths[f'{target_name}_result'], timepoint=timepoint)
    zonal_gdf, figures = gis.zonal_statistics(gpkg_path=paths[f'{target_name}_result'],
                         raster_path=paths[f'{raster_name}_tif'],
                         output_buffer_path=paths['buffered_points'],
                         filtering_logic=filtering_logic,
                         output_zonal_gpkg=paths[f'{raster_name}'],
                         buffer_geom_path=buffer_types[target_name],
                         save_plots=True)
    return zonal_gdf, figures

def main(paths):
    # preprocess_dataset(paths, 'Palapa June2019 CHM', 'frogs', timepoint=2019, filtering_logic=gis.clip_below_zero)
    # preprocess_dataset(paths, 'Palapa June2019 GLI', 'frogs', timepoint=2019, filtering_logic=gis.clip_below_zero)
    # preprocess_dataset(paths, 'Palapa June2019 ExG', 'frogs', timepoint=2019, filtering_logic=gis.clip_below_zero)

    ### Combining and analyzing data into dataframes ###
    # print(gis.gpd.read_file(paths['frogs']))
    # region_data = gis.get_region_data(paths['frogs'], 'Frog.abundance')
    # gis.create_boxplot_from_data(region_data, 'Frog abundance')
    # gis.create_distribution_plots_from_data(region_data, 'Frog abundance')
    # plt.show()
    
    merged_df = statistical_modelling.load_data(
            [#('GLI', paths['GLI'])
            #('ExG', paths['ExG']),
            #('DEM', paths['DEM']),
            ('CHM', paths['CHM'])], filter = "OPE|BC")

    print(f"Finished merging columns : {merged_df.columns}")
    pca_results, interpretation = statistical_modelling.comprehensive_PCA_analysis(merged_df)
    
    # Option 2: Use specific features
    # features = ['mean_CHM', 'std_CHM', 'max_CHM', 'average_canopy_openness']
    # pca_results, interpretation = comprehensive_PCA_analysis(merged_df, target_columns=features)
    
    # You can also access specific results:
    print(f"\nKey findings:")
    print(f"PC1 explains {pca_results['explained_variance_ratio'][0]*100:.1f}% of variance")
    print(f"Most important variables for PC1: {interpretation['pc1_key_variables'][:3]}")
    print(f"Treatment separation quality: {interpretation['separation_quality']}")

    # BC_df = merged_df[merged_df['point.label'].str.contains("BC", case=False, na=False)]
    # print(BC_df.head())
    # plot_relations(BC_df,'average_canopy_openness', 'mean_CHM', geo = False)
    # plot_relations(merged_df,'Frog.abundance', 'mean_CHM', geo = False)


    features = [column for column in merged_df.columns if column not in ['geometry', 'point.label', 'treatment', 'Frog.abundance', 'Frog.richness']]
    print(features)
    # feature_diagnostics(merged_df, 'average_canopy_openness', features)
    # print("\n \n \n \n \n \n \n")
    # analyze_chm_correlations(merged_df, features)
    # selected_features = smart_feature_selection_pipeline(merged_df, 'average_canopy_openness', features)



    ### Statistical Modelling ###

    # statistical_modelling.random_forest_regression(merged_df, 'Frog.abundance', ['canopy_openness_CHM', 'mean_CHM'])
    # statistical_modelling.random_forest_ensemble(merged_df, 'average_canopy_openness', [feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'average_canopy_openness']])
    # statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if'CHM' in column and column != 'geometry_CHM' and column != 'name_CHM'], display=False)
    # statistical_modelling.multi_linear_regression_display(merged_df, 'Frog.abundance', features, display=False)
    statistical_modelling.enhanced_multi_linear_regression_display(merged_df, 'Frog.abundance', features, display=False)


paths = {
#Raw input files (csv, coordinates)
'canopy_openness_csv': "Data/3.4-canopy.openness.csv", # USE WITH buffered_points !! According to protocol
'frogs_csv': "Data/4.3_Frogs.csv", # USE WITH 100m_transects !! According to protocol
'veg_plots_coordinates': "Data/Palapa_veg_plots.gpkg", #"Data/Rerta koordinate 2018_09_24.gpkg",
'abcd_coordinates': "Data/Palapa_ABCD.gpkg", 
'Palapa June2019 CHM_tif' : "G:/My Drive/UROP/UROP Rerta Palapa June2019 CHM.tif",
'Palapa June2019 ExG_tif' : "G:/My Drive/UROP/UROP Rerta Palapa June2019 ExG.tif",
'Palapa June2019 GLI_tif' : "G:/My Drive/UROP/UROP Rerta Palapa June2019 GLI.tif",
'Palapa June2019 DTM_tif' : "G:/My Drive/UROP/UROP Rerta Palapa June2019 DTM.tif",

# Buffer geometries
'veg_plots_centre_coordinates': "Data/Palapa_veg_plots_centres.gpkg",
'veg_plots_corner_coordinates': "Data/Palapa_veg_plots_corners.gpkg",
'treatment_buffers' : "Data/TreatmentRegions.gpkg",
'100m_transects' : "Data/Palapa_transects_buffer.gpkg",

# Processed files
'result_data': "result_data.gpkg",
'canopy_openness_result': "canopy_openness_result.gpkg",
'frogs_result': "Frogs_result.gpkg",
'buffered_points': "buffered_points.gpkg",
'Palapa June2019 CHM': "Data/Palapa June2019 CHM statistics.gpkg",
'GLI': "Data/Palapa June2019 GLI statistics.gpkg",
'ExG': "Data/Palapa June2019 ExG statistics.gpkg",
'DEM': "Data/Palapa June2019 DEM statistics.gpkg",
'CHM': "Data/Palapa June2019 CHM statistics.gpkg"
}

#preprocess_dataset(paths, 'CHM', 'frogs', timepoint=2019, filtering_logic=gis.clip_below_zero)
main(paths)
