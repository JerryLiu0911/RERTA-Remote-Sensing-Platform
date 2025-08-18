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

def main(paths):

    ### Extracting and aligning coordinates ###
    # align_coords.frogs("Data/4.3_Frogs.csv", paths['100m_transects'], paths['frogs'], timepoint=2019)
    # coordinate_extraction.extract_corner_coords(paths['coordinates'], paths['result_data'])
    # align_coords.canopy_openness(paths['canopy_openness'], paths['result_data'], paths['canopy_openness_result'], timepoint=2019)
    # zonal_gdf, figures = gis.zonal_statistics(gpkg_path=paths['canopy_openness_result'],
    #                     raster_path="G:/My Drive/UROP/UROP Rerta Palapa June2019 CHM.tif",
    #                     output_buffer_path=paths['buffered_points'], 
    #                     filtering_logic=gis.clip_below_zero,
    #                     output_zonal_gpkg=paths['CHM'],
    #                     buffer_geom_path=paths['result_data'],
    #                     save_plots=True)


    # zonal_gdf, figures = gis.zonal_statistics(gpkg_path=paths['canopy_openness_result'],
    #                     raster_path="G:/My Drive/UROP/UROP RERTA Palapa June2019 ExG.tif",
    #                     output_buffer_path=paths['buffered_points'], 
    #                     #filtering_logic=gis.clip_below_zero,
    #                     output_zonal_gpkg=paths['ExG'],
    #                     buffer_geom_path=paths['result_data'])


    # gis.raster_calculation_zonal(gpkg_path=paths['canopy_openness_result'],
    #                               raster_path="G:/My Drive/UROP/imagery/rerta_palapa_ortho20190501.tif",
    #                               output_buffer_path=paths['buffered_points'],
    #                               output_zonal_gpkg=paths['GLI'],
    #                               buffer_geom_path=paths['result_data'],
    #                               calculation_func=gis.create_calculation_functions()['Green_Leaf_Index'],
    #                               calculation_name='Green_Leaf_Index')


    ### Combining and analyzing data into dataframes ###
    print(gis.gpd.read_file(paths['frogs']))
    region_data = gis.get_region_data(paths['frogs'], 'Frog.abundance')
    gis.create_boxplot_from_data(region_data, 'Frog abundance')
    gis.create_distribution_plots_from_data(region_data, 'Frog abundance')
    plt.show()
    
    merged_df = statistical_modelling.load_data(
            [#('GLI', paths['GLI'])
            #('ExG', paths['ExG']),
            #('DEM', paths['DEM']),
            ('CHM', paths['CHM'])], filter = "OPE|BC")

    print(f"Finished merging columns : {merged_df.columns}")

    # BC_df = merged_df[merged_df['point.label'].str.contains("BC", case=False, na=False)]
    # print(BC_df.head())
    # plot_relations(BC_df,'average_canopy_openness', 'mean_CHM', geo = False)
    plot_relations(merged_df,'Frog.abundance', 'mean_CHM', geo = False)


    features = [column for column in merged_df.columns if column not in ['geometry', 'point.label', 'treatment', 'Frog.abundance', 'Frog.richness']]
    print(features)
    # feature_diagnostics(merged_df, 'average_canopy_openness', features)
    # print("\n \n \n \n \n \n \n")
    # analyze_chm_correlations(merged_df, features)
    # selected_features = smart_feature_selection_pipeline(merged_df, 'average_canopy_openness', features)



    ### Statistical Modelling ###

    # statistical_modelling.random_forest_regression(merged_df, 'average_canopy_openness', ['canopy_openness_CHM', 'mean_ExG', 'mean_CHM'])
    # statistical_modelling.random_forest_ensemble(merged_df, 'average_canopy_openness', [feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'average_canopy_openness']])
    # statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if'CHM' in column and column != 'geometry_CHM' and column != 'name_CHM'], display=False)
    statistical_modelling.multi_linear_regression_display(merged_df, 'Frog.abundance', features, display=False)


paths = {
#Raw input files (csv, coordinates)
'canopy_openness': "Data/3.4-canopy.openness.csv", # USE WITH buffered_points !! According to protocol
'frog_biodiversity': "Data/3.4-frog.biodiversity.csv", # USE WITH 100m_transects !! According to protocol
'veg_plots_coordinates': "Data/Palapa_veg_plots.gpkg", #"Data/Rerta koordinate 2018_09_24.gpkg",
'abcd_coordinates': "Data/Palapa_ABCD.gpkg", 
'CHM_tif' : "G:/My Drive/UROP/UROP Rerta Palapa June2019 CHM.tif",
'ExG_tif' : "G:/My Drive/UROP/UROP Rerta Palapa June2019 ExG.tif",
'GLI_tif' : "G:/My Drive/UROP/UROP Rerta Palapa June2019 GLI.tif",
'DTM_tif' : "G:/My Drive/UROP/UROP Rerta Palapa June2019 DTM.tif",

# Buffer geometries
'treatment_buffers' : "Data/TreatmentRegions.gpkg",
'100m_transects' : "Data/Palapa_transects_buffer.gpkg",

# Processed files
'result_data': "result_data.gpkg",
'canopy_openness_result': "canopy_openness_result.gpkg",
'frogs': "Frogs_result.gpkg",
'buffered_points': "buffered_points.gpkg",
'GLI': "Data/Palapa June2019 GLI statistics.gpkg",
'ExG': "Data/Palapa June2019 ExG statistics.gpkg",
'DEM': "Data/Palapa June2019 DEM statistics.gpkg",
'CHM': "Data/Palapa June2019 CHM statistics.gpkg"
}
main(paths)