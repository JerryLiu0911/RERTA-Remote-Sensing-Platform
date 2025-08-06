import align_coords
import coordinate_extraction
import statistical_modelling
import gis as gis
import matplotlib.pyplot as plt
import numpy as np


def plot_relations(df, target, column, geo = False):
    """
    Plots average canopy openness against CHM for the BC points.
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
        plt.scatter(df[target], df[column]
                    , label=target
                    , color='blue', alpha=0.5
                    , s=10)
        plt.xlabel(target)
        plt.ylabel(column)
        plt.legend()
        plt.show()

def main(paths):

    #coordinate_extraction.extract_corner_coords(paths['coordinates'], paths['result_data'])
    #align_coords.canopy_openness(paths['canopy_openness'], paths['result_data'], paths['canopy_openness_result'], timepoint='post1')
    gis.zonal_statistics(paths['canopy_openness_result'], "G:/My Drive/UROP/UROP Rerta Palapa June2019 CHM.tif", paths['buffered_points'], paths['CHM'], buffer_geom_path=paths['result_data'])

    merged_df = statistical_modelling.load_data(
            [#('GLI', paths['GLI']),
            #('ExG', paths['ExG']),
            #('DEM', paths['DEM']),
            ('CHM', paths['CHM'])], filter = "OPE|BC")

    # print(merged_df)
    BC_df = merged_df#[merged_df['point.label'].str.contains("BC", case=False, na=False)]
    print(BC_df.head())
    # plot_relations(BC_df,'average_canopy_openness', 'mean_CHM', geo = False)

    # coordinate_extraction.extract_corner_coords(paths['coordinates'], paths['result_data'])
    # gis.create_buffer(gis.gpd.read_file(paths['canopy_openness_result']), paths['buffered_points'], buffer_geom=gis.gpd.read_file(paths['result_data']))

    # print([feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'name','average_canopy_openness']])
    statistical_modelling.random_forest_regression(merged_df, 'average_canopy_openness', [feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'name_CHM','average_canopy_openness']])
    # statistical_modelling.random_forest_ensemble(merged_df, 'average_canopy_openness', [feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'average_canopy_openness']])
    # statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if'CHM' in column and column != 'geometry_CHM'], display=False)
    # statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if column not in ['geometry', 'point.label', 'average_canopy_openness']], display=False)


paths = {
'canopy_openness': "Data/3.4-canopy.openness.csv",
'coordinates': "Data/Rerta_veg_plots.gpkg",#"Data/Rerta koordinate 2018_09_24.gpkg",
'result_data': "result_data.gpkg",
'canopy_openness_result': "canopy_openness_result.gpkg",
'buffered_points': "buffered_points.gpkg",
'GLI': "Data/Palapa June2019 GLI statistics.gpkg",
'ExG': "Data/Palapa June2019 ExG statistics.gpkg",
'DEM': "Data/Palapa June2019 DEM statistics.gpkg",
'CHM': "Data/Palapa June2019 CHM statistics.gpkg"
}
main(paths)
