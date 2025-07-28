import align_coords
import coordinate_extraction
import statistical_modelling
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
        plt.xlabel(column)
        plt.ylabel(target)
        plt.legend()
        plt.show()


def main():
    paths = {
    'canopy_openness': "Data/3.4-canopy.openness.csv",
    'coordinates': "Data/Rerta koordinate 2018_09_24.gpkg",
    'GLI': "Data/Palapa June2019 GLI statistics.gpkg",
    'ExG': "Data/Palapa June2019 ExG statistics.gpkg",
    'DEM': "Data/Palapa June2019 DEM statistics.gpkg",
    'CHM': "Data/Palapa June2019 CHM statistics.gpkg"
    }
    coordinate_extraction.extract_coords(paths['coordinates'])
    align_coords.canopy_openness(paths['canopy_openness'],timepoint=2019)
    merged_df = statistical_modelling.load_data(
            [('GLI', paths['GLI']),
            ('ExG', paths['ExG']),
            ('DEM', paths['DEM']),
            ('CHM', paths['CHM'])])

    BC_df = merged_df[merged_df['point.label'].str.contains("BC", case=False, na=False)]
    plot_relations(BC_df,'average_canopy_openness', '_mean_CHM', geo =True)


    print("sync testtt")
    #statistical_modelling.random_forest_regression(merged_df, 'average_canopy_openness', [feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'average_canopy_openness']])
    statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if'CHM' in column], display=False)
    #statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if column not in ['geometry', 'point.label', 'average_canopy_openness']], display=False)

main()