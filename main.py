import align_coords
import coordinate_extraction
import statistical_modelling
import matplotlib.pyplot as plt
import numpy as np


paths = {
    'canopy_openness': "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/3.4-canopy.openness.csv",
    'coordinates': "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/Rerta koordinate 2018_09_24.gpkg",
    'GLI': "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/Palapa June2019 GLI statistics.gpkg",
    'ExG': "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/Palapa June2019 ExG statistics.gpkg",
    'DEM': "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/Palapa June2019 DEM statistics.gpkg",
    'CHM': "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/Palapa June2019 CHM statistics.gpkg"
}
coordinate_extraction.extract_coords(paths['coordinates'])
align_coords.canopy_openness(paths['canopy_openness'])
merged_df = statistical_modelling.load_data(
        [('GLI', paths['GLI']),
         ('ExG', paths['ExG']),
         ('DEM', paths['DEM']),
         ('CHM', paths['CHM'])])

BC_df = merged_df[merged_df['point.label'].str.contains("BC", case=False, na=False)]


def plot_relations(df, target, column, geo = False):
    """
    Plots average canopy openness against CHM for the BC points.
    """
    if geo:
        plt.scatter(df[target], df['point.label']
                    , label=target
                    , color='blue', alpha=0.5
                    , s=10)
        plt.scatter(df[column], df['point.label']
                    , label=column
                    , color='red', alpha=0.5
                    , s=10)
        plt.xlabel(column + 'and ' + target)
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

# plot_relations(BC_df,'average_canopy_openness', '_mean_CHM', geo =True)

print(type(merged_df[[feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'average_canopy_openness']]]))

# print(np.array(merged_df[[feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'average_canopy_openness']]]))
statistical_modelling.random_forest_regression(merged_df, 'average_canopy_openness', [feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'average_canopy_openness']])
#statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if'CHM' in column], display=False)
#statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if column not in ['geometry', 'point.label', 'average_canopy_openness']], display=False)