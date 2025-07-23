import align_coords
import coordinate_extraction
import statistical_modelling


paths = {
    'canopy_openness': "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/3.4-canopy.openness.csv",
    'coordinates': "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/Rerta koordinate 2018_09_24.gpkg",
    'GLI': "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/Palapa June2019 GLI statistics.gpkg",
    'ExG': "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/Palapa June2019 ExG statistics.gpkg",
    'DEM': "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/Palapa June2019 DEM statistics.gpkg"
}
coordinate_extraction.extract_coords(paths['coordinates'])
align_coords.canopy_openness(paths['canopy_openness'])
merged_df = statistical_modelling.load_data([('GLI',paths['GLI']),
          ('ExG',paths['ExG']),
          ('DEM', paths['DEM'])])
# Assuming you have your x and y data and the calculated slope (m) and intercept (b)
# Replace with your actual data and calculated values

statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if ('DEM' in column) & (column!='geometry_DEM')])