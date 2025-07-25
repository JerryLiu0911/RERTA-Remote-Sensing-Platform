import pandas as pd
import geopandas as gpd
import fiona
import re

''' 
Attaches geometry coordinates extracted from coordinate_extraction to the given csv files. Ideally, this would be changed as we obtain the actually geometry data from the csv files.
As each csv file is different, this function will need to be edited for each csv file, especially for those without geometry data.
'''

def standardize_names_for_canopy_openness(name):
    """
    Standardizes the 'point.label' column into the format 'treatment-EAST/WEST-transect-BC/OPE/OPC'
    from the GeoDataFrame by replacing specific patterns. 
    ***SPECIFICALLY FOR 3.4-canopy-openness.csv***
    """
    identifier = name.split()
    if len(identifier)>=2:
        identifier[1] = identifier[1].upper()
    else:
        identifier = name.split('-')
        identifier[1] = identifier[1].upper()
        if re.search("Opc",identifier[3], re.IGNORECASE):
            identifier[3] = "OPC"
    identifier[2] = re.findall(r'\d+', identifier[2])[0]  #Only keep the numeric part
    
    # Remove later
    temp = identifier[2]
    identifier[2] = identifier[1]
    identifier[1] = temp

    
    name = '-'.join(identifier)
    return name

def canopy_openness(path, timepoint="post1"):
  ''' 
  Extracts canopy openness data from a CSV file, calculates the average openness, and filters by timepoint.
  Default timepoint is "post1", but for more recent data "post 3" should be considered.
  
  '''
  coordinates_path = "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/result_data.gpkg"
  canopy_path = path
  try:
    canopy_df_filtered = pd.read_csv(canopy_path)
    coordinates_gdf = gpd.read_file(coordinates_path)

    # Define the columns to average
    openness_cols = ['canopy.openness.to.river', 'canopy.openness.from.river', 'canopy.openness.right', 'canopy.openness.left']

    # Convert columns to numeric, coercing errors to NaN
    for col in openness_cols:
        if col in canopy_df_filtered.columns:
            canopy_df_filtered[col] = pd.to_numeric(canopy_df_filtered[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    # Calculate the average of the specified canopy openness columns
    # Only attempt to average if all columns are present
    if all(col in canopy_df_filtered.columns for col in openness_cols):
        canopy_df_filtered['average_canopy_openness'] = canopy_df_filtered[openness_cols].mean(axis=1)
    else:
        print("Error: Not all required columns for averaging were found.")


    # Filter by 'timepoint' after calculating the average
    canopy_df_filtered = canopy_df_filtered[canopy_df_filtered['timepoint'].str.contains(timepoint, case=False, na=False)]
    
    # Standardize names in the 'point.label' column
    canopy_df_filtered['point.label'] = canopy_df_filtered['point.label'].apply(standardize_names_for_canopy_openness)

    # Average across time points for a given point.label
    # Maybe consider other methods of averaging if there are multiple time points?
    canopy_df_filtered = canopy_df_filtered.groupby('point.label').agg({'average_canopy_openness': 'mean'}).reset_index()

    # Merge with coordinates_gdf to attach geometry
    canopy_df_filtered = canopy_df_filtered.merge(coordinates_gdf, left_on='point.label', right_on='name', how='left')
    
    # Drop unnecessary columns and keep only relevant ones
    canopy_df_filtered = canopy_df_filtered.drop([column for column in canopy_df_filtered.columns if column not in ['point.label', 'average_canopy_openness','geometry']], axis=1)
    
    # Create a GeoDataFrame and save as a gpkg file
    merged_gdf = gpd.GeoDataFrame(canopy_df_filtered, geometry='geometry')
    merged_gdf.to_file("canopy_openness_result.gpkg", driver="GPKG")

    return merged_gdf

  except FileNotFoundError:
    print(f"Error: The file was not found at {canopy_path}")
  except Exception as e:
    print(f"An error occurred: {e}")