import geopandas as gpd
import pandas as pd
import re

def standardize_names_for_extract_coords(name):
    """
    Standardizes the 'name' column in the GeoDataFrame by replacing specific patterns. 
    ***SPECIFICALLY FOR Rerta koordinate 2018_09_24.gpkg***
    """
    identifier = name.split('-')
    if identifier[1] == 'E':
      identifier[1] = 'EAST'
    elif identifier[1] == 'W':
      identifier[1] = 'WEST'
    
    identifier[2] = re.findall(r'\d+', identifier[2])[0]  # Only keep the numeric part
    
    if re.search("OPc", identifier[3], re.IGNORECASE):
      identifier[3] = 'OPC'
    elif re.search("OPe", identifier[3], re.IGNORECASE):
      identifier[3] = 'OPE'
    elif re.search("buff", identifier[3], re.IGNORECASE):
      identifier[3] = 'BC'

    name = '-'.join(identifier)
    return name

def extract_coords(path):
  gpkg_path = path
  try:
    gdf = gpd.read_file(gpkg_path)
  except Exception as e :
    print("Error reading files")
    return

  # Filter rows where the 'name' column contains "core" (case-insensitive)
  result_gdf = gdf[gdf['name'].str.contains("co|edg|buffe", case=False, na=False)]
  result_gdf['name'] = result_gdf['name'].apply(standardize_names_for_extract_coords)

  # Display the result
  result_gdf = result_gdf.drop([column for column in result_gdf.columns if column not in ['name', 'geometry']], axis=1)
  #result_gdf.to_file("result_data.gpkg", driver="GPKG")
  return result_gdf