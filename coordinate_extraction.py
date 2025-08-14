import geopandas as gpd
import pandas as pd
import re

def standardize_names_for_extract_central_coords(name):
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
    elif re.search("buffercore", identifier[3], re.IGNORECASE):
      identifier[3] = 'BC'

    # temp = identifier[2]
    # identifier[2] = identifier[1]
    # identifier[1] = temp

    name = '-'.join(identifier)
    return name

def standardize_names_for_extract_corner_coords(name):
    """
    Standardizes the 'name' column in the GeoDataFrame by replacing specific patterns. 
    ***SPECIFICALLY FOR Rerta koordinate 2018***
    """
    identifier = name.split('-')
    if identifier[1] == 'E':
      identifier[1] = 'EAST'
    elif identifier[1] == 'W':
      identifier[1] = 'WEST'
    
    identifier[2] = re.findall(r'\d+', identifier[2])[0]  # Only keep the numeric part
    
    if re.search("OPC", identifier[3], re.IGNORECASE):
      edge_number = re.findall(r'\d+', identifier[3])
      identifier[3] = 'OPC'
    elif re.search("OPE", identifier[3], re.IGNORECASE):
      identifier[3] = 'OPE'
    elif re.search("BC", identifier[3], re.IGNORECASE):
      identifier[3] = 'BC'

    # temp = identifier[2]
    # identifier[2] = identifier[1]
    # identifier[1] = temp

    name = '-'.join(identifier)
    return name

def extract_central_coords(source_path, destination_path):
  """
  Extracts the central coordinates of vegetation plots OPcore, OPedge and Buffercore from a GeoPackage file, filters rows based on specific patterns in the 'name' column,
  and standardizes the 'name' column.
  
  Args:
    source_path (str): Path to the source GeoPackage file.
    destination_path (str): Path to save the filtered and standardized GeoDataFrame.

  Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the filtered and standardized data.
  """
  gpkg_path = source_path
  try:
    gdf = gpd.read_file(gpkg_path)
  except Exception as e :
    print("Error reading files")
    return

  # Filter rows where the 'name' column contains "OP" (case-insensitive)
  result_gdf = gdf[gdf['name'].str.contains("core|edge", case=False, na=False)]
  result_gdf['name'] = result_gdf['name'].apply(standardize_names_for_extract_central_coords)

  # Display the result
  result_gdf = result_gdf.drop([column for column in result_gdf.columns if column not in ['name', 'geometry']], axis=1)
  result_gdf.to_file(destination_path, driver="GPKG")
  return result_gdf



def extract_corner_coords(source_path, destination_path):
  """
  Extracts the corner coordinates of the vegetation plots from a GeoPackage file, filters rows based on specific patterns in the 'name' column,
  and standardizes the 'name' column.
  
  Args:
    source_path (str): Path to the source GeoPackage file.
    destination_path (str): Path to save the filtered and standardized GeoDataFrame.

  Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the filtered and standardized data.
  """
  gpkg_path = source_path
  try:
    gdf = gpd.read_file(gpkg_path)
  except Exception as e :
    print("Error reading files")
    return

  # Filter rows where the 'name' column contains "veg" (case-insensitive)
  result_gdf = gdf[gdf['name'].str.contains("veg", case=False, na=False)]
  result_gdf['name'] = result_gdf['name'].apply(standardize_names_for_extract_corner_coords)

  # Display the result
  result_gdf = result_gdf.drop([column for column in result_gdf.columns if column not in ['name', 'geometry']], axis=1)
  result_gdf.to_file(destination_path, driver="GPKG")
  return result_gdf

def extract_ABCD_coords(source_path, destination_path):
    """
    Extracts the corner coordinates of the vegetation plots from a GeoPackage file, filters rows based on specific patterns in the 'name' column,
    and standardizes the 'name' column.
    
    Args:
        source_path (str): Path to the source GeoPackage file.
        destination_path (str): Path to save the filtered and standardized GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the filtered and standardized data.
    """
    gpkg_path = source_path
    try:
        gdf = gpd.read_file(gpkg_path)
    except Exception as e:
        print("Error reading files")
        return

    # Filter rows where the 'name' column contains "veg" (case-insensitive)
    result_gdf = gdf[gdf['name'].str.contains("veg", case=False, na=False)]
    result_gdf['name'] = result_gdf['name'].apply(standardize_names_for_extract_corner_coords)

    # Display the result
    result_gdf = result_gdf.drop([column for column in result_gdf.columns if column not in ['name', 'geometry']], axis=1)
    result_gdf.to_file(destination_path, driver="GPKG")
    return result_gdf
