import geopandas as gpd
import pandas as pd
import re


def _standardize_side(identifier):
    """Replace single-letter cardinal codes with full words."""
    if identifier[1] == 'E':
        identifier[1] = 'EAST'
    elif identifier[1] == 'W':
        identifier[1] = 'WEST'
    identifier[2] = re.findall(r'\d+', identifier[2])[0]
    return identifier


def standardize_names_for_extract_central_coords(name):
    """
    Standardize a raw GPS waypoint name into treatment-EAST/WEST-transect-BC/OPE/OPC format.
    Specifically for the Palapa vegetation-plot centre points.

    Example: "A-E-12m-OPc" -> "A-EAST-12-OPC"
    """
    identifier = _standardize_side(name.split('-'))
    if re.search("OPc", identifier[3], re.IGNORECASE):
        identifier[3] = 'OPC'
    elif re.search("OPe", identifier[3], re.IGNORECASE):
        identifier[3] = 'OPE'
    elif re.search("buffercore", identifier[3], re.IGNORECASE):
        identifier[3] = 'BC'
    return '-'.join(identifier)


def standardize_names_for_extract_corner_coords(name):
    """Standardize raw GPS waypoint names for vegetation-plot corner points."""
    identifier = _standardize_side(name.split('-'))
    if re.search("OPC", identifier[3], re.IGNORECASE):
        identifier[3] = 'OPC'
    elif re.search("OPE", identifier[3], re.IGNORECASE):
        identifier[3] = 'OPE'
    elif re.search("BC", identifier[3], re.IGNORECASE):
        identifier[3] = 'BC'
    return '-'.join(identifier)


def standardize_names_for_extract_100m_transect_coords(name):
    """Standardize raw GPS waypoint names for 100 m transect lines."""
    identifier = _standardize_side(name[:-5].split('-'))
    if re.search("OPC", identifier[3], re.IGNORECASE):
        identifier[3] = 'OPC'
    elif re.search("OPE", identifier[3], re.IGNORECASE):
        identifier[3] = 'OPE'
    elif re.search("BC", identifier[3], re.IGNORECASE):
        identifier[3] = 'BC'
    elif re.search("river", identifier[3], re.IGNORECASE):
        identifier[3] = 'RV'
    return '-'.join(identifier)


def extract_veg_plots_central_coordinates(source_path, destination_path=None):
    """
    Extract vegetation-plot centre points (OPcore, OPedge, Buffercore) from a GeoPackage,
    standardize the name column, and optionally save the result.

    Args:
        source_path (str): Path to the source GeoPackage.
        destination_path (str | None): If provided, save the result here as GPKG.

    Returns:
        gpd.GeoDataFrame: Filtered and standardized GeoDataFrame.
    """
    try:
        gdf = gpd.read_file(source_path)
    except Exception:
        print("Error reading files")
        return

    result_gdf = gdf[gdf['name'].str.contains("core|edge", case=False, na=False)].copy()
    result_gdf['name'] = result_gdf['name'].apply(standardize_names_for_extract_central_coords)
    result_gdf = result_gdf[['name', 'geometry']]

    if destination_path:
        result_gdf.to_file(destination_path, driver="GPKG")
        print(f"Centre points of vegetation plots saved to: {destination_path}")

    return result_gdf


# Alias used in tests and scripts
extract_central_coords = extract_veg_plots_central_coordinates


def extract_veg_plots_corner_coordinates(source_path, destination_path=None):
    """
    Extract vegetation-plot corner points from a GeoPackage and standardize names.

    Args:
        source_path (str): Path to the source GeoPackage.
        destination_path (str | None): If provided, save the result here as GPKG.

    Returns:
        gpd.GeoDataFrame: Filtered and standardized GeoDataFrame.
    """
    try:
        gdf = gpd.read_file(source_path)
    except Exception:
        print("Error reading files")
        return

    result_gdf = gdf[gdf['name'].str.contains("veg", case=False, na=False)].copy()
    result_gdf['name'] = result_gdf['name'].apply(standardize_names_for_extract_corner_coords)
    result_gdf = result_gdf[['name', 'geometry']]

    if destination_path:
        result_gdf.to_file(destination_path, driver="GPKG")
        print(f"Corner points of vegetation plots saved to: {destination_path}")

    return result_gdf


def extract_ABCD_coords(source_path, destination_path=None):
    """
    Extract ABCD treatment region coordinates from a GeoPackage and standardize names.

    Args:
        source_path (str): Path to the source GeoPackage.
        destination_path (str | None): If provided, save the result here as GPKG.

    Returns:
        gpd.GeoDataFrame: Filtered and standardized GeoDataFrame.
    """
    try:
        gdf = gpd.read_file(source_path)
    except Exception:
        print("Error reading files")
        return

    result_gdf = gdf[gdf['name'].str.contains("veg", case=False, na=False)].copy()
    result_gdf['name'] = result_gdf['name'].apply(standardize_names_for_extract_corner_coords)
    result_gdf = result_gdf[['name', 'geometry']]

    if destination_path:
        result_gdf.to_file(destination_path, driver="GPKG")
        print(f"Coordinates of each treatment region saved to: {destination_path}")

    return result_gdf


def extract_100m_transect_coords(source_path, destination_path=None):
    """
    Extract 100 m transect line coordinates from a GeoPackage and standardize names.

    Args:
        source_path (str): Path to the source GeoPackage.
        destination_path (str | None): If provided, save the result here as GPKG.

    Returns:
        gpd.GeoDataFrame: Filtered and standardized GeoDataFrame.
    """
    try:
        result_gdf = gpd.read_file(source_path)
    except Exception:
        print("Error reading files")
        return

    result_gdf = result_gdf.copy()
    result_gdf['name'] = result_gdf['name'].apply(standardize_names_for_extract_100m_transect_coords)
    result_gdf = result_gdf[['name', 'geometry']]

    if destination_path:
        result_gdf.to_file(destination_path, driver="GPKG")
        print(f"Transect coordinates saved to: {destination_path}")

    return result_gdf
