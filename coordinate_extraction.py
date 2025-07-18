import geopandas as gpd
import fiona
import pandas as pd

def extract_coordinates(gpkg_path):
    '''Extracts coordinates from a GeoPackage file and filters rows based on the 'name' column. 
    classifies the coordinates based on "Buffer core", "OP edge" and "OP core", plotting the results for confirmation.'''


    # Check if the file exists
    if not fiona.supported_drivers.get('GPKG'):
        print("GeoPackage driver is not supported.")
        return None
    try:
        gdf = gpd.read_file(gpkg_path)
    except Exception as e:
        print(f"Error reading GeoPackage: {e}")
        return None

    # Filter rows where the 'name' column contains "core" (case-insensitive)
    core_gdf = gdf[gdf['name'].str.contains("co|edg|buffe", case=False, na=False)]

    # Extract coordinates (geometry.x, geometry.y)
    coordinates = core_gdf.geometry.apply(lambda geom: (geom.x, geom.y))

    # Create a DataFrame with the coordinates
    coordinates_df = pd.DataFrame(coordinates.tolist(), columns=['Longitude', 'Latitude'])

    # Concatenate the original GeoDataFrame with the coordinates DataFrame
    result_gdf = pd.concat([core_gdf.reset_index(drop=True), coordinates_df], axis=1)

    # Display the result
    print(result_gdf.head())

    ax = core_gdf.plot(markersize=5)
    ax.set_title("Spatial Distribution of Coordinates")

extract_coordinates("Data/Rerta koordinate 2018_09_24.gpkg")