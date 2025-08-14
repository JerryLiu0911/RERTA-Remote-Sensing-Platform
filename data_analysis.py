
import geopandas as gpd
import pandas as pd

def gpkg_to_csv(gpkg_path, csv_path, include_geometry=False):
    """
    Convert GeoPackage to CSV file
    
    Args:
        gpkg_path (str): Path to the input GeoPackage file
        csv_path (str): Path for the output CSV file
        include_geometry (bool): Whether to include geometry as WKT text
    """
    try:
        # Read the GeoPackage
        gdf = gpd.read_file(gpkg_path)
        
        if include_geometry:
            # Convert geometry to WKT (Well-Known Text) format
            gdf['geometry_wkt'] = gdf['geometry'].apply(lambda x: x.wkt)
            # Drop the original geometry column
            df = gdf.drop(columns=['geometry'])
        else:
            # Drop geometry column entirely
            df = gdf.drop(columns=['geometry'])
        
        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"Successfully exported to {csv_path}")
        
    except Exception as e:
        print(f"Error converting GPKG to CSV: {e}")

# Usage example:
gpkg_to_csv("Data/Rerta_ABCD.gpkg", "Data/Rerta_ABCD.csv", include_geometry=True)
print("Conversion complete.")
canopy_df_filtered = pd.read_csv("Data/Rerta_ABCD.csv")
print(canopy_df_filtered.head())