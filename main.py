import geopandas as gpd
import fiona
import pandas as pd

print("Hello")
gdb_path = "Data/Rerta koordinate 2018_09_24.gpkg"

# List available layers
layers = fiona.listlayers(gdb_path)
print("Layers in GDB:", layers)

# Read a specific layer (replace 'your_layer_name' with one from the printed list)
layer_name = layers[0]  # or set to the desired layer
gdf = gpd.read_file(gdb_path, layer=layer_name)
print(gdf.head())

