import pytest
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import os

def test_hello():
    assert 1 + 1 == 2

from coordinate_extraction import standardize_names_for_extract_coords, extract_coords

def test_standardize_names_for_extract_coords_basic():
    name = "A-E-12m-OPc"
    result = standardize_names_for_extract_coords(name)
    assert result == "A-EAST-12-OPC"

def test_extract_coords_filters_and_standardizes(tmp_path):
    # Create a mock GeoDataFrame
    data = {
        'name': [
            "A-E-150m-OPco",    # should be included
            "A-W-250m-Buffe",  # should be included
            "B-E-150M-OPedg",     # should be included
            "A-E-8-foo",     # should NOT be included
        ],
        'geometry': [Point(1,2), Point(2,3), Point(3,4), Point(4,5)],
        'other': [1,2,3,4]
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    test_gpkg = tmp_path / "test.gpkg"
    gdf.to_file(test_gpkg, driver="GPKG")

    result_gdf = extract_coords(str(test_gpkg))
    # Only 3 rows should remain
    assert len(result_gdf) == 3
    # Only 'name' and 'geometry' columns should remain
    assert set(result_gdf.columns) == {'name', 'geometry'}
    # Check standardized names
    expected_names = {"A-EAST-150-OPC", "A-WEST-250-BC", "B-EAST-150-OPE"}
    assert set(result_gdf['name']) == expected_names