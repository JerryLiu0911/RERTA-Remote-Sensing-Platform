import geopandas as gpd
from shapely.geometry import Point

from rerta.coordinate_extraction import (
    standardize_names_for_extract_central_coords,
    extract_central_coords,
)


def test_standardize_central_coords_opc():
    assert standardize_names_for_extract_central_coords("A-E-12m-OPc") == "A-EAST-12-OPC"


def test_standardize_central_coords_ope():
    assert standardize_names_for_extract_central_coords("B-E-150m-OPedge") == "B-EAST-150-OPE"


def test_standardize_central_coords_bc():
    assert standardize_names_for_extract_central_coords("A-W-250m-buffercore") == "A-WEST-250-BC"


def test_extract_central_coords_filters_and_standardizes(tmp_path):
    """Only rows whose name contains 'core' or 'edge' should be kept; names are standardized."""
    data = {
        'name': [
            "A-E-150m-OPcore",     # included — contains "core"
            "A-W-250m-buffercore", # included — contains "core"
            "B-E-150m-OPedge",     # included — contains "edge"
            "A-E-8-foo",           # excluded — no "core" or "edge"
        ],
        'geometry': [Point(1, 2), Point(2, 3), Point(3, 4), Point(4, 5)],
        'extra_col': [1, 2, 3, 4],
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    gdf.to_file(str(tmp_path / "test.gpkg"), driver="GPKG")

    result = extract_central_coords(str(tmp_path / "test.gpkg"))

    assert len(result) == 3
    assert set(result.columns) == {'name', 'geometry'}
    assert set(result['name']) == {"A-EAST-150-OPC", "A-WEST-250-BC", "B-EAST-150-OPE"}


def test_extract_central_coords_no_save(tmp_path):
    """Omitting destination_path should not write any extra files."""
    data = {
        'name': ["A-E-50m-OPcore"],
        'geometry': [Point(1, 2)],
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
    src = str(tmp_path / "src.gpkg")
    gdf.to_file(src, driver="GPKG")

    result = extract_central_coords(src)
    assert result is not None
    assert len(list(tmp_path.glob("*.gpkg"))) == 1  # only the source file
