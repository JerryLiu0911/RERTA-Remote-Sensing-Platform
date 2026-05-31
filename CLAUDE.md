# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RERTA (Riparian Ecosystem Restoration in Tropical Agriculture) is a Python platform for analyzing ecological relationships using UAV-derived remote sensing data, GPS field measurements, and biodiversity surveys collected at the Palapa site in Borneo.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis pipeline (from project root)
python scripts/run_analysis.py

# Run tests
pytest

# Run a single test
pytest tests/test_coordinate_extraction.py::test_standardize_central_coords_opc
```

## Repository Structure

```
rerta/                      # Python package — all source modules
  coordinate_extraction.py  # GPS waypoint parsing and name standardization
  load_field_data.py        # CSV field data loading and alignment
  gis.py                    # Zonal statistics and spatial analysis
  statistical_modelling.py  # Feature selection, PCA, RF regression

scripts/
  run_analysis.py           # Main pipeline entry point (replaces old main.py)
  data_analysis.py          # Utility: compare/rename file sets

notebooks/
  project_playground.ipynb  # Tutorial and worked examples

data/
  raw/field/                # CSV field measurements (canopy, frogs, erosion, seed removal)
  raw/coordinates/          # Original GPS GeoPackage and GDB files
  spatial/                  # Reference/boundary GeoPackages (plot corners, transect buffers, treatment regions)
  processed/                # Computed zonal-statistics GeoPackages (output of preprocessing)

results/                    # All plot outputs
  vi/                       # Vegetation-index model diagnostics
  bands/                    # Spectral-band (PCA) model diagnostics
  bands_vi/                 # Combined bands+VI model diagnostics
  treatment/                # Treatment-level distribution/boxplot comparisons

tests/
  test_coordinate_extraction.py
```

## Pipeline Architecture

Four sequential stages — the preprocessing block in `scripts/run_analysis.py` is commented out and only needs to run once per new dataset:

### 1. Coordinate Extraction (`rerta/coordinate_extraction.py`)
Reads GPS waypoints from GeoPackage files and standardizes point labels into the canonical format: `treatment-EAST/WEST-transect_number-BC/OPE/OPC` (e.g. `A-EAST-150-OPC`). Public API: `extract_veg_plots_corner_coordinates`, `extract_veg_plots_central_coordinates` (aliased as `extract_central_coords`), `extract_100m_transect_coords`. The per-function `standardize_names_for_*` helpers are now module-level functions and independently testable.

### 2. Field Data Loading (`rerta/load_field_data.py`)
Each `load_*()` function (e.g. `load_canopy_openness`, `load_erosion_sticks`, `load_frogs`, `load_seed_removal`) reads a CSV from `data/raw/field/`, filters by timepoint, standardizes `point.label` into canonical format, and aggregates multiple readings per point. `align_coords()` merges multiple loaded DataFrames against extracted corner coordinates and saves the result as a GeoPackage to `data/processed/`. The `timepoint` parameter accepts an int (year), a string (e.g. `"post2"`), or a `(start_date, end_date)` tuple.

### 3. Geospatial Analysis (`rerta/gis.py`)
`zonal_statistics()` is the core function: it clips a UAV raster (`.tif`) to each buffer polygon, applies a `filtering_logic` function, optionally applies `proxies` (e.g. `GLCM` for texture, `canopy_openness_proxy` for canopy cover), and saves per-buffer statistics as GeoPackage to `data/processed/`. Multi-band rasters (orthomosaics) produce separate `_band1.gpkg … _band7.gpkg` files. Distribution and boxplot figures save under `results/`.

### 4. Statistical Modelling (`rerta/statistical_modelling.py`)
`load_data()` merges multiple zonal-statistics GeoPackages on `point.label`. `smart_feature_selection_pipeline()` performs theory-guided + correlation-based feature selection appropriate for small ecological datasets (~26 samples). Three modelling options run in `scripts/run_analysis.py`: `vi` (vegetation indices), `bands` (raw spectral bands with PCA), and `bands+vi` (combined). Active model: `random_forest_regression()`. Available but commented out: `multi_linear_regression_display()`, `linear_mixed_model()`. Diagnostics save to `results_vi/`, `results_bands/`, `results_bands+vi/`.

### Orchestration (`scripts/run_analysis.py`)
Defines the `paths` dict mapping logical names to file paths. Large raster `.tif` files are stored externally on Google Drive (`G:/My Drive/`) or a local drive (`D:/Jerry/`) — **not in the repo**. The script is guarded with `if __name__ == "__main__"` so it can be safely imported. Column names are sanitized to snake_case (spaces and dots → underscores) before modelling.

## Key Data Conventions

- **Buffer geometries**: `veg_plots_corner_coordinates` (25×25m plots, convex hull) for erosion sticks/canopy/seed removal; `100m_transects` for frogs
- **Point label format**: `treatment-EAST/WEST-transect_number-BC/OPE/OPC` — BC = Buffer Core, OPC = Open Path Core, OPE = Open Path Edge
- **Transformations applied before modelling**: proportion/canopy_openness columns → arcsin-sqrt; abundance/richness → log
- **CRS handling**: all spatial data is reprojected to UTM on load if the source CRS is geographic (EPSG:4326)
- **Zonal stats columns computed per buffer**: `mean`, `range`, `cv`; plus proxy-specific columns like `canopy_openness`, `contrast`, `homogeneity`, `energy`, `correlation` (GLCM texture)
