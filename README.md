# RERTA Remote Sensing Platform

A comprehensive Python-based platform for analyzing ecological relationships using remote sensing data, field measurements, and geospatial analysis. This platform integrates UAV-derived vegetation indices, canopy height models, and biodiversity data to investigate ecological patterns and treatment effects.

## 🌳 Project Overview

### For an example workflow and tutorial, please look at the Project Playground.ipynb jupyter notebook ###

The RERTA (Remote Environmental Research and Technology Applications) platform enables researchers to:

- **Integrate multi-source ecological data** from field measurements, UAV remote sensing, and biodiversity surveys
- **Perform spatial analysis** using zonal statistics and buffer-based calculations
- **Investigate ecological relationships** between forest structure, canopy openness, and biodiversity
- **Compare treatment effects** across different forest management interventions
- **Generate publication-ready visualizations** and statistical analyses

## 🗂️ File Structure and Workflow

### **Core Analysis Modules**

```
├── main.py                     # Main execution script and workflow orchestration
├── project_playground.ipynb    # Example and tutorial for implementation
├── statistical_modelling.py    # Statistical analysis and machine learning tools
├── gis.py                      # Geospatial analysis and zonal statistics
├── coordinate_extraction.py    # GPS coordinate processing and standardization
├── align_coords.py             # Data alignment and merging utilities
└── README.md                   # This comprehensive guide
```

### **Data Processing Pipeline**

#### **1. Coordinate Extraction and Standardization** (`coordinate_extraction.py`)
- **Purpose**: Extract and standardize GPS coordinates from field data
- **Functions**:
  - `extract_central_coords()`: Extract vegetation plot centers
  - `extract_corner_coords()`: Extract plot boundary coordinates
  - `extract_transect_coords()`: Extract transect line coordinates
  - `standardize_names_*()`: Standardize naming conventions across datasets

#### **2. Data Alignment and Integration** (`align_coords.py`)
- **Purpose**: Merge field measurements with spatial coordinates
- **Key Functions**:
  - `canopy_openness()`: Process canopy openness field measurements
  - `frogs()`: Process frog abundance and biodiversity data
  - `standardize_names_*()`: Ensure consistent naming across datasets

#### **3. Geospatial Analysis** (`gis.py`)
- **Purpose**: Perform spatial analysis on raster data using field plot locations
- **Core Functions**:
  - `zonal_statistics()`: Calculate statistics within buffer zones
  - `create_buffer()`: Generate spatial buffers around sampling points
  - `clip_below_zero()` / `clip_and_remove_outliers()`: Data filtering
  - `create_distribution_plots_from_data()`: Generate distribution visualizations
  - `create_boxplot_from_data()`: Create treatment comparison plots

#### **4. Statistical Modeling** (`statistical_modelling.py`)
- **Purpose**: Analyze relationships between variables using various statistical methods
- **Available Methods**:
  - ✅ **Simple Linear Regression**: `simple_linear_regression()`
  - ✅ **Multiple Linear Regression**: `multi_linear_regression_display()`
  - ✅ **Random Forest Regression**: `random_forest_regression()`
  - ✅ **Hyperparameter Tuning**: `tune_random_forest()`
  - ✅ **Model Ensemble**: `random_forest_ensemble()`
  - ✅ **Feature Importance Analysis**: Built into Random Forest functions

#### **5. Main Workflow Orchestration** (`main.py`)
- **Purpose**: Coordinate the entire analysis pipeline
- **Workflow Sections**:
  1. **Data Preparation**: Coordinate extraction and alignment
  2. **Spatial Analysis**: Zonal statistics calculation
  3. **Statistical Modeling**: Relationship analysis
  4. **Visualization**: Plot generation and result interpretation

## 🔄 Complete Analysis Workflow

### **Phase 1: Data Preparation**
```python
# Extract coordinates from GPS data
coordinate_extraction.extract_corner_coords(source_path, destination_path)

# Align field measurements with coordinates
align_coords.canopy_openness(canopy_path, coordinates_path, output_path)
align_coords.frogs(frogs_path, coordinates_path, output_path)
```

### **Phase 2: Spatial Analysis**
```python
# Calculate zonal statistics for remote sensing data
zonal_gdf, figures = gis.zonal_statistics(
    gpkg_path="field_data.gpkg",
    raster_path="UAV_canopy_height_model.tif",
    output_zonal_gpkg="spatial_statistics.gpkg",
    filtering_logic=gis.clip_and_remove_outliers,
    buffer_geom_path="treatment_regions.gpkg",
    show_plots=True
)
```

### **Phase 3: Data Integration**
```python
# Merge multiple datasets
merged_df = statistical_modelling.load_data([
    ('CHM', 'canopy_height_statistics.gpkg'),
    ('ExG', 'vegetation_index_statistics.gpkg'),
    ('Biodiversity', 'frog_abundance_data.gpkg')
], filter="OPE|BC")  # Filter for specific treatment types
```

### **Phase 4: Statistical Analysis**
```python
# Analyze relationships between variables
features = ['mean_CHM', 'std_CHM', 'canopy_openness', 'mean_ExG']

# Simple correlation analysis
statistical_modelling.multi_linear_regression_display(
    merged_df, 'Frog.abundance', features, display=True)

# Advanced machine learning (for larger datasets)
model, mse, rmse, r2, predictions = statistical_modelling.random_forest_regression(
    merged_df, 'Frog.abundance', features, display=True)
```

## 📊 Key Analysis Capabilities

### **Ecological Relationship Investigation**

1. **Forest Structure → Biodiversity**
   ```python
   # Analyze how canopy height affects frog abundance
   plot_relations(merged_df, 'Frog.abundance', 'mean_CHM')
   ```

2. **Treatment Effects Comparison**
   ```python
   # Compare different forest management treatments
   region_data = gis.get_region_data(data_path, 'Frog.abundance')
   gis.create_boxplot_from_data(region_data, 'Frog Abundance')
   ```

3. **Multi-scale Analysis**
   ```python
   # Analyze at different spatial scales using various buffer sizes
   # Buffer creation with different distances
   buffered = gis.create_buffer(points, output_path, buffer_distance=25.0)
   ```

### **Supported Data Types**

- **Remote Sensing Data**: CHM (Canopy Height Model), ExG (Excess Green Index), GLI (Green Leaf Index)
- **Field Measurements**: Canopy openness, forest structure metrics
- **Biodiversity Data**: Frog abundance, species richness
- **Spatial Data**: GPS coordinates, treatment boundaries, transect lines

### **Statistical Methods Available**

| Method | Use Case | Dataset Size | Interpretability |
|--------|----------|--------------|------------------|
| **Pearson Correlation** | Linear relationships | Any | High |
| **Spearman Correlation** | Monotonic relationships | Any | High |
| **Linear Regression** | Simple relationships | Small-Medium | High |
| **Random Forest** | Complex relationships | Large (100+ samples) | Medium |
| **Ensemble Methods** | Robust predictions | Large | Low |

## 🚀 Getting Started

### **Requirements**

```bash
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
geopandas>=0.9.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Geospatial analysis
rasterio>=1.2.0
rasterstats>=0.15.0
fiona>=1.8.0

# Statistical analysis
scipy>=1.7.0
statsmodels>=0.12.0
```

### **Installation**

1. Clone the repository:
```bash
git clone [your-repo-url]
cd RERTA-Remote-Sensing-Platform
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your data paths in `main.py`:
```python
paths = {
    'canopy_openness': "Data/canopy_measurements.csv",
    'CHM_tif': "path/to/canopy_height_model.tif",
    'coordinates': "Data/GPS_coordinates.gpkg",
    # ... other paths
}
```

### **Quick Start Example**

```python
import main

# Define your data paths
paths = {
    'canopy_openness': "Data/3.4-canopy.openness.csv",
    'frogs': "Data/frog_abundance.gpkg", 
    'CHM': "Data/canopy_height_statistics.gpkg",
    'treatment_buffers': "Data/TreatmentRegions.gpkg"
}

# Run the complete analysis pipeline
main.main(paths)
```

## 📈 Output and Results

### **Generated Outputs**

1. **Spatial Statistics Files** (`.gpkg` format)
   - Zonal statistics for each treatment area
   - Merged datasets with spatial coordinates
   - Buffer geometries for analysis

2. **Visualization Files** (`.png` format)
   - Distribution plots by treatment
   - Box plots comparing treatments
   - Scatter plots showing relationships
   - Feature importance plots

3. **Statistical Results** (Console output)
   - Correlation coefficients
   - Regression model performance
   - Feature importance rankings
   - Cross-validation scores

## 🔧 Advanced Features

### **Custom Filtering Functions**
```python
# Create custom data filters
def custom_filter(data):
    return data[(data >= threshold_min) & (data <= threshold_max)]

# Apply in zonal statistics
gis.zonal_statistics(..., filtering_logic=custom_filter)
```

### **Multiple Correlation Methods**
```python
# Compare different correlation methods
correlations = {
    'pearson': df['var1'].corr(df['var2'], method='pearson'),
    'spearman': df['var1'].corr(df['var2'], method='spearman'),
    'kendall': df['var1'].corr(df['var2'], method='kendall')
}
```

### **Ensemble Modeling**
```python
# Use ensemble methods for robust predictions
models, predictions, individual_preds = statistical_modelling.random_forest_ensemble(
    merged_df, target_variable, feature_list)
```

## 🤝 Contributing

### **Adding New Analysis Methods**
1. Add new functions to `statistical_modelling.py`
2. Update `main.py` to include new analysis in workflow
3. Add documentation and examples
4. Test with sample datasets

### **Adding New Data Types**
1. Create processing functions in `align_coords.py`
2. Add coordinate extraction methods if needed
3. Update `load_data()` function to handle new formats
4. Test integration with existing pipeline

## 📝 Citation

If you use this platform in your research, please cite:

```
[Your Citation Information]
RERTA Remote Sensing Platform for Ecological Analysis
[DOI/URL if available]
```

## 📞 Contact and Support

- **Primary Contact**: [Your contact information]
- **Issues**: Please report bugs and feature requests via GitHub issues
- **Documentation**: See individual function docstrings for detailed parameter descriptions

## 🔍 Troubleshooting

### **Common Issues**

1. **CRS Mismatch**: Ensure all spatial data use consistent coordinate systems
2. **Small Dataset Size**: Use linear regression instead of Random Forest for < 50 samples
3. **Missing Data**: Check data alignment and coordinate standardization
4. **Memory Issues**: Process large rasters in chunks or reduce buffer sizes

### **Performance Optimization**

- Use `filtering_logic` to reduce data volume before analysis
- Process subsets of data for initial exploration
- Use appropriate buffer sizes for your research questions
- Consider data type optimization for large datasets

---

**Version**: 2.0  
**Last Updated**: [Current Date]  
**Platform**: Windows/Linux/MacOS  
**Python Version**: 3.8+
