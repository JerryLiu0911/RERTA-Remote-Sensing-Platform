# RERTA Remote Sensing Platform

A Python-based platform for analyzing remote sensing data, focusing on canopy openness and vegetation indices.

## Features

- Load and merge multiple geospatial datasets (CSV, GeoPackage)

- Calculate and analyze vegetation indices (e.g., ExG, GLI)

- Statistical modeling tools:

  ✅ Simple Linear Regression

  ✅ Random Forest Regression

  ✅ Feature Importance Ranking

- Spatial coordinate matching for ground-truth and UAV data

- Visualization of results (scatter plots, regression curves, etc.)

## Requirements

```bash
numpy
pandas
geopandas
scikit-learn
matplotlib
fiona
```

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Load and Prepare Data
```python
import statistical_modelling

# Load data
dataframes = [
    ('canopy_openness', 'path/to/canopy.csv'),
    ('ExG', 'path/to/ExG.gpkg'),
    ('GLI', 'path/to/GLI.gpkg')
]
merged_df = statistical_modelling.load_data(dataframes)
```

### Perform Random Forest Regression
```python
model, mse, rmse, r2, y_pred = statistical_modelling.random_forest_regression(
    df=merged_df,
    target='average_canopy_openness',
    variables=['ExG_mean', 'GLI_mean'],
    display=True
)
```

## Project Structure

- `main.py`: Main execution script
- `statistical_modelling.py`: Statistical analysis functions
- `align_coords.py`: Coordinate alignment utilities
- `coordinate_extraction.py`: Functions for extracting coordinates

## Contributing

[Add contribution guidelines]

## License

[Add license information]

## Contact

[Add contact information]
