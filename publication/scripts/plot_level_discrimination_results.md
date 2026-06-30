# Plot-level treatment discrimination (D2) — option=vi

Kruskal-Wallis across treatments at the **plot** level (one value per plot). `eta2` = effect size; `p_fdr` = Benjamini-Hochberg adjusted.

Treatments: ['A', 'B', 'C', 'D'].  Significant (FDR<0.05): 20 / 39 metrics; structural/textural among them: 11.

| metric | type | H | p_raw | p_fdr | eta2 |
|---|---|--:|--:|--:|--:|
| range_GLI | spectral | 31.54 | 0.000 | 0.000 | 0.65 |
| homogeneity_ReNDVI | textural | 23.38 | 0.000 | 0.000 | 0.46 |
| contrast_Clre | textural | 23.32 | 0.000 | 0.000 | 0.46 |
| homogeneity_Clre | textural | 23.19 | 0.000 | 0.000 | 0.46 |
| range_Clre | spectral | 22.56 | 0.000 | 0.000 | 0.44 |
| energy_Clre | textural | 22.30 | 0.000 | 0.000 | 0.44 |
| contrast_ReNDVI | textural | 20.10 | 0.000 | 0.001 | 0.39 |
| range_ReNDVI | spectral | 19.74 | 0.000 | 0.001 | 0.38 |
| mean_DEM | structural | 19.16 | 0.000 | 0.001 | 0.37 |
| range_GNDVI | spectral | 17.77 | 0.000 | 0.002 | 0.34 |
| energy_ReNDVI | textural | 16.79 | 0.001 | 0.003 | 0.31 |
| cv_GLI | spectral | 16.74 | 0.001 | 0.003 | 0.31 |
| canopy_openness_DEM | structural | 16.28 | 0.001 | 0.003 | 0.30 |
| correlation_Clre | textural | 16.08 | 0.001 | 0.003 | 0.30 |
| cv_GNDVI | spectral | 14.58 | 0.002 | 0.006 | 0.26 |
| cv_Clre | spectral | 14.14 | 0.003 | 0.007 | 0.25 |
| homogeneity_GNDVI | textural | 13.34 | 0.004 | 0.009 | 0.23 |
| range_NDVI | spectral | 12.60 | 0.006 | 0.012 | 0.22 |
| cv_ReNDVI | spectral | 11.11 | 0.011 | 0.023 | 0.18 |
| correlation_ReNDVI | textural | 9.96 | 0.019 | 0.037 | 0.16 |
| mean_NDVI | spectral | 8.82 | 0.032 | 0.059 | 0.13 |
| cv_NDVI | spectral | 8.73 | 0.033 | 0.059 | 0.13 |
| contrast_GNDVI | textural | 8.42 | 0.038 | 0.065 | 0.12 |
| homogeneity_NDVI | textural | 8.21 | 0.042 | 0.068 | 0.12 |
| energy_GNDVI | textural | 7.96 | 0.047 | 0.073 | 0.11 |
| mean_GNDVI | spectral | 7.84 | 0.049 | 0.074 | 0.11 |
| range_DEM | structural | 6.33 | 0.097 | 0.139 | 0.08 |
| energy_GLI | textural | 6.14 | 0.105 | 0.146 | 0.07 |
| correlation_GLI | textural | 6.03 | 0.110 | 0.148 | 0.07 |
| energy_NDVI | textural | 5.18 | 0.159 | 0.206 | 0.05 |
| contrast_NDVI | textural | 5.11 | 0.164 | 0.206 | 0.05 |
| mean_GLI | spectral | 3.57 | 0.312 | 0.380 | 0.01 |
| homogeneity_GLI | textural | 3.39 | 0.335 | 0.396 | 0.01 |
| mean_Clre | spectral | 3.03 | 0.387 | 0.435 | 0.00 |
| contrast_GLI | textural | 3.01 | 0.391 | 0.435 | 0.00 |
| cv_DEM | structural | 1.71 | 0.635 | 0.635 | 0.00 |
| mean_ReNDVI | spectral | 2.24 | 0.525 | 0.563 | 0.00 |
| correlation_GNDVI | textural | 2.19 | 0.534 | 0.563 | 0.00 |
| correlation_NDVI | textural | 1.80 | 0.615 | 0.631 | 0.00 |
