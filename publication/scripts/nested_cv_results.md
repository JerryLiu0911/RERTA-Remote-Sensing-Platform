# Leakage-free CV validation — tuned grid

_options=['vi', 'bands', 'bands+vi'], estimators=['OLS', 'RF_tuned'], CV=RepeatedKFold(5x2), RF n_iter=8_

`nested_r2` is the honest out-of-sample estimate (transform re-fit per fold). `leaky_r2` fixes the transform on all rows (current main.py). `optimism = leaky - nested`. `baseline_r2` is a mean predictor.

| option | target | est | n | leaky R² | nested R² | base R² | optimism |
|---|---|---|--:|--:|--:|--:|--:|
| vi | average_canopy_openness | OLS | 48 | +0.187 | +0.177 | -0.077 | +0.010 |
| vi | average_canopy_openness | RF_tuned | 48 | +0.048 | +0.040 | -0.077 | +0.008 |
| vi | Circle_change_mm | OLS | 32 | -0.126 | -0.236 | -0.118 | +0.110 |
| vi | Circle_change_mm | RF_tuned | 32 | -0.074 | -0.111 | -0.118 | +0.037 |
| vi | Harvesting_path_change_mm | OLS | 32 | -0.077 | -0.064 | -0.091 | -0.012 |
| vi | Harvesting_path_change_mm | RF_tuned | 32 | -0.262 | -0.208 | -0.091 | -0.054 |
| vi | Windrow_change_mm | OLS | 32 | -0.167 | -0.162 | -0.059 | -0.006 |
| vi | Windrow_change_mm | RF_tuned | 32 | +0.162 | +0.011 | -0.059 | +0.151 |
| vi | average_change_mm | OLS | 32 | -0.061 | -0.053 | -0.119 | -0.008 |
| vi | average_change_mm | RF_tuned | 32 | +0.214 | +0.062 | -0.119 | +0.152 |
| vi | average_seed_removed_proportion | OLS | 48 | -0.082 | -0.099 | -0.031 | +0.018 |
| vi | average_seed_removed_proportion | RF_tuned | 48 | -0.011 | -0.054 | -0.031 | +0.042 |
| bands | average_canopy_openness | OLS | 48 | +0.048 | +0.051 | -0.077 | -0.003 |
| bands | average_canopy_openness | RF_tuned | 48 | -0.090 | +0.001 | -0.077 | -0.091 |
| bands | Circle_change_mm | OLS | 32 | -0.195 | -0.173 | -0.118 | -0.021 |
| bands | Circle_change_mm | RF_tuned | 32 | -0.168 | -0.232 | -0.118 | +0.064 |
| bands | Harvesting_path_change_mm | OLS | 32 | -0.179 | -0.162 | -0.091 | -0.017 |
| bands | Harvesting_path_change_mm | RF_tuned | 32 | -0.102 | -0.064 | -0.091 | -0.038 |
| bands | Windrow_change_mm | OLS | 32 | -0.239 | -0.242 | -0.059 | +0.002 |
| bands | Windrow_change_mm | RF_tuned | 32 | -0.158 | -0.049 | -0.059 | -0.110 |
| bands | average_change_mm | OLS | 32 | -0.250 | -0.230 | -0.119 | -0.019 |
| bands | average_change_mm | RF_tuned | 32 | -0.056 | -0.045 | -0.119 | -0.011 |
| bands | average_seed_removed_proportion | OLS | 48 | -0.223 | -0.221 | -0.031 | -0.002 |
| bands | average_seed_removed_proportion | RF_tuned | 48 | +0.148 | +0.047 | -0.031 | +0.101 |
| bands+vi | average_canopy_openness | OLS | 48 | +0.236 | +0.144 | -0.077 | +0.092 |
| bands+vi | average_canopy_openness | RF_tuned | 48 | +0.040 | +0.045 | -0.077 | -0.005 |
| bands+vi | Circle_change_mm | OLS | 32 | -0.176 | -0.057 | -0.118 | -0.119 |
| bands+vi | Circle_change_mm | RF_tuned | 32 | -0.137 | -0.239 | -0.118 | +0.102 |
| bands+vi | Harvesting_path_change_mm | OLS | 32 | -0.077 | -0.054 | -0.091 | -0.023 |
| bands+vi | Harvesting_path_change_mm | RF_tuned | 32 | -0.262 | -0.059 | -0.091 | -0.203 |
| bands+vi | Windrow_change_mm | OLS | 32 | -0.167 | -0.256 | -0.059 | +0.088 |
| bands+vi | Windrow_change_mm | RF_tuned | 32 | +0.162 | -0.076 | -0.059 | +0.238 |
| bands+vi | average_change_mm | OLS | 32 | -0.061 | -0.094 | -0.119 | +0.033 |
| bands+vi | average_change_mm | RF_tuned | 32 | +0.214 | +0.139 | -0.119 | +0.074 |
| bands+vi | average_seed_removed_proportion | OLS | 48 | -0.074 | -0.108 | -0.031 | +0.035 |
| bands+vi | average_seed_removed_proportion | RF_tuned | 48 | -0.073 | -0.304 | -0.031 | +0.231 |
