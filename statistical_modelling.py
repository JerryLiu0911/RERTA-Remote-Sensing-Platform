import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

def load_data(dataframes):
  '''
  Merges all df's into one merged_df with suffixes (e.g. _mean_ExG).
  Input should be of the form [('name','path'),('name2','path2')]
  '''
  merged_df = pd.DataFrame()
  for i in range(len(dataframes)):
      name, path = dataframes[i]
      df = gpd.read_file(path)
      df = df.dropna() # Remove rows with NaN values
      df_cleaned = df[df['point.label'].str.contains("OPE|BC", case=False, na=False)] # Remove OPC as the orthomosaic is not well defined at the edges
      df_cleaned = df_cleaned.rename(columns={col: f'{col}_{name}' for col in df_cleaned.columns if col != 'point.label'})
      if i == 0:
          merged_df = df_cleaned
          merged_df = merged_df.rename(columns={f'average_canopy_openness_{name}': 'average_canopy_openness'})
      else:
          merged_df = merged_df.merge(df_cleaned, on='point.label', how='inner')
          if np.allclose(merged_df['average_canopy_openness'], merged_df[f'average_canopy_openness_{name}'], equal_nan=True):
              # If they are the same, drop one and rename the other
              merged_df = merged_df.drop(columns=[f'average_canopy_openness_{name}'])
              print("average_canopy_openness_ExG and average_canopy_openness_GLI are the same. Merged into one column.")
          else:
              print("average_canopy_openness_ExG and average_canopy_openness_GLI are not the same.")

  # Display the merged dataframe
  print(merged_df.head())
  return merged_df


def simple_linear_regression(x, y):
  """
  Performs simple linear regression using the matrix method.

  Args:
    x: A numpy array of independent variable values.
    y: A numpy array of dependent variable values.

  Returns:
    A tuple containing the slope (m) and y-intercept (b) of the regression line.
  """
  n = len(x)
  if n != len(y):
    raise ValueError("Input arrays must have the same length.")

  # Add a column of ones for the intercept term
  X = np.vstack([np.ones(n), x]).T

  # Calculate coefficients using the matrix method: (X^T * X)^(-1) * X^T * y
  # X.T is the transpose of X
  # np.linalg.inv() calculates the inverse of a matrix
  # @ is the matrix multiplication operator
  coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

  # The coefficients are [b, m]
  b, m = coefficients

  return m, b

def multi_linear_regression_display(merged_df, target, variables, display = True):
  '''
  Performs linear regression with 'target' on all variables stated in the 'variables' list. 
  'target' and elements in variables should be names of columns in merged_df.
  Displays results.
  stores the best model in best_model.
  '''
  best_model = []
  
  for variable_name in variables:
    x = np.array(merged_df[variable_name])
    y = np.array(merged_df[target])
    m, b = simple_linear_regression(x, y)
    y_pred = m * x + b
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    if len(best_model)==0:
      best_model = [variable_name, m, b, mse, rmse, r2]
    elif r2 > best_model[5]:
      best_model = [variable_name, m, b, mse, rmse, r2]

    if display:
      plt.figure(figsize=(10, 6))
      plt.scatter(x, y, label='Data')
      plt.plot(x, y_pred, color='red', label=f'Linear Regression: y = {m:.2f}x + {b:.2f}')

      # Add labels and title for ExG plot
      plt.xlabel(variable_name)
      plt.ylabel(target)
      plt.title(f'Linear Regression of {variable_name} vs. {target}')

      # Add text annotations for ExG metrics
      plt.text(0.05, 0.95, f'MSE: {mse:.2f}', transform=plt.gca().transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
      plt.text(0.05, 0.90, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
      plt.text(0.05, 0.85, f'R-squared: {r2:.2f}', transform=plt.gca().transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
      
      plt.legend()
      plt.grid(True)
      plt.show()
  print(f'Best model: {best_model}')

  # Assuming you have your x and y data and the calculated slope (m) and intercept (b)


merged_df = load_data([('GLI','/content/drive/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/Palapa June2019 GLI statistics.gpkg'),
          ('ExG','/content/drive/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/Palapa June2019 ExG statistics.gpkg'),
          ('DEM', '/content/drive/MyDrive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/Data/Palapa June2019 DEM statistics.gpkg')])
# Assuming you have your x and y data and the calculated slope (m) and intercept (b)
# Replace with your actual data and calculated values

multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if ('GLI' in column) & (column!='geometry_GLI')], display=True)