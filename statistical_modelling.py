import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV



def check_geometries(geom1, geom2, tolerance=1e-8):
    """
    Compare two geometries using equals_exact with a small tolerance
    to account for floating point differences
    """
    return all(g1.equals_exact(g2, tolerance) for g1, g2 in zip(geom1, geom2))


def load_data(dataframes):
  '''
  Merges all df's into one merged_df with suffixes (e.g. _mean_ExG).
  Input should be of the form [('name','path'),('name2','path2')]
  '''
  merged_df = pd.DataFrame()
  for i in range(len(dataframes)):
      name, path = dataframes[i]
      try:
        df = gpd.read_file(path)
      except Exception as e:
        print(f"Error reading file {path}: {e}")
        continue
      df = df.dropna() # Remove rows with NaN values
      df_cleaned = df[df['point.label'].str.contains("OPE|BC", case=False, na=False)] # Remove OPC as the orthomosaic is not well defined at the edges
      df_cleaned = df_cleaned.rename(columns={col: f'{col}_{name}' for col in df_cleaned.columns if col != 'point.label'})
      if i == 0:
          merged_df = df_cleaned
          merged_df = merged_df.rename(columns={f'average_canopy_openness_{name}': 'average_canopy_openness'})
          merged_df = merged_df.rename(columns={f'geometry_{name}': 'geometry'})
      else:
          merged_df = merged_df.merge(df_cleaned, on='point.label', how='inner')
          if np.allclose(merged_df['average_canopy_openness'], merged_df[f'average_canopy_openness_{name}'], equal_nan=True) & check_geometries(merged_df['geometry'], merged_df[f'geometry_{name}']):
              # If they are the same, drop one and rename the other
              merged_df = merged_df.drop(columns=[f'average_canopy_openness_{name}'])
              merged_df = merged_df.drop(columns=[f'geometry_{name}'])
              print(f"No discrepancies found in {name}, merged successfully.")
          else:
              print(f"discrepancies found in {name}, NOT MERGED.")
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

def random_forest_regression(df, target, variables, display=True, max_depth = 7, n_estimators=100, test_size=0.2, random_state=42):
    """
    Performs random forest regression using scikit-learn.

    Args:
        df: Pandas DataFrame containing the data.
        target: Name of the target variable (dependent variable).
        variables: List of independent variable names (features).
        display: Whether to display the feature importance plot (default: True).
        n_estimators: Number of trees in the forest (default: 100)
        test_size: Proportion of dataset to include in the test split (default: 0.2)
        random_state: Random state for reproducibility (default: 42)

    Returns:
        tuple: (model, mse, rmse, r2, y_pred)
            - model: Trained RandomForestRegressor
            - mse: Mean squared error
            - rmse: Root mean squared error
            - r2: R-squared score
            - y_pred: Predicted values
    """
    y = np.array(df[target])
    x = np.array(df[variables])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # Create and train the model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth
    )

    model.fit(X_train, y_train)

    # Make predictions
    y_pred_test = model.predict(X_test)
    y_pred = model.predict(X_train)

    # Get feature importances
    importances = model.feature_importances_
    features = df[variables].columns

    # Calculate metrics
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, y_pred)

    # Print metrics
    print(f'Mean Squared Error for training: {mse:.2f}')
    print(f'Root Mean Squared Error for training: {rmse:.2f}')
    print(f'R-squared for training: {r2:.2f}')

    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_test)
    print(f'Mean Squared Error for testing: {mse:.2f}')
    print(f'Root Mean Squared Error for testing: {rmse:.2f}')
    print(f'R-squared for testing: {r2:.2f}')

    if display:
      # Plot fit
      plt.figure()
      plt.scatter(y_test, y_pred_test, color='red')
      plt.xlabel('Actual Canopy Openness')
      plt.ylabel('Predicted Canopy Openness')
      # Add text annotations for metrics
      plt.text(0.05, 0.95, f'MSE: {mse:.2f}', transform=plt.gca().transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
      plt.text(0.05, 0.90, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
      plt.text(0.05, 0.85, f'R-squared: {r2:.2f}', transform=plt.gca().transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
      plt.title('Random Forest Ensemble Predictions')
      plt.legend()
      plt.show()
      # Plot feature importances
      plt.figure(figsize=(10,6))
      bars = plt.barh(features, importances)
      for bar in bars:
          width = bar.get_width()
          plt.text(width, 
                  bar.get_y() + bar.get_height()/2,
                  f'{width:.3f}',
                  ha='left',
                  va='center',
                  fontsize=10)
      plt.xlabel('Importance Score')
      plt.title('Feature Relevance to Canopy Openness')
      plt.tight_layout()
      plt.show()

    return model, mse, rmse, r2, y_pred

def random_forest_ensemble(df, target, variables, n_estimators=100, test_size=0.2, random_state=42):
    """
    Performs random forest regression using an ensemble of models.

    Args:
        df: Pandas DataFrame containing the data.
        target: Name of the target variable (dependent variable).
        variables: List of independent variable names (features).
        n_estimators: Number of trees in the forest (default: 100)
        test_size: Proportion of dataset to include in the test split (default: 0.2)
        random_state: Random state for reproducibility (default: 42)

    Returns:
        y_pred_sum: Sum of predicted values from the ensemble model.
    """
    # Reshape variables to be a 2D array
    # x = variables.reshape(-1, 1)
    # y = target
    y = np.array(df[target])
    x = np.array(df[variables])


    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # Create and train multiple models

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []

    for train_index, val_index in kf.split(x_train):
        x_train_sub, x_val = x_train[train_index], x_train[val_index]
        y_train_sub, y_val = y_train[train_index], y_train[val_index]

        model = RandomForestRegressor(max_depth=5, n_estimators=100)
        model.fit(x_train_sub, y_train_sub)
        y_val_pred = model.predict(x_val)
        r2_score_value = r2_score(y_val, y_val_pred)
        plt.figure()
        plt.scatter(y_val, y_val_pred, color='red')
        plt.xlabel('Actual Canopy Openness')
        plt.ylabel('Predicted Canopy Openness')
        # Add text annotations for metrics
        plt.text(0.05, 0.85, f'R-squared: {r2_score_value:.2f}', transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        plt.title('Random Forest Ensemble Predictions')
        plt.legend()
        plt.show()
        models.append((model, r2_score_value))

    preds = [model.predict(x_test) for model, _ in models]

    val_scores = np.array([r2 for _, r2 in models])
    weights = val_scores / val_scores.sum()

    y_pred = np.average(np.array(preds), axis=0, weights=weights)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    plt.figure()
    plt.scatter(y_test, y_pred, color='red')
    plt.xlabel('Actual Canopy Openness')
    plt.ylabel('Predicted Canopy Openness')
    # Add text annotations for metrics
    plt.text(0.05, 0.95, f'MSE: {mse:.2f}', transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.text(0.05, 0.90, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.text(0.05, 0.85, f'R-squared: {r2:.2f}', transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.title('Random Forest Ensemble Predictions')
    plt.show()

    return models, y_pred, preds

def tune_random_forest(x_train, y_train, random_state=42, n_iter=20):
    # Define parameter grid
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    rf = RandomForestRegressor(random_state=random_state)

    rs = RandomizedSearchCV(rf, param_distributions=param_dist,
                            n_iter=n_iter, cv=3, scoring='r2',
                            random_state=random_state, n_jobs=-1, verbose=1)

    rs.fit(x_train, y_train)

    print(f"Best params: {rs.best_params_}")
    print(f"Best CV R2: {rs.best_score_:.3f}")

    return rs.best_estimator_



def multi_linear_regression_display(df, target, variables, display = False):
  '''
  Performs multiple linear regression and displays the results.
  Outputs the best model based on R-squared value.
    Args:
        df: Pandas DataFrame containing the data.
        target: Name of the target variable (dependent variable).
        variables: List of independent variable names (features).
        display: Whether to display the feature importance plot (default: False).

    Returns:
        best_model: List containing the best model's variable name, slope, intercept, mse, rmse, and r2.
    """
  '''
  best_model = []
  
  for variable_name in variables:
    x = np.array(df[variable_name])
    y = np.array(df[target])
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

      # Add labels and title for the plot
      plt.xlabel(variable_name)
      plt.ylabel(target)
      plt.title(f'Linear Regression of {variable_name} vs. {target}')

      # Add text annotations for metrics
      plt.text(0.05, 0.95, f'MSE: {mse:.2f}', transform=plt.gca().transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
      plt.text(0.05, 0.90, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
      plt.text(0.05, 0.85, f'R-squared: {r2:.2f}', transform=plt.gca().transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
      
      plt.legend()
      plt.grid(True)
      plt.show()
  if not display: 
    x = np.array(df[best_model[0]])
    y_pred = best_model[1] * x + best_model[2]
    mse = best_model[3]
    rmse = best_model[4]
    r2 = best_model[5]
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label='Data')
    plt.plot(x, y_pred, color='red', label=f'Linear Regression: y = {m:.2f}x + {b:.2f}')

    # Add labels and title for the plot
    plt.xlabel(best_model[0])
    plt.ylabel(target)
    plt.title(f'Linear Regression of {best_model[0]} vs. {target}')

    # Add text annotations for metrics
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