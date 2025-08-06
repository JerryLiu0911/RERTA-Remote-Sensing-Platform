import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


# ...existing code...


def debug_data_quality(df, target, variables):
    """Debug data quality after GIS processing"""
    
    print("=== POST-GIS DATA QUALITY CHECK ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Target: {target}")
    print(f"Features: {variables}")
    
    # Check for missing/invalid values in features
    print(f"\nFeature value ranges:")
    for var in variables:
        if var in df.columns:
            values = df[var]
            print(f"  {var}: {values.min():.3f} to {values.max():.3f} (count: {values.count()}, NaN: {values.isna().sum()})")
        else:
            print(f"  {var}: MISSING COLUMN!")
    
    # Check target distribution
    print(f"\nTarget ({target}) distribution:")
    print(df[target].describe())
    
    # Check for correlation between features and target
    print(f"\nCorrelations with {target}:")
    correlations = df[variables + [target]].corr()[target].sort_values(ascending=False)
    print(correlations)
    
    # Check for constant/near-constant features
    low_variance_features = []
    for var in variables:
        if var in df.columns:
            if df[var].std() < 0.01:  # Very low variance
                low_variance_features.append(var)
    
    if low_variance_features:
        print(f"\nLow variance features (remove these): {low_variance_features}")
    
    # Check for outliers
    print(f"\nPotential outliers (>3 std from mean):")
    for var in variables:
        if var in df.columns:
            mean_val = df[var].mean()
            std_val = df[var].std()
            outliers = df[(df[var] > mean_val + 3*std_val) | (df[var] < mean_val - 3*std_val)]
            if len(outliers) > 0:
                print(f"  {var}: {len(outliers)} outliers")
    
    return correlations

def random_forest_regression(df, target, variables, display=True, test_size=0.2, random_state=42):
    """Enhanced random forest with better debugging"""
    
    # Add comprehensive debugging
    correlations = debug_data_quality(df, target, variables)
    
    # Filter out features with near-zero correlation
    meaningful_features = []
    for var in variables:
        if var in correlations and abs(correlations[var]) > 0.05:  # At least 5% correlation
            meaningful_features.append(var)
    
    if len(meaningful_features) == 0:
        print("❌ CRITICAL: No features have meaningful correlation with target!")
        print("This suggests:")
        print("  1. Spatial misalignment between field data and remote sensing")
        print("  2. Scale mismatch between measurements")
        print("  3. CHM processing removed all meaningful variation")
        print("  4. Wrong field plot locations")
        return None, None, None, None, None
    
    print(f"\n✅ Using {len(meaningful_features)} meaningful features: {meaningful_features}")
    
    # Prepare data with only meaningful features
    y = np.array(df[target])
    x = np.array(df[meaningful_features])
    
    # Check for data issues
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        print("❌ Found NaN/infinite values in features")
        return None, None, None, None, None
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        print("❌ Found NaN/infinite values in target")
        return None, None, None, None, None
    
    print(f"\nData summary:")
    print(f"  Samples: {x.shape[0]}")
    print(f"  Features: {x.shape[1]}")
    print(f"  Target range: {y.min():.2f} to {y.max():.2f}")
    print(f"  Target std: {y.std():.2f}")
    
    # Check if we have enough data
    if x.shape[0] < 10:
        print("❌ Too few samples for reliable modeling")
        return None, None, None, None, None
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train/test split: {x_train.shape[0]}/{x_test.shape[0]} samples")
    
    # Try a simple baseline first
    print("\n=== TRYING BASELINE MODEL ===")
    baseline_rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
    baseline_scores = cross_val_score(baseline_rf, x_train, y_train, cv=min(5, x_train.shape[0]), scoring='r2')
    print(f"Baseline CV R²: {baseline_scores.mean():.3f} ± {baseline_scores.std():.3f}")
    
    if baseline_scores.mean() < -0.5:
        print("❌ Even baseline model fails - fundamental data problem!")
        return None, None, None, None, None
    
    # Proceed with hyperparameter tuning
    model, best_score = tune_random_forest(x_train, y_train, random_state=random_state, n_iter=10)
    
    # Rest of your existing code...
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_pred_train)
    
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Print enhanced results
    print(f'\n=== FINAL RESULTS ===')
    print(f'Training R²: {train_r2:.3f}')
    print(f'Test R²: {test_r2:.3f}')
    print(f'Features used: {meaningful_features}')
    
    if display and test_r2 > -0.5:
        # Your existing plotting code
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_train, y_pred_train, alpha=0.6)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Training (R² = {train_r2:.3f})')
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred_test, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Test (R² = {test_r2:.3f})')
        
        plt.tight_layout()
        plt.show()
    
    return model, test_mse, test_rmse, test_r2, y_pred_test