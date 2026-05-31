import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
# from pymer4.models import glmer
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, LeaveOneOut
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import shap



def check_geometries(geom1, geom2, tolerance=1e-8):
    """
    Compare two geometries using equals_exact with a small tolerance
    to account for floating point differences
    """

    is_exact = [g1.equals_exact(g2, tolerance) for g1, g2 in zip(geom1, geom2)]

    for i, (g1, g2) in enumerate(zip(geom1, geom2)):
        if not g1.equals_exact(g2, tolerance):
            print(f"Geometry mismatch at index {i}:")
            print(f"  Geometry 1: {g1}")
            print(f"  Geometry 2: {g2}")

    return all(g1.equals_exact(g2, tolerance) for g1, g2 in zip(geom1, geom2))

def smart_feature_selection_pipeline(merged_df, target, all_possible_features, display = False):
    """
    Multi-stage feature selection appropriate for small datasets
    """
    print("=== SMART FEATURE SELECTION FOR SMALL DATASETS ===")
    
    n_samples = len(merged_df)
    print(f"Dataset size: {n_samples} samples")
    
    # STAGE 1: Theory-based pre-filtering
    print(f"\n STAGE 1: THEORY-BASED PRE-FILTERING")
    
    # Based on literature, these should be most relevant:
    theory_based_candidates = {
        'height_metrics': [col for col in all_possible_features if 'DEM' in col and any(stat in col for stat in ['mean', 'median', 'max', 'canopy_openness'])],
        'variability_metrics': [col for col in all_possible_features if 'DEM' in col and any(stat in col for stat in ['std', 'range', 'cv'])],
        'greenness_metrics': [col for col in all_possible_features if any(index in col for index in ['Clre', 'ReNDVI', 'GLI', 'NDVI', 'GNDVI']) and any(stat in col for stat in ['mean', 'std', 'range', 'cv'])],
        'spectral_metrics': [col for col in all_possible_features if 'band' in col and any(stat in col for stat in ['mean', 'std', 'range', 'cv'])],
        'texture_metrics': [col for col in all_possible_features if any(tex in col for tex in ['contrast', 'entropy', 'homogeneity', 'dissimilarity', 'ASM', 'energy', 'correlation'])]
    }
    # Select BEST representative from each category
    stage1_features = []
    for category, candidates in theory_based_candidates.items():
        if candidates:
            num_candidates = max(3, int(round(len(candidates)/4)))
            #print(f"  {category}: {len(candidates)} candidates → selecting best {num_candidates}")

            # Calculate target correlations for candidates in this category
            category_corrs = []
            for feat in candidates:
                if feat in merged_df.columns:
                    corr = abs(merged_df[target].corr(merged_df[feat], method='spearman'))
                    category_corrs.append((feat, corr))
            
            # Take top 1-2 from each category
            category_corrs.sort(key=lambda x: x[1], reverse=True)
            selected_from_category = [feat for feat, _ in category_corrs[:num_candidates]]
            stage1_features.extend(selected_from_category)

            # for feat, corr in category_corrs[:num_candidates]:
            #     print(f"     {feat}: {corr:.3f}")
    
   # print(f"  Stage 1 result: {len(stage1_features)} features")
    
    # STAGE 2: VIF-based refinement
    # print(f"\n STAGE 2: VIF-BASED MULTICOLLINEARITY FILTERING")
    # Remove features with high VIF (> 5 is a common threshold)
    if len(stage1_features) > 1:
        X = merged_df[stage1_features].dropna()
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        # print(vif_data)
        # Iteratively remove the feature with the highest VIF above threshold
        features_vif = stage1_features.copy()
        while True:
            X = merged_df[features_vif].dropna()
            vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            max_vif = max(vifs)
            if max_vif > 5 and len(features_vif) > 1:
                remove_idx = vifs.index(max_vif)
                # print(f"  Removing {features_vif[remove_idx]} (VIF={max_vif:.2f})")
                features_vif.pop(remove_idx)
            else:
                break
        stage2_features = features_vif
    else:
        stage2_features = stage1_features
    # print(f"  Stage 2 result: {len(stage2_features)} features")

    if display:
        sns.heatmap(merged_df[[column for column in merged_df.columns if column in stage2_features]].corr(method='spearman'), annot=True, fmt='.2f', cmap='coolwarm')
        plt.show()
    # STAGE 3: Sample size validation
    # print(f"\n STAGE 3: SAMPLE SIZE VALIDATION")
    
    ratio = n_samples / len(stage2_features) if stage2_features else 0
    # print(f"  Sample-to-feature ratio: {ratio:.1f}:1")
    
    if ratio < 8:
        print(f"   Still too many features for dataset size!")
        print(f"  Further reducing to top {min(3, n_samples//5)} features...")
        
        # Final ranking by target correlation
        final_rankings = []
        for feat in stage2_features:
            corr = abs(merged_df[target].corr(merged_df[feat]))
            final_rankings.append((feat, corr))
        
        final_rankings.sort(key=lambda x: x[1], reverse=True)
        max_features = min(3, n_samples//5, len(final_rankings))
        stage3_features = [feat for feat, _ in final_rankings[:max_features]]
        
        # print(f"  Final selection:")
        # for feat, corr in final_rankings[:max_features]:
        #     print(f"     {feat}: {corr:.3f}")
    else:
        stage3_features = stage2_features
    #     print(f"   Feature count appropriate for dataset size")
    
    # print(f"\n FINAL RESULT: {len(stage3_features)} high-quality features")
    # print(f"   Features: {stage3_features}")
    # print(f"   Final ratio: {n_samples/len(stage3_features):.1f}:1")
    
    return stage3_features

def load_data(dataframes, dependent_variables, 
filter = None,):
    '''
    Merges all df's into one merged_df with suffixes (e.g. _mean_ExG).
        
    
    Args:
        dataframes: A list of tuples specifying the dataframes you wish to merge with paths. Input should be of the form [('name','path'),('name2','path2')]
        filter: A string to filter the point.label column using regex, selects only those column names included in the expression. (e.g. "OPE|BC"). If None, no filtering is done.

    Returns:
        merged_df: A pandas DataFrame containing the merged data.
    '''

    print(f"\n \n Loading and merging data {[name for name, path in dataframes]}...")
    merged_df = pd.DataFrame()
    for i in range(len(dataframes)):
        name, path = dataframes[i]
        try:
            df = gpd.read_file(path)
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            continue


        # df = df.dropna() # Remove rows with NaN values

        # print(f"Loaded data from {name}")
        # print(df.head())

        if filter != None:
            df = df[df['point.label'].str.contains(filter, case=False, na=False)] # Choice to remove erroneous labels
        df = df.rename(columns={col: f'{col}_{name}' for col in df.columns if col != 'point.label'})

        if i == 0:
            merged_df = df
            merged_df = merged_df.rename(columns={f'geometry_{name}': 'geometry'})
            merged_df = merged_df.rename(columns={f'treatment_{name}': 'treatment'})

            for column in dependent_variables:
                merged_df = merged_df.rename(columns={f'{column}_{name}': column})
                if column not in merged_df.columns:
                    print(f"Warning !!!!! : {column} not found in {name} columns.")
            print(f"Loaded data from {name}")

        else:
            #if np.allclose(merged_df['average_canopy_openness'], df[f'average_canopy_openness_{name}'], equal_nan=True) & check_geometries(merged_df['geometry'], df[f'geometry_{name}']):
            if check_geometries(merged_df['geometry'], df[f'geometry_{name}']):
                # If they are the same, drop one and rename the other
                for column in dependent_variables:
                    if f'{column}_{name}' in df.columns:
                        df = df.drop(columns=[f'{column}_{name}'], axis=1)
                        # print(f"dropping columns {column}")
                        # print("remaining columns : ", df.columns)
                    else:
                        print(f"Warning !!!!! : {column} not found in merged_df columns.")
                df = df.drop(columns=[f'geometry_{name}', f'treatment_{name}'], axis=1)  # Drop geometry and treatment columns from df
                merged_df = merged_df.merge(df, on='point.label', how='inner')
                print(f"No discrepancies found in {name}, merged successfully.")
            else:
                print(f" discrepancies found in {name}, NOT MERGED !!! ")
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
  # X = np.vstack([np.ones(n), x]).T


  # Calculate coefficients using the matrix method: (X^T * X)^(-1) * X^T * y
  # X.T is the transpose of X
  # np.linalg.inv() calculates the inverse of a matrix
  # @ is the matrix multiplication operator
  # coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

  # The coefficients are [b, m]
  # b, m = coefficients

  results = sm.OLS(y, x).fit()  # Fit the model

  return results

def generalised_linear_model(y, x, family):
    """
    Fits a generalized linear model (GLM) to the data.

    Args:
        y: The dependent variable (response).
        x: The independent variable(s) (predictors).
        family: The GLM family to use (default: Gaussian).

    Returns:
        The fitted GLM results.
    """
    n = len(x)
    if n != len(y):
        raise ValueError("Input arrays must have the same length.")
    
    model = sm.GLM(y, x, family=family)
    results = model.fit()
    return results

def linear_mixed_model(df, target, features, display=False, option=''):
    """
    Fits a Generalized Linear Mixed Model (GLMM) with 'treatment' as a random effect.
    Displays AIC and a predicted vs actual plot.
    """

    df = df[[target]+features+['treatment']].dropna()
    features_str = ' + '.join(features)
    formula = f"{target} ~ {features_str}"

    model = smf.mixedlm(formula, df, groups=df['treatment'])
    result = model.fit(method='bfgs', reml=False)
    print(result.summary())
    print(f"AIC: {result.aic:.2f}")
    print(f"BIC: {result.bic:.2f}")
    y_true = df[target]
    y_pred = result.fittedvalues
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    pseudo_r2 = 1 - ss_res / ss_tot
    print(f"Pseudo R² (fixed effects): {pseudo_r2:.3f}")


    # Predicted vs Actual plot
    if display:
        y_true = df[target]
        y_pred = result.fittedvalues
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        treatments = df['treatment'].values
        unique_treatments = np.unique(treatments)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_treatments)))

        for treatment, color in zip(unique_treatments, colors):
            mask = treatments == treatment
            axes[0].scatter(y_pred[mask], y_true[mask], alpha=0.7, c=[color], label=str(treatment))
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal fit')
        axes[0].set_xlabel(f'Predicted {target}')
        axes[0].set_ylabel(f'Actual {target}')
        axes[0].set_title(f'LMM: Actual vs Predicted {target}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)


        # Residuals histogram
        residuals = result.resid
        axes[1].hist(residuals, bins=30, alpha=0.7)
        axes[1].set_title("LMM Residuals")
        axes[1].set_xlabel("Residual")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'Results_{option}/lmm_diagnostics_{target}.png', dpi=300)
        # plt.show()

    return result
    # except Exception as e:
    #     print(f"GLMM fitting failed: {e}")
    #     return None

def log_transform(data):
    """
    Applies a log transformation to the input data.
    """
    min_val = data.min()
    if min_val <= 0:
        print(f"Using log1p transformation (has non-positive values)")
        data = np.log1p(data)
    else:
        print(f"Using natural log transformation")
        data = np.log(data)
    return data

def arcsinc_sqrt_transform(data):
    """
    Applies an arcsine square root transformation to the input data.
    """
    data = np.clip(data, 0, None)  # Clip negative values
    return np.arcsin(np.sqrt(data))

def multi_linear_regression_display(df, targets, features, display = False, option = ''):
    '''
    Performs multiple linear regression and displays the results.
    Outputs the best model based on R-squared value.
        Args:
            df: Pandas DataFrame containing the data.
            targets (list of str or str): Name(s) of the target variable(s) (dependent variable(s)).
            features: List of independent variable names (features).
            display: Whether to display the feature importance plot (default: False).

        Returns:
            best_model: List containing the best model's variable name, slope, intercept, mse, rmse, and r2.
        '''

    best_model = None

    if isinstance(targets, str):
        targets = [targets]

    for target in targets:
        target_df = df[[target]+features].dropna()
        print(target)
        y = np.array(target_df[target])
        print(y.dtype)
        x = np.array(target_df[features])
        print(x.dtype)

        print(f"=== ENHANCED REGRESSION ANALYSIS: {target} ===")
        
        print(f"\nTARGET VARIABLE DIAGNOSTICS:")
        print(f"Mean: {y.mean():.3f}")
        print(f"Median: {np.median(y):.3f}")
        print(f"Std: {y.std():.3f}")
        print(f"Skewness: {stats.skew(y):.3f}")
        print(f"Min: {y.min()}, Max: {y.max()}")
        print(f"Zeros: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
        
        # Recommend analysis approach
        is_count = (y == y.astype(int)).all() and (y >= 0).all()
        is_highly_skewed = abs(stats.skew(y)) > 1.5
        has_many_zeros = (y == 0).mean() > 0.3

        significance_level = 0.05

        print(f"\nRECOMMENDATIONS:")
        if is_count:
            if has_many_zeros:
                print(" Use Zero-Inflated Poisson or Negative Binomial GLM")
            else:
                print(" Use Poisson or Negative Binomial GLM")
        
        if is_highly_skewed:
            print(" Consider log transformation or GLM with appropriate family")
            if (y == 0).mean() > 0.3:
                print("  - Zero-Inflated or Hurdle model may be appropriate")

        try:
            if len(y) <= 5000:
                stat, p = stats.shapiro(y)
                test_name = 'shapiro'
            else:
                stat, p = stats.normaltest(y)
                test_name = 'normaltest'
        except Exception as e:
            print(f"Error performing normality test: {e}")
            stat, p, test_name = None, None, 'error'

        print({'test': test_name, 'stat': float(stat), 'pvalue': float(p), 'normal': (p > significance_level)})

        print(f"\nStandardizing features...")
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        x = sm.add_constant(x)  # Add constant term for intercept
        results = simple_linear_regression(x, y)
        b, coeffs = results.params[0], results.params[1:]
        print(results.params)
        y_pred = results.fittedvalues
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        print(f"Results for {target}:")
        print(results.summary())

        if len(results.resid) <= 5000:
            stat, p = stats.shapiro(np.array(results.resid))
            test_name = 'shapiro'
        else:
            stat, p = stats.normaltest(np.array(results.resid))
            test_name = 'normaltest'

        bp_stat, bp_pvalue = het_breuschpagan(results.resid, results.model.exog)[:2]

        print({'test': test_name, 'stat': float(stat), 'pvalue': float(p), 'normal': (p > significance_level), 'bp_stat': float(bp_stat), 'bp_pvalue': float(bp_pvalue), 'bp_homogeneous': (bp_pvalue > significance_level)})

        if best_model is None:
            best_model = [target, y, x, results]
        elif results.rsquared > best_model[3].rsquared:
            best_model = [target, y, x, results]

        if display:

            fig, axes = plt.subplots(1, 3, figsize=(20, 6))

            # Bar plot of coefficients (predictor importance)
            best_coeffs = best_model[3].params[1:]
            axes[0].barh(features, best_coeffs, color='skyblue')
            axes[0].set_xlabel('Standardized Coefficient')
            axes[0].set_title('Predictor Importance')
            for i, v in enumerate(best_coeffs):
                axes[0].text(v, i, f'{v:.3f}', va='center', ha='left', fontsize=10)
            axes[0].grid(True, alpha=0.3)

            # Actual vs Predicted plot
            axes[1].scatter(y_pred, y, alpha=0.7)
            axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal fit')
            axes[1].set_xlabel(f'Predicted {target}')
            axes[1].set_ylabel(f'Actual {target}')
            axes[1].set_title(f'Actual vs Predicted {target}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            # Annotate metrics
            axes[1].text(0.05, 0.95, f'MSE: {mse}\nRMSE: {rmse}\nR²: {r2:.2f}',
                        transform=axes[1].transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
            
            # Residuals vs Fitted plot
            residuals = y - y_pred
            axes[2].scatter(y_pred, residuals, alpha=0.7)
            axes[2].axhline(0, color='red', linestyle='--')
            axes[2].set_xlabel('Fitted Values')
            axes[2].set_ylabel('Residuals')
            axes[2].set_title('Residuals vs Fitted Values')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'Results_{option}/regression_diagnostics_{target}.png', dpi=300)
            # plt.show()

    if not display: 
        y = best_model[1]
        y_pred = best_model[3].predict(best_model[2])
        target = best_model[0]
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        coeffs = best_model[3].params[1:]

        print(f"Results for {target}:")
        print(best_model[3].summary())
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot of coefficients (predictor importance)
        axes[0].barh(features, coeffs, color='skyblue')
        axes[0].set_xlabel('Standardized Coefficient')
        axes[0].set_title('Predictor Importance')
        for i, v in enumerate(coeffs):
            axes[0].text(v, i, f'{v:.3f}', va='center', ha='left', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Actual vs Predicted plot
        axes[1].scatter(y_pred, y, alpha=0.7)
        axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal fit')
        axes[1].set_xlabel(f'Predicted {target}')
        axes[1].set_ylabel(f'Actual {target}')
        axes[1].set_title(f'Actual vs Predicted {target}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        # Annotate metrics
        axes[1].text(0.05, 0.95, f'MSE: {mse}\nRMSE: {rmse}\nR²: {r2:.2f}',
                    transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(f'Results_{option}/regression_diagnostics_{target}.png', dpi=300)
        # plt.show()


    print(f"Best model is for target {best_model[0]} with R² = {best_model[3].rsquared:.3f}")

def random_forest_regression(df, target, features, display=True, test_size=0.2, random_state=42, option = ''):
    """
    Performs random forest regression using scikit-learn.

    Args:
        df: Pandas DataFrame containing the data.
        target: Name of the target variable (dependent variable).
        features: List of independent variable names (features).
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
    # Split by treatment
    df = df[[target]+features+['treatment']].dropna()
    treatment_data = {treatment: data for treatment, data in df.groupby('treatment')}
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for treatment, data in treatment_data.items():
        y = np.array(data[target])
        x = np.array(data[features])
        # Split the data
        x_train_treatment, x_test_treatment, y_train_treatment, y_test_treatment = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        print(f"number of training samples added from treatment {treatment}: {len(x_train_treatment)}")
        print(f"number of training labels added from treatment {treatment}: {len(y_train_treatment)}")
        print(f"number of test samples added from treatment {treatment}: {len(x_test_treatment)}")
        print(f"number of test labels added from treatment {treatment}: {len(y_test_treatment)}")
        x_train.extend(x_train_treatment)
        x_test.extend(x_test_treatment)
        y_train.extend(y_train_treatment)
        y_test.extend(y_test_treatment)

    # Checking splitting is evenly distributed between treatments:
    print(f"Total training samples: {len(x_train)}")
    print(f"Total training labels: {len(y_train)}")
    print(f"Total test samples: {len(x_test)}")
    print(f"Total test labels: {len(y_test)}")

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # Create and train the model
    model, best_score =  tune_random_forest(x_train, y_train, random_state=random_state)
    # model = RandomForestRegressor(n_estimators=1000,random_state=random_state, )
    # model.fit(x_train, y_train)

    print(f"Best cross-validated R^2 score during tuning: {best_score:.3f}")

    # Make predictions
    y_pred_test = model.predict(x_test)
    y_pred = model.predict(x_train)

    # Get feature importances
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_train)
    shap_importance = np.mean(np.abs(shap_values), axis=0)
    importances = model.feature_importances_
    # features = df[features].columns

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
      fig, axes = plt.subplots(1, 3, figsize=(20, 6))
      axes[0].scatter(y_test, y_pred_test, color='red')
      axes[0].set_xlabel(f'Actual {target}')
      axes[0].set_ylabel(f'Predicted {target}')
      # Add text annotations for metrics
      axes[0].text(0.05, 0.95, f'MSE: {mse:.2f}', transform=axes[0].transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
      axes[0].text(0.05, 0.90, f'RMSE: {rmse:.2f}', transform=axes[0].transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
      axes[0].text(0.05, 0.85, f'R-squared: {r2:.2f}', transform=axes[0].transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
      axes[0].set_title('Random Forest Ensemble Predictions')
      axes[0].legend()

      # Plot feature importances

      bars = axes[1].barh(features, importances)
      for bar in bars:
          width = bar.get_width()
          axes[1].text(width, 
                  bar.get_y() + bar.get_height()/2,
                  f'{width:.3f}',
                  ha='left',
                  va='center',
                  fontsize=10)
      plt.xlabel('Importance Score')
      plt.title(f'Feature Relevance to {target}')
      plt.tight_layout()

      bars = axes[2].barh(features, shap_importance)
      for bar in bars:
          width = bar.get_width()
          axes[2].text(width, 
                  bar.get_y() + bar.get_height()/2,
                  f'{width:.3f}',
                  ha='left',
                  va='center',
                  fontsize=10)
      axes[1].set_xlabel('Importance Score SHAP')
      axes[1].set_title(f'Feature Relevance to {target}')
      plt.tight_layout()
      plt.savefig(f'Results_{option}/random_forest_diagnostics_{target}.png', dpi=300)
    #   plt.show()

    return model, mse, rmse, r2, y_pred

def tune_random_forest(x_train, y_train, random_state=42, n_iter=20):
    # Define parameter grid
    param_dist = {
        'n_estimators': [500, 1000, 2000],#np.random.randint(300, 600, size=10).tolist(),              # More trees for better performance
        'max_depth': [2, 5, 7, None], #np.random.randint(3, 10, size=3).tolist() + [None],                     # Wider depth range
        # 'min_samples_split': [2, 4, 8],#np.random.randint(2, 20, size=n_iter).tolist(),             # More granular control
        # 'min_samples_leaf': [1, 2, 4, 10],#np.random.randint(1, 10, size=n_iter).tolist(),              # Prevent overfitting
        'max_features': [1.0, 'log2', 'sqrt'], # Mix of strings and floats
        # 'criterion': ['squared_error', 'absolute_error'], # Different loss functions
        #'max_samples': [0.7, 0.8, 0.9, 1]#np.random.uniform(0.3, 0.7, size=n_iter)                 # Sample fraction for bootstrap
    }


    rf = RandomForestRegressor(random_state=random_state)

    rs = RandomizedSearchCV(rf,
                            param_distributions=param_dist,
                            n_iter=n_iter,
                            #cv=KFold(n_splits=10, shuffle=True, random_state=random_state),
                            cv=5,
                            scoring='r2',
                            random_state=random_state, 
                            return_train_score=True,
                            n_jobs=-1, 
                            verbose=1
                            )

    rs.fit(x_train, y_train)

    print(f"Best params: {rs.best_params_}")
    print(f"Best CV R2: {rs.best_score_:.3f}")
    # Check for overfitting
    results_df = pd.DataFrame(rs.cv_results_)
    best_idx = rs.best_index_
    train_score = results_df.loc[best_idx, 'mean_train_score']
    val_score = results_df.loc[best_idx, 'mean_test_score']

    print(f"Training R²: {train_score:.4f}")
    print(f"Validation R²: {val_score:.4f}")
    print(f"Overfitting gap: {train_score - val_score:.4f}")

    return rs.best_estimator_, rs.best_score_
    
def enhanced_multi_linear_regression_display(df, target, features, display=True):
    """
    Enhanced version of your function with proper diagnostics and model selection
    """
    
    print(f"=== ENHANCED REGRESSION ANALYSIS: {target} ===")
    
    # First, diagnose the target variable
    target_data = df[target].dropna()
    
    print(f"\nTARGET VARIABLE DIAGNOSTICS:")
    print(f"Mean: {target_data.mean():.3f}")
    print(f"Median: {target_data.median():.3f}")
    print(f"Std: {target_data.std():.3f}")
    print(f"Skewness: {target_data.skew():.3f}")
    print(f"Min: {target_data.min()}, Max: {target_data.max()}")
    print(f"Zeros: {(target_data == 0).sum()} ({(target_data == 0).mean()*100:.1f}%)")
    
    # Recommend analysis approach
    is_count = (target_data == target_data.astype(int)).all() and (target_data >= 0).all()
    is_highly_skewed = abs(target_data.skew()) > 2
    has_many_zeros = (target_data == 0).mean() > 0.3
    
    print(f"\nRECOMMENDATIONS:")
    if is_count:
        if has_many_zeros:
            print(" Use Zero-Inflated Poisson or Negative Binomial GLM")
        else:
            print(" Use Poisson or Negative Binomial GLM")
    
    if is_highly_skewed:
        print(" Consider log transformation or GLM with appropriate family")
    
    # Analyze each feature
    best_models = {}
    
    for feature in features:
        df[target]=df[target].dropna()
        print(f"\n{'='*50}")
        print(f"ANALYZING: {feature}")
        print(f"{'='*50}")
        
        # Correlation analysis
        correlations = {}
        for method in ['pearson', 'spearman', 'kendall']:
            corr = df[target].corr(df[feature], method=method)
            correlations[method] = corr
            print(f"{method.capitalize()} correlation: {corr:.4f}")
        
        models = {}
        X = sm.add_constant(df[feature].dropna())
        
        # 1. OLS (your current method)
        try:
            y = df[target]
            ols_model = sm.OLS(y, X).fit()
            models['OLS'] = ols_model
            
            print(f"\nOLS Results:")
            print(f"  R²: {ols_model.rsquared:.4f}")
            print(f"  p-value: {ols_model.f_pvalue:.4f}")
            print(f"  AIC: {ols_model.aic:.2f}")
            
        except Exception as e:
            print(f"OLS failed: {e}")

        # 2. Negative Binomial GLM (for count data)
        if is_count:
            try:
                negative_binomial_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
                models['Negative_Binomial'] = negative_binomial_model

                print(f"\nNegative Binomial GLM Results:")
                print(f"  Pseudo R²: {1 - negative_binomial_model.deviance/negative_binomial_model.null_deviance:.4f}")
                print(f"  p-value: {negative_binomial_model.pvalues.iloc[1]:.4f}")
                print(f"  AIC: {negative_binomial_model.aic:.2f}")
                
            except Exception as e:
                print(f"Poisson GLM failed: {e}")
        
        # 3. Log-transformed OLS (for skewed data)
        if is_highly_skewed:
            try:
                y_log = np.log1p(y)  # log(y+1)
                log_ols_model = sm.OLS(y_log, X).fit()
                models['Log_OLS'] = log_ols_model
                
                print(f"\nLog-transformed OLS Results:")
                print(f"  R²: {log_ols_model.rsquared:.4f}")
                print(f"  p-value: {log_ols_model.f_pvalue:.4f}")
                print(f"  AIC: {log_ols_model.aic:.2f}")
                
            except Exception as e:
                print(f"Log-transformed OLS failed: {e}")
        
        # Select best model for this feature
        if models:
            # Select based on AIC for GLMs, R² for OLS
            if 'Negative_Binomial' in models and models['Negative_Binomial'].aic < models.get('OLS', type('obj', (object,), {'aic': float('inf')})).aic:
                best_model = ('Negative_Binomial', models['Negative_Binomial'])
            elif 'Log_OLS' in models and models['Log_OLS'].rsquared > models.get('OLS', type('obj', (object,), {'rsquared': 0})).rsquared:
                best_model = ('Log_OLS', models['Log_OLS'])
            else:
                best_model = ('OLS', models['OLS'])
            
            best_models[feature] = best_model
            print(f"\nBEST MODEL for {feature}: {best_model[0]}")
        
        # Plotting
        if display and models:
            plt.figure(figsize=(15, 5))
            
            x_vals = df[feature]
            y_vals = df[target]
            
            for i, (model_name, model) in enumerate(models.items(), 1):
                plt.subplot(1, len(models), i)
                
                plt.scatter(x_vals, y_vals, alpha=0.6)
                
                # Predictions
                if model_name == 'Log_OLS':
                    y_pred_log = model.predict(X)
                    y_pred = np.expm1(y_pred_log)  # Transform back
                else:
                    y_pred = model.predict(X)
                
                plt.plot(x_vals, y_pred, 'r-', linewidth=2)
                
                # Statistics
                if hasattr(model, 'rsquared'):
                    r2 = model.rsquared
                    p_val = model.f_pvalue
                else:
                    r2 = 1 - model.deviance/model.null_deviance
                    p_val = model.pvalues[1]
                
                plt.title(f'{model_name}\nR²={r2:.3f}, p={p_val:.4f}')
                plt.xlabel(feature)
                plt.ylabel(target)
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    return best_models

def multi_GLM_display(df, targets, features, display = False):
    '''
    Performs multiple linear regression and displays the results.
    Outputs the best model based on R-squared value.
        Args:
            df: Pandas DataFrame containing the data.
            targets (list of str or str): Name(s) of the target variable(s) (dependent variable(s)).
            features: List of independent variable names (features).
            display: Whether to display the feature importance plot (default: False).

        Returns:
            best_model: List containing the best model's variable name, slope, intercept, mse, rmse, and r2.
        '''

    best_model = None
    family = sm.families.Gaussian()

    if isinstance(targets, str):
        targets = [targets]

    for target in targets:
        target_df = df[[target]+features].dropna()
        print(target)
        y = np.array(target_df[target])
        print(y.dtype)
        x = np.array(target_df[features])
        print(x.dtype)

        print(f"=== ENHANCED REGRESSION ANALYSIS: {target} ===")
        
        print(f"\nTARGET VARIABLE DIAGNOSTICS:")
        print(f"Mean: {y.mean():.3f}")
        print(f"Median: {np.median(y):.3f}")
        print(f"Std: {y.std():.3f}")
        print(f"Skewness: {stats.skew(y):.3f}")
        print(f"Min: {y.min()}, Max: {y.max()}")
        print(f"Zeros: {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
        
        # Recommend analysis approach
        is_count = (y == y.astype(int)).all() and (y >= 0).all()
        is_highly_skewed = abs(stats.skew(y)) > 1.5
        has_many_zeros = (y == 0).mean() > 0.3

        significance_level = 0.05

        print(f"\nRECOMMENDATIONS:")
        if is_count:
            if has_many_zeros:
                print(" Use Zero-Inflated Poisson or Negative Binomial GLM")
            else:
                print(" Use Poisson or Negative Binomial GLM")
        
        if is_highly_skewed:
            print(" Consider log transformation or GLM with appropriate family")
            if (y == 0).mean() > 0.3:
                print("  - Zero-Inflated or Hurdle model may be appropriate")

        try:
            if len(y) <= 5000:
                stat, p = stats.shapiro(y)
                test_name = 'shapiro'
            else:
                stat, p = stats.normaltest(y)
                test_name = 'normaltest'
        except Exception as e:
            print(f"Error performing normality test: {e}")
            stat, p, test_name = None, None, 'error'

        print({'test': test_name, 'stat': float(stat), 'pvalue': float(p), 'normal': (p > significance_level)})

        if any(x in target for x in ['proportion', 'canopy_openness']):
            print("Warning: Target variable is a proportion, using Binomial distribution")
            family = sm.families.Binomial()

        if p <= significance_level:
            print("Warning: Target variable is not normally distributed.")
            if is_highly_skewed:
                print("Considering log transformation or GLM with appropriate family")
                if (y == 0).mean() > 0.3:
                    print("  - Zero-Inflated or Hurdle model may be appropriate")
                if is_count:
                    if y.std()**2 / y.mean() > 1.5:
                        print("  - Overdispersed : Negative Binomial model may be appropriate")
                        family = sm.families.NegativeBinomial()
                    else:
                        print("  - Poisson model may be appropriate")
                        family = sm.families.Poisson()

        print(f"\nStandardizing features...")
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        x = sm.add_constant(x)  # Add constant term for intercept
        results = generalised_linear_model(x, y, family)
        print(results.params)
        y_pred = results.fittedvalues
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        print(f"Results for {target}:")
        print(results.summary())

        print(results.resid)
        if len(results.resid) <= 5000:
            stat, p = stats.shapiro(np.array(results.resid))
            test_name = 'shapiro'
        else:
            stat, p = stats.normaltest(np.array(results.resid))
            test_name = 'normaltest'

        bp_stat, bp_pvalue = het_breuschpagan(results.resid, results.model.exog)[:2]

        print({'test': test_name, 'stat': float(stat), 'pvalue': float(p), 'normal': (p > significance_level), 'bp_stat': float(bp_stat), 'bp_pvalue': float(bp_pvalue), 'bp_homogeneous': (bp_pvalue > significance_level)})

        if best_model is None:
            best_model = [target, y, x, results]
        elif results.rsquared > best_model[3].rsquared:
            best_model = [target, y, x, results]

        if display:

            fig, axes = plt.subplots(1, 3, figsize=(20, 6))

            # Bar plot of coefficients (predictor importance)
            best_coeffs = best_model[3].params[1:]
            axes[0].barh(features, best_coeffs, color='skyblue')
            axes[0].set_xlabel('Standardized Coefficient')
            axes[0].set_title('Predictor Importance')
            for i, v in enumerate(best_coeffs):
                axes[0].text(v, i, f'{v:.3f}', va='center', ha='left', fontsize=10)
            axes[0].grid(True, alpha=0.3)

            # Actual vs Predicted plot
            axes[1].scatter(y_pred, y, alpha=0.7)
            axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal fit')
            axes[1].set_xlabel(f'Predicted {target}')
            axes[1].set_ylabel(f'Actual {target}')
            axes[1].set_title(f'Actual vs Predicted {target}')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            # Annotate metrics
            axes[1].text(0.05, 0.95, f'MSE: {mse}\nRMSE: {rmse}\nR²: {r2:.2f}',
                        transform=axes[1].transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
            
            # Residuals vs Fitted plot
            residuals = y - y_pred
            axes[2].scatter(y_pred, residuals, alpha=0.7)
            axes[2].axhline(0, color='red', linestyle='--')
            axes[2].set_xlabel('Fitted Values')
            axes[2].set_ylabel('Residuals')
            axes[2].set_title('Residuals vs Fitted Values')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    if not display: 
        y = best_model[1]
        y_pred = best_model[3].predict(best_model[2])
        target = best_model[0]
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        coeffs = best_model[3].params[1:]

        print(f"Results for {target}:")
        print(best_model[3].summary())
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot of coefficients (predictor importance)
        axes[0].barh(features, coeffs, color='skyblue')
        axes[0].set_xlabel('Standardized Coefficient')
        axes[0].set_title('Predictor Importance')
        for i, v in enumerate(coeffs):
            axes[0].text(v, i, f'{v:.3f}', va='center', ha='left', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Actual vs Predicted plot
        axes[1].scatter(y_pred, y, alpha=0.7)
        axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ideal fit')
        axes[1].set_xlabel(f'Predicted {target}')
        axes[1].set_ylabel(f'Actual {target}')
        axes[1].set_title(f'Actual vs Predicted {target}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        # Annotate metrics
        axes[1].text(0.05, 0.95, f'MSE: {mse}\nRMSE: {rmse}\nR²: {r2:.2f}',
                    transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    print(f"Best model is for target {best_model[0]} with R² = {best_model[3].rsquared:.3f}")

def PCA_analysis(df, target_columns=None, treatment_column='treatment', n_components=None, display=True):
    """
    Performs Principal Component Analysis to investigate relationships between treatments
    and identify trends in the data.
    
    Args:
        df: Pandas DataFrame containing the data
        target_columns: List of columns to include in PCA. If None, uses all numeric columns except treatment
        treatment_column: Name of the treatment column for grouping and coloring
        n_components: Number of principal components to compute. If None, computes all
        display: Whether to display plots and detailed results
    
    Returns:
        dict: Dictionary containing PCA results, loadings, explained variance, etc.
    """

    
    print("="*80)
    print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("="*80)
    
    # Prepare data
    if target_columns is None:
        # Use all numeric columns except treatment and geometry
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [treatment_column, 'geometry', 'point.label']
        target_columns = [col for col in numeric_cols if col not in exclude_cols]
    
    print(f"Variables included in PCA: {target_columns}")
    print(f"Treatment column: {treatment_column}")
    
    # Separate features and treatments
    X = df[target_columns]
    treatments = df[treatment_column]
    
    # Standardize the features (important for PCA)
    print(f"\nStandardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    if n_components is None:
        n_components = min(len(target_columns), len(df))

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Calculate results
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    print(f"\nPCA RESULTS:")
    print(f"Number of components: {n_components}")
    print(f"Total variance explained by all components: {cumulative_variance[-1]:.3f}")
    
    # Show explained variance for each component
    print(f"\nExplained variance by component:")
    for i in range(min(5, len(explained_variance_ratio))):  # Show first 5 components
        print(f"  PC{i+1}: {explained_variance_ratio[i]:.3f} ({explained_variance_ratio[i]*100:.1f}%)")
        print(f"  Cumulative: {cumulative_variance[i]:.3f} ({cumulative_variance[i]*100:.1f}%)")
    
    unique_treatments = treatments.unique()

    if display:
        # 1. Scree plot
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 3, 1)
        plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
        plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, 'ro-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.legend(['Individual', 'Cumulative'])
        plt.grid(True, alpha=0.3)
        
        # Add 80% variance line
        plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.7, label='80% variance')
        
        # 2. PCA Biplot (PC1 vs PC2)
        plt.subplot(2, 3, 2)
        
        # Get unique treatments and assign colors
        unique_treatments = treatments.unique()
        unique_treatments = np.sort(unique_treatments)  # Sort treatments for consistent color mapping
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_treatments)))

        for treatment, color in zip(unique_treatments, colors):
            mask = treatments == treatment
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[color], label=treatment, alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% variance)')
        plt.title('PCA Biplot: PC1 vs PC2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add loading vectors
        for i, feature in enumerate(target_columns):
            plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, 
                     head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.6)
            plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, feature, 
                    fontsize=8, ha='center', va='center')
        
        # 3. PC1 vs PC3
        if n_components >= 3:
            plt.subplot(2, 3, 3)
            for treatment, color in zip(unique_treatments, colors):
                mask = treatments == treatment
                plt.scatter(X_pca[mask, 0], X_pca[mask, 2], 
                           c=[color], label=treatment, alpha=0.7, s=50)
            
            plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)')
            plt.ylabel(f'PC3 ({explained_variance_ratio[2]*100:.1f}% variance)')
            plt.title('PCA: PC1 vs PC3')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 4. Loadings heatmap - FIXED VERSION
        plt.subplot(2, 3, 4)
        n_components_to_show = min(5, n_components, len(target_columns))
        print(loadings)
        
        # Create loadings dataframe with correct dimensions
        loadings_df = pd.DataFrame(
            loadings[:, :n_components_to_show],  # This should be [n_variables, n_components]
            columns=[f'PC{i+1}' for i in range(n_components_to_show)],
            index=target_columns  # This should match the number of rows in loadings
        )
        
        # Debug print to verify dimensions
        print(f"DEBUG: loadings shape: {loadings.shape}")
        print(f"DEBUG: target_columns length: {len(target_columns)}")
        print(f"DEBUG: n_components_to_show: {n_components_to_show}")
        
        sns.heatmap(loadings_df, annot=True, cmap='RdBu_r', center=0, 
                   fmt='.2f', cbar_kws={'label': 'Loading'})
        plt.title('Variable Loadings on Principal Components')
        plt.ylabel('Variables')
        
        # 5. Treatment separation analysis
        plt.subplot(2, 3, 5)
        
        # Calculate centroids for each treatment
        treatment_centroids = {}
        for treatment in unique_treatments:
            mask = treatments == treatment
            centroid_pc1 = X_pca[mask, 0].mean()
            centroid_pc2 = X_pca[mask, 1].mean()
            treatment_centroids[treatment] = (centroid_pc1, centroid_pc2)
            
            # Plot individual points
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[list(unique_treatments).index(treatment)]], 
                       label=f'{treatment} (n={mask.sum()})', alpha=0.5, s=30)
            
            # Plot centroid
            plt.scatter(centroid_pc1, centroid_pc2, 
                       c=[colors[list(unique_treatments).index(treatment)]], 
                       s=200, marker='X', edgecolors='black', linewidth=2)
        
        plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% variance)')
        plt.title('Treatment Centroids and Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Variable contribution to PC1 and PC2
        plt.subplot(2, 3, 6)
        
        # Calculate variable contributions (squared loadings)
        contributions_pc1 = loadings[:, 0]**2
        contributions_pc2 = loadings[:, 1]**2
        
        x = np.arange(len(target_columns))
        width = 0.35
        
        plt.bar(x - width/2, contributions_pc1, width, label='PC1', alpha=0.7)
        plt.bar(x + width/2, contributions_pc2, width, label='PC2', alpha=0.7)
        
        plt.xlabel('Variables')
        plt.ylabel('Squared Loading (Contribution)')
        plt.title('Variable Contributions to PC1 and PC2')
        plt.xticks(x, target_columns, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("Results/PCA_analysis_results.png")
        plt.show()
    
    # Statistical analysis of treatment separation
    print(f"\n" + "="*60)
    print("TREATMENT SEPARATION ANALYSIS")
    print(f"="*60)
    
    # Calculate treatment centroids in PC space
    treatment_stats = {}
    for treatment in unique_treatments:
        mask = treatments == treatment
        treatment_data = X_pca[mask]
        
        treatment_stats[treatment] = {
            'n_samples': mask.sum(),
            'pc1_mean': treatment_data[:, 0].mean(),
            'pc1_std': treatment_data[:, 0].std(),
            'pc2_mean': treatment_data[:, 1].mean(),
            'pc2_std': treatment_data[:, 1].std(),
        }
        
        print(f"\n{treatment}:")
        print(f"  Sample size: {mask.sum()}")
        print(f"  PC1: {treatment_data[:, 0].mean():.3f} ± {treatment_data[:, 0].std():.3f}")
        print(f"  PC2: {treatment_data[:, 1].mean():.3f} ± {treatment_data[:, 1].std():.3f}")
    
    # Perform MANOVA-like analysis
    print(f"\n" + "="*60)
    print("MOST IMPORTANT VARIABLES")
    print(f"="*60)
    
    # Variables most important for PC1 and PC2
    pc1_importance = np.abs(loadings[:, 0])
    pc2_importance = np.abs(loadings[:, 1])
    
    pc1_top_vars = sorted(zip(target_columns, pc1_importance), key=lambda x: x[1], reverse=True)
    pc2_top_vars = sorted(zip(target_columns, pc2_importance), key=lambda x: x[1], reverse=True)
    
    print(f"Most important variables for PC1:")
    for i, (var, importance) in enumerate(pc1_top_vars[:5]):
        print(f"  {i+1}. {var}: {importance:.3f}")
    
    print(f"\nMost important variables for PC2:")
    for i, (var, importance) in enumerate(pc2_top_vars[:5]):
        print(f"  {i+1}. {var}: {importance:.3f}")
    
    # Calculate distances between treatment centroids
    print(f"\n" + "="*60)
    print("TREATMENT DISTANCES (Euclidean distance between centroids)")
    print(f"="*60)
    
    treatment_list = list(unique_treatments)
    for i in range(len(treatment_list)):
        for j in range(i+1, len(treatment_list)):
            t1, t2 = treatment_list[i], treatment_list[j]
            
            # Distance in PC1-PC2 space
            dist = np.sqrt(
                (treatment_stats[t1]['pc1_mean'] - treatment_stats[t2]['pc1_mean'])**2 +
                (treatment_stats[t1]['pc2_mean'] - treatment_stats[t2]['pc2_mean'])**2
            )
            print(f"{t1} ↔ {t2}: {dist:.3f}")
    
    # Return comprehensive results
    results = {
        'pca_model': pca,
        'scaler': scaler,
        'loadings': loadings,
        'loadings_df': pd.DataFrame(
            loadings, 
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=target_columns
        ),
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance': cumulative_variance,
        'transformed_data': X_pca,
        'treatment_stats': treatment_stats,
        'variable_names': target_columns,
        'treatments': treatments,
        'pc1_top_variables': pc1_top_vars,
        'pc2_top_variables': pc2_top_vars
    }
    
    return results, X_pca, treatments

def PCA_interpretation(pca_results, alpha=0.05):
    """
    Provides detailed interpretation of PCA results
    
    Args:
        pca_results: Dictionary returned from PCA_analysis
        alpha: Significance level for determining important loadings
    
    Returns:
        dict: Interpretation summary
    """
    
    print("="*80)
    print("PCA INTERPRETATION")
    print("="*80)
    
    explained_var = pca_results['explained_variance_ratio']
    loadings_df = pca_results['loadings_df']
    treatment_stats = pca_results['treatment_stats']
    
    interpretation = {}
    
    # 1. How many components to retain?
    cumvar = pca_results['cumulative_variance']
    
    # Kaiser criterion (eigenvalue > 1)
    eigenvalues = pca_results['pca_model'].explained_variance_
    kaiser_components = np.sum(eigenvalues > 1)
    
    # 80% variance criterion
    var_80_components = np.argmax(cumvar >= 0.8) + 1
    
    print(f"COMPONENT RETENTION CRITERIA:")
    print(f"  Kaiser criterion (eigenvalue > 1): {kaiser_components} components")
    print(f"  80% variance criterion: {var_80_components} components")
    print(f"  PC1 + PC2 explain: {cumvar[1]*100:.1f}% of total variance")
    
    # 2. Interpret PC1
    pc1_loadings = loadings_df['PC1'].abs().sort_values(ascending=False)
    significant_pc1 = pc1_loadings[pc1_loadings > pc1_loadings.quantile(1-alpha)]
    
    print(f"\nPC1 INTERPRETATION:")
    print(f"  Explains {explained_var[0]*100:.1f}% of variance")
    print(f"  Key variables (top loadings):")
    for var in significant_pc1.index[:5]:
        loading = loadings_df.loc[var, 'PC1']
        print(f"    {var}: {loading:.3f}")
    
    # 3. Interpret PC2
    pc2_loadings = loadings_df['PC2'].abs().sort_values(ascending=False)
    significant_pc2 = pc2_loadings[pc2_loadings > pc2_loadings.quantile(1-alpha)]
    
    print(f"\nPC2 INTERPRETATION:")
    print(f"  Explains {explained_var[1]*100:.1f}% of variance")
    print(f"  Key variables (top loadings):")
    for var in significant_pc2.index[:5]:
        loading = loadings_df.loc[var, 'PC2']
        print(f"    {var}: {loading:.3f}")
    
    # 4. Treatment separation
    print(f"\nTREATMENT SEPARATION:")
    
    treatments = list(treatment_stats.keys())
    max_separation = 0
    most_separated_pair = None
    
    for i in range(len(treatments)):
        for j in range(i+1, len(treatments)):
            t1, t2 = treatments[i], treatments[j]
            
            # Calculate separation in PC1-PC2 space
            pc1_diff = abs(treatment_stats[t1]['pc1_mean'] - treatment_stats[t2]['pc1_mean'])
            pc2_diff = abs(treatment_stats[t1]['pc2_mean'] - treatment_stats[t2]['pc2_mean'])
            total_separation = np.sqrt(pc1_diff**2 + pc2_diff**2)
            
            if total_separation > max_separation:
                max_separation = total_separation
                most_separated_pair = (t1, t2)
    
    if most_separated_pair:
        print(f"  Most separated treatments: {most_separated_pair[0]} and {most_separated_pair[1]}")
        print(f"  Separation distance: {max_separation:.3f}")
    
    # 5. Recommendations
    print(f"\nRECOMMENDATIONS:")
    
    if explained_var[0] > 0.4:
        print(f"   PC1 captures substantial variance - focus on its key variables")
    else:
        print(f"    PC1 captures low variance - consider more variables or different approach")
    
    if cumvar[1] > 0.6:
        print(f"   PC1+PC2 explain most variance - 2D visualization is informative")
    else:
        print(f"    PC1+PC2 explain little variance - consider additional components")
    
    if max_separation > 2.0:
        print(f"   Clear treatment separation - PCA successfully distinguishes treatments")
    elif max_separation > 1.0:
        print(f"   Moderate treatment separation - some treatment differences visible")
    else:
        print(f"   Poor treatment separation - treatments are similar in this variable space")
    
    print(f"   Focus research on variables with high PC1/PC2 loadings")
    print(f"   Consider interaction effects between key variables")
    
    interpretation = {
        'recommended_components': max(kaiser_components, var_80_components),
        'pc1_key_variables': significant_pc1.index.tolist(),
        'pc2_key_variables': significant_pc2.index.tolist(),
        'most_separated_treatments': most_separated_pair,
        'separation_quality': 'High' if max_separation > 2 else 'Medium' if max_separation > 1 else 'Low'
    }
    
    return interpretation

def comprehensive_PCA_analysis(df, display=True, target_columns=None, treatment_column='treatment'):
    """
    Performs complete PCA analysis with interpretation
    """
    
    print(" COMPREHENSIVE PCA ANALYSIS")
    
    # Perform PCAs
    pca_results, X_pca, treatments = PCA_analysis(df, target_columns, treatment_column, display=display)
    
    # Interpret results
    interpretation = PCA_interpretation(pca_results)

    return pca_results, interpretation, X_pca, treatments

# # Generate synthetic data
# np.random.seed(42)
# n = 100
# x = np.random.uniform(-10, 10, n)
# y = np.random.uniform(-10, 10, n)
# z = x + 30

# # Create DataFrame
# df = pd.DataFrame({'x': x, 'z': z})

# # Optionally add some noise
# # z = x**2 + y**2 + np.random.normal(0, 10, n)
# # df['z'] = z

# # Add squared terms for linear regression
# df['x2'] = df['x']
# # df['y2'] = np.sin(df['y'])

# # Test the function
# features = ['x2']# , 'y2']
# targets = ['z']
# multi_linear_regression_display(df, targets, features, display=True)