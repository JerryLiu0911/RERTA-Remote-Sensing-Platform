import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV, LeaveOneOut
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
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

def smart_feature_selection_pipeline(merged_df, target, all_possible_features):
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
        'height_metrics': [col for col in all_possible_features if 'CHM' in col and any(stat in col for stat in ['mean', 'median', 'max'])],
        'variability_metrics': [col for col in all_possible_features if 'CHM' in col and any(stat in col for stat in ['std', 'range', 'cv'])],
        'greenness_metrics': [col for col in all_possible_features if any(index in col for index in ['Clre', 'ReNDVI', 'GLI']) and any(stat in col for stat in ['std', 'range', 'cv'])],
        'texture_metrics': [col for col in all_possible_features if any(tex in col for tex in ['contrast', 'entropy', 'homogeneity', 'dissimilarity', 'ASM', 'energy', 'correlation'])]
    }
    # Select BEST representative from each category
    stage1_features = []
    for category, candidates in theory_based_candidates.items():
        if candidates:
            print(f"  {category}: {len(candidates)} candidates → selecting best {max(3, int(len(candidates)/4))}")
            
            # Calculate target correlations for candidates in this category
            category_corrs = []
            for feat in candidates:
                if feat in merged_df.columns:
                    corr = abs(merged_df[target].corr(merged_df[feat], method='spearman'))
                    category_corrs.append((feat, corr))
            
            # Take top 1-2 from each category
            category_corrs.sort(key=lambda x: x[1], reverse=True)
            selected_from_category = [feat for feat, _ in category_corrs[:max(3, int(len(candidates)/4))]]
            stage1_features.extend(selected_from_category)

            for feat, corr in category_corrs[:int(max(3, len(candidates)/4))]:
                print(f"     {feat}: {corr:.3f}")
    
    print(f"  Stage 1 result: {len(stage1_features)} features")
    
    # STAGE 2: Correlation-based refinement
    print(f"\n STAGE 2: CORRELATION-BASED REFINEMENT")
    
    # Remove highly correlated features within our selected set
    if len(stage1_features) > 1:

        # Find and remove redundant features
        feature_corr_matrix = merged_df[stage1_features].corr()
        
        # Find and remove redundant features
        to_remove = set()
        for i, feat1 in enumerate(stage1_features):
            for j, feat2 in enumerate(stage1_features[i+1:], i+1):
                if feat1 not in to_remove and feat2 not in to_remove:
                    corr = abs(feature_corr_matrix.iloc[i, j])
                    if corr > 0.8:  # High correlation threshold
                        # Keep the one with higher target correlation
                        target_corr1 = abs(merged_df[target].corr(merged_df[feat1]))
                        target_corr2 = abs(merged_df[target].corr(merged_df[feat2]))
                        
                        if target_corr1 >= target_corr2:
                            to_remove.add(feat2)
                            print(f"  Removing {feat2} (corr with {feat1}: {corr:.3f})")
                        else:
                            to_remove.add(feat1)
                            print(f"  Removing {feat1} (corr with {feat2}: {corr:.3f})")
        
        stage2_features = [f for f in stage1_features if f not in to_remove]
    else:
        stage2_features = stage1_features
    
    print(f"  Stage 2 result: {len(stage2_features)} features")
    sns.heatmap(merged_df[[column for column in merged_df.columns if column in stage2_features]].corr(method='spearman'), annot=True, fmt='.2f', cmap='coolwarm')
    plt.show()
    # STAGE 3: Sample size validation
    print(f"\n STAGE 3: SAMPLE SIZE VALIDATION")
    
    ratio = n_samples / len(stage2_features) if stage2_features else 0
    print(f"  Sample-to-feature ratio: {ratio:.1f}:1")
    
    if ratio < 5:
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
        
        print(f"  Final selection:")
        for feat, corr in final_rankings[:max_features]:
            print(f"     {feat}: {corr:.3f}")
    else:
        stage3_features = stage2_features
        print(f"   Feature count appropriate for dataset size")
    
    print(f"\n FINAL RESULT: {len(stage3_features)} high-quality features")
    print(f"   Features: {stage3_features}")
    print(f"   Final ratio: {n_samples/len(stage3_features):.1f}:1")
    
    return stage3_features

def data_diagnostics(df, dependent, group, identify_distributions = True, alpha=0.05, make_plots=False):
    """
    Run basic checks for one-way ANOVA suitability.
    Returns dict with diagnostics and a recommendation.
    """

    out = {}
    sub = df[[dependent, group]].dropna()
    if sub.shape[0] == 0:
        raise ValueError("No data for the requested columns.")
    
    # Split groups
    groups = [g[dependent].astype(float).values for _, g in sub.groupby(group)]
    labels = list(sub[group].unique())
    ns = [len(g) for g in groups]
    out['n_per_group'] = dict(zip(labels, ns))
    out['total_n'] = sub.shape[0]

    # Requirement checks
    out['groups_count'] = len(groups)
    out['independence_note'] = "Check study design: observations must be independent (cannot be tested automatically)."

    # Enhanced descriptive statistics per group
    desc_stats = {}
    for i, g in enumerate(groups):
        name = labels[i] if i < len(labels) else f'G{i}'
        if len(g) == 0:
            desc_stats[name] = {'count': 0}
            continue
            
        stats_dict = {
            'count': len(g),
            'mean': float(np.mean(g)),
            'median': float(np.median(g)),
            'std': float(np.std(g)),
            'min': float(np.min(g)),
            'max': float(np.max(g)),
            'range': float(np.max(g) - np.min(g)),
            'q1': float(np.percentile(g, 25)),
            'q3': float(np.percentile(g, 75)),
            'iqr': float(np.percentile(g, 75) - np.percentile(g, 25)),
            'skewness': float(stats.skew(g)),
            'kurtosis': float(stats.kurtosis(g)),
            'cv': float(np.std(g) / np.mean(g)) if np.mean(g) != 0 else np.nan
        }
        desc_stats[name] = stats_dict
    
    out['descriptive_stats'] = desc_stats
    
    # Print descriptive statistics summary
    print("\nDESCRIPTIVE STATISTICS BY GROUP:")
    for group_name, stats_dict in desc_stats.items():
        print(f"\n{group_name} (n={stats_dict.get('count', 0)}):")
        print(f"  Mean ± SD: {stats_dict.get('mean', np.nan):.3f} ± {stats_dict.get('std', np.nan):.3f}")
        print(f"  Median [Q1, Q3]: {stats_dict.get('median', np.nan):.3f} [{stats_dict.get('q1', np.nan):.3f}, {stats_dict.get('q3', np.nan):.3f}]")
        print(f"  Range: {stats_dict.get('min', np.nan):.3f} - {stats_dict.get('max', np.nan):.3f}")
        print(f"  Skewness: {stats_dict.get('skewness', np.nan):.3f}")
        print(f"  Kurtosis: {stats_dict.get('kurtosis', np.nan):.3f}")
        print(f"  CV: {stats_dict.get('cv', np.nan):.3f}")

    # Normality per group (Shapiro if sample small, D'Agostino for larger)
    normal_results = {}
    for i, g in enumerate(groups):
        name = labels[i] if i < len(labels) else f'G{i}'
        if len(g) < 3:
            normal_results[name] = {'test': None, 'pvalue': None, 'normal': None, 'note': 'too few samples'}
            continue
        try:
            if len(g) <= 5000:
                stat, p = stats.shapiro(g)
                test_name = 'shapiro'
            else:
                stat, p = stats.normaltest(g)
                test_name = 'normaltest'
            normal_results[name] = {'test': test_name, 'stat': float(stat), 'pvalue': float(p), 'normal': (p > alpha)}
        except Exception as e:
            normal_results[name] = {'test': 'error', 'error': str(e), 'normal': None}
    out['normality_per_group'] = normal_results

    # Homogeneity of variances - Levene (median is robust)
    try:
        lev_stat, lev_p = stats.levene(*groups, center='median')
        out['levene'] = {'stat': float(lev_stat), 'pvalue': float(lev_p), 'homogeneous': (lev_p > alpha)}
    except Exception as e:
        out['levene'] = {'error': str(e)}

    # Outliers via IQR per group
    outliers = {}
    for i, g in enumerate(groups):
        name = labels[i] if i < len(labels) else f'G{i}'
        if len(g) == 0:
            outliers[name] = {'count': 0}
            continue
        q1, q3 = np.percentile(g, [25,75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
        cnt = np.sum((g < lower) | (g > upper))
        outliers[name] = {'count': int(cnt), 'lower': float(lower), 'upper': float(upper)}
    out['outliers_per_group'] = outliers

    # Fit OLS and ANOVA table (type II)
    formula = f'Q("{(dependent)}") ~ C({group})'
    model = sm.OLS.from_formula(formula, data=sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    out['anova_table'] = anova_table.to_dict()
    # residual tests
    resid = model.resid
    if len(resid) >= 3:
        try:
            r_stat, r_p = stats.shapiro(resid) if len(resid) <= 5000 else stats.normaltest(resid)
            out['residual_normality'] = {'test': 'shapiro' if len(resid)<=5000 else 'normaltest',
                                            'stat': float(r_stat), 'pvalue': float(r_p), 'normal': (r_p > alpha)}
        except Exception as e:
            out['residual_normality'] = {'error': str(e)}
    else:
        out['residual_normality'] = {'note': 'too few residuals for test'}

    # Distribution identification
    if identify_distributions:
        warnings.filterwarnings('ignore')  # Suppress distribution fitting warnings
        
        distributions = {}
        # Candidate distributions to test - include both standard and GLM-relevant distributions
        candidate_dists = [
            stats.norm, stats.lognorm, stats.expon, stats.gamma, 
            stats.beta, stats.weibull_min, stats.t, stats.chi2,
            stats.nbinom, stats.poisson
        ]
        dist_names = ['Normal', 'Log-normal', 'Exponential', 'Gamma', 
                      'Beta', 'Weibull', 'Student\'s t', 'Chi-squared',
                      'Negative Binomial', 'Poisson']
        
        # Map distributions to GLM families
        glm_family_map = {
            'Normal': {'family': 'Gaussian', 'default_link': 'identity'},
            'Log-normal': {'family': 'Gaussian', 'default_link': 'log', 'transform': 'log'},
            'Gamma': {'family': 'Gamma', 'default_link': 'inverse'},
            'Exponential': {'family': 'Gamma', 'default_link': 'log'},
            'Poisson': {'family': 'Poisson', 'default_link': 'log'},
            'Negative Binomial': {'family': 'NegativeBinomial', 'default_link': 'log'},
            'Beta': {'family': 'Beta', 'default_link': 'logit'}
        }
        
        for i, g in enumerate(groups):
            name = labels[i] if i < len(labels) else f'G{i}'
            if len(g) < 5:  # Need enough data for fitting
                distributions[name] = {'best_fit': None, 'note': 'too few samples'}
                continue
                
            g_cleaned = g[~np.isnan(g)]  # Remove NaNs
            if len(g_cleaned) < 3:
                distributions[name] = {'best_fit': None, 'note': 'too few non-NaN samples'}
                continue
                
            # Check if this is count data
            is_count = np.all(g_cleaned == np.round(g_cleaned)) and np.all(g_cleaned >= 0)
            is_binary = np.all(np.isin(g_cleaned, [0, 1]))
            is_proportion = np.all((g_cleaned >= 0) & (g_cleaned <= 1))
            
            # Rescale data for distributions with bounded support
            g_min, g_max = np.min(g_cleaned), np.max(g_cleaned)
            g_range = g_max - g_min
            
            # Scale to [0,1] for Beta
            if g_range > 0:
                g_scaled_01 = (g_cleaned - g_min) / g_range
            else:
                g_scaled_01 = np.zeros_like(g_cleaned)
            
            # Scale to positive for gamma, lognorm, etc.
            if g_min <= 0:
                g_scaled_pos = g_cleaned - g_min + 0.01  # Shift to positive
            else:
                g_scaled_pos = g_cleaned
                
            # Fit distributions and calculate AIC
            results = []
            
            # Calculate additional GLM-relevant statistics
            mean_val = np.mean(g_cleaned)
            var_val = np.var(g_cleaned)
            dispersion = var_val / mean_val if mean_val > 0 else np.nan
            zero_prop = np.mean(g_cleaned == 0)
            
            for j, dist in enumerate(candidate_dists):
                try:
                    # Skip distributions that don't match data characteristics
                    if is_count and dist not in [stats.poisson, stats.nbinom]:
                        continue
                    if is_binary and dist not in [stats.bernoulli, stats.binom]:
                        continue
                    if is_proportion and dist not in [stats.beta]:
                        continue
                        
                    # Use appropriate data scaling for each distribution
                    if dist == stats.beta:
                        data_to_fit = g_scaled_01
                    elif dist in [stats.lognorm, stats.gamma, stats.weibull_min, stats.chi2, stats.expon]:
                        data_to_fit = g_scaled_pos
                    elif dist == stats.poisson or dist == stats.nbinom:
                        if not is_count:
                            continue
                        data_to_fit = g_cleaned
                    else:
                        data_to_fit = g_cleaned
                        
                    # Fit distribution
                    params = dist.fit(data_to_fit)
                    
                    # Calculate log-likelihood
                    log_likelihood = np.sum(dist.logpdf(data_to_fit, *params))
                    
                    # Calculate AIC
                    k = len(params)
                    aic = 2 * k - 2 * log_likelihood
                    
                    # K-S test
                    ks_stat, ks_p = stats.kstest(data_to_fit, dist.cdf, args=params)
                    
                    results.append({
                        'distribution': dist_names[j],
                        'params': params,
                        'aic': aic,
                        'ks_stat': ks_stat,
                        'ks_p': ks_p,
                        'dist_obj': dist,
                        'scaled_data': data_to_fit
                    })
                except Exception as e:
                    continue  # Skip failed fits
                    
            # Find best fit (lowest AIC)
            if results:
                results.sort(key=lambda x: x['aic'])
                best_fit = results[0]
                
                # Get GLM family information if available
                glm_family = glm_family_map.get(best_fit['distribution'], 
                                              {'family': 'Unknown', 'default_link': 'identity'})
                
                # Override family based on data characteristics
                if is_count and dispersion > 1.5:  # Overdispersion detected
                    glm_family = {'family': 'Negative Binomial', 'default_link': 'log'}
                elif is_count:
                    glm_family = {'family': 'Poisson', 'default_link': 'log'}
                elif is_binary:
                    glm_family = {'family': 'Binomial', 'default_link': 'logit'}
                elif is_proportion:
                    glm_family = {'family': 'Beta', 'default_link': 'logit'}
                    
                # Handle zero-inflation
                if is_count and zero_prop > 0.3:
                    glm_family['zero_inflated'] = True
                    glm_family['family'] = f"Zero-inflated {glm_family['family']}"
                
                distributions[name] = {
                    'best_fit': best_fit['distribution'],
                    'aic': best_fit['aic'],
                    'ks_p': best_fit['ks_p'],
                    'params': best_fit['params'],
                    'dist_obj': best_fit['dist_obj'],
                    'all_fits': results,
                    'scaled_data': best_fit['scaled_data'],
                    'glm_family': glm_family,
                    'is_count': is_count,
                    'is_binary': is_binary,
                    'is_proportion': is_proportion,
                    'dispersion': dispersion,
                    'zero_proportion': zero_prop
                }
                
                # Interpret skewness and kurtosis
                skew = stats_dict.get('skewness', 0)
                kurt = stats_dict.get('kurtosis', 0)
                
                skew_interp = "approximately symmetric"
                if skew > 1:
                    skew_interp = "highly positively skewed"
                elif skew > 0.5:
                    skew_interp = "moderately positively skewed"
                elif skew < -1:
                    skew_interp = "highly negatively skewed"
                elif skew < -0.5:
                    skew_interp = "moderately negatively skewed"
                    
                kurt_interp = "mesokurtic (normal-like)"
                if kurt > 3:
                    kurt_interp = "leptokurtic (heavy-tailed)"
                elif kurt < -0.5:
                    kurt_interp = "platykurtic (light-tailed)"
                    
                distributions[name]['shape_interp'] = {
                    'skewness': skew_interp,
                    'kurtosis': kurt_interp
                }
            else:
                distributions[name] = {'best_fit': None, 'note': 'fitting failed for all distributions'}
                
        out['distributions'] = distributions
        
        # Print distribution analysis
        print("\nDISTRIBUTION AND GLM ANALYSIS:")
        for group_name, dist_info in distributions.items():
            print(f"\n{group_name}:")
            if dist_info.get('best_fit'):
                print(f"  Best fit: {dist_info['best_fit']}")
                print(f"  AIC: {dist_info.get('aic', 'N/A'):.3f}")
                print(f"  KS test p-value: {dist_info.get('ks_p', 'N/A'):.3f}")
                
                if 'glm_family' in dist_info:
                    glm_family = dist_info['glm_family']
                    print(f"  Recommended GLM family: {glm_family['family']}")
                    print(f"  Recommended link function: {glm_family['default_link']}")
                    
                    # Additional GLM diagnostics
                    if dist_info.get('is_count'):
                        print(f"  Data type: Count data")
                        print(f"  Dispersion: {dist_info.get('dispersion', 'N/A'):.3f}")
                        if dist_info.get('dispersion', 1) > 1.5:
                            print(f"   Overdispersion detected - consider Negative Binomial")
                        if dist_info.get('zero_proportion', 0) > 0.3:
                            print(f"   Zero-inflation detected ({dist_info.get('zero_proportion', 0)*100:.1f}%)")
                    elif dist_info.get('is_binary'):
                        print(f"  Data type: Binary data")
                    elif dist_info.get('is_proportion'):
                        print(f"  Data type: Proportion data")
                        
                if 'shape_interp' in dist_info:
                    print(f"  Distribution shape: {dist_info['shape_interp']['skewness']}, {dist_info['shape_interp']['kurtosis']}")
            else:
                print(f"  {dist_info.get('note', 'No distribution identified')}")

    # Recommendation logic
    normals = [v['normal'] for v in normal_results.values() if isinstance(v.get('normal'), (bool, np.bool_))]
    all_normal = all(normals) if normals else None
    hom = out.get('levene', {}).get('homogeneous', None)

    if all_normal is True and hom is True:
        rec = "One-way ANOVA is appropriate (assumptions satisfied)."
    elif hom is False and all_normal is True:
        rec = "Variances unequal -> use Welch's ANOVA or robust methods."
    elif all_normal is False and hom is True:
        rec = "Non-normal groups -> Kruskal-Wallis may be preferable (nonparametric)."
    elif all_normal is False and hom is False:
        rec = "Both normality and homogeneity violated -> consider nonparametric tests or transform data."
    else:
        rec = "Insufficient info; inspect plots and group sizes."

    print(f"All normal: {all_normal}, Homogeneous: {hom}")

    out['recommendation'] = rec

    # Optional plots
    if make_plots:

        
        # 1. Combined figure with histograms and distribution fits
        for i, g_name in enumerate(labels):
            g = groups[i]
            if len(g) < 3:
                continue
                
            # Create histogram with distribution fit
            fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
            
            # Histogram and density plot
            ax1 = plt.subplot(gs[0, :])
            sns.histplot(g, bins=30, kde=True, stat='density', alpha=0.6, ax=ax1)
            
            # Add descriptive stats text box
            stats_text = (
                f"n = {desc_stats[g_name]['count']}\n"
                f"Mean ± SD: {desc_stats[g_name]['mean']:.3f} ± {desc_stats[g_name]['std']:.3f}\n"
                f"Median: {desc_stats[g_name]['median']:.3f}\n"
                f"Range: [{desc_stats[g_name]['min']:.3f}, {desc_stats[g_name]['max']:.3f}]\n"
                f"Skewness: {desc_stats[g_name]['skewness']:.3f}\n"
                f"Kurtosis: {desc_stats[g_name]['kurtosis']:.3f}"
            )
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add distribution fits if available
            if identify_distributions and 'distributions' in out and g_name in out['distributions']:
                dist_info = out['distributions'][g_name]
                if dist_info.get('best_fit') and dist_info.get('dist_obj') and dist_info.get('params'):
                    dist_obj = dist_info['dist_obj']
                    params = dist_info['params']
                    scaled_data = dist_info['scaled_data']
                    
                    # Generate points for PDF
                    x = np.linspace(np.min(scaled_data), np.max(scaled_data), 1000)
                    y = dist_obj.pdf(x, *params)
                    
                    ax1.plot(x, y, 'r-', linewidth=2, label=f"Best fit: {dist_info['best_fit']}")
                    
                    # Add top 3 distributions
                    if 'all_fits' in dist_info and len(dist_info['all_fits']) > 1:
                        colors = ['g-', 'b-', 'm-']
                        for j, fit_info in enumerate(dist_info['all_fits'][1:4]):  # Next 3 best fits
                            dist_obj = fit_info['dist_obj']
                            params = fit_info['params']
                            
                            y = dist_obj.pdf(x, *params)
                            ax1.plot(x, y, colors[j % len(colors)], linewidth=1, alpha=0.7, 
                                    label=f"{fit_info['distribution']} (AIC: {fit_info['aic']:.1f})")
            
            ax1.set_title(f'Distribution Analysis: {g_name}', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # QQ plot
            ax2 = plt.subplot(gs[1, 0])
            stats.probplot(g, dist="norm", plot=ax2)
            ax2.set_title('Normal Q-Q Plot')
            ax2.grid(True, alpha=0.3)
            
            # Boxplot
            ax3 = plt.subplot(gs[1, 1])
            sns.boxplot(y=g, ax=ax3)
            ax3.set_title('Boxplot')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        
        # 2. Group comparison boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=group, y=dependent, data=sub)
        plt.title('Group Comparison', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 3. QQ plot of residuals if model exists
        if 'anova_table' in out:
            plt.figure(figsize=(8, 6))
            sm.qqplot(model.resid, line='s')
            plt.title('Q-Q Plot of Residuals', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.show()

    return out

def load_data(dataframes, filter = None, dependent_variables = ["average_canopy_openness", "Frog.richness", "Frog.abundance"]):
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

        print(f"Loaded data from {name}")
        print(df.head())

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
                        print(f"dropping columns {column}")
                        print("remaining columns : ", df.columns)
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
  x = sm.add_constant(x)  # Add constant term for intercept


  # Calculate coefficients using the matrix method: (X^T * X)^(-1) * X^T * y
  # X.T is the transpose of X
  # np.linalg.inv() calculates the inverse of a matrix
  # @ is the matrix multiplication operator
  # coefficients = np.linalg.inv(X.T @ X) @ X.T @ y

  # The coefficients are [b, m]
  # b, m = coefficients

  results = sm.OLS(y, x).fit()  # Fit the model

  return results

def multi_linear_regression_display(df, targets, features, display = False):
  '''
  Performs multiple linear regression and displays the results.
  Outputs the best model based on R-squared value.
    Args:
        df: Pandas DataFrame containing the data.
        target: Name of the target variable (dependent variable).
        features: List of independent variable names (features).
        display: Whether to display the feature importance plot (default: False).

    Returns:
        best_model: List containing the best model's variable name, slope, intercept, mse, rmse, and r2.
    '''
  
  best_model = []
  
  for target in targets:
    print(f"Correlation of with {target}:")
    print(df[target].corr(df[features], method='spearman'))
    x = np.array(df[features])
    y = np.array(df[target])
    results = simple_linear_regression(x, y)
    b, m = results.params[0], results.params[1]
    y_pred = m * x + b
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    print(f"Results for {target}:")
    print(results.summary())
    
    if len(best_model)==0:
      best_model = [target, m, b, mse, rmse, r2]
    elif r2 > best_model[5]:
      best_model = [target, m, b, mse, rmse, r2]

    if display:
      plt.figure(figsize=(10, 6))
      plt.subplot(1, 1, 1)
      plt.scatter(y_pred, y, label='Data')
      # Add labels and title for the plot
      plt.xlabel(f"predicted_{target}")
      plt.ylabel(target)
      plt.title(f'Linear Regression of predicted {target} vs. {target}')

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
    plt.grid(True)
    plt.show()
       
  print(f'Best model: {best_model}')

  # Assuming you have your x and y data and the calculated slope (m) and intercept (b)

  x = np.array(df[best_model[0]])
  y = np.array(df[target])
  m = best_model[1]
  b = best_model[2]

def random_forest_regression(df, target, features, display=True, test_size=0.2, random_state=42):
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
      plt.figure()
      plt.scatter(y_test, y_pred_test, color='red')
      plt.xlabel('Actual Frog Abundance')
      plt.ylabel('Predicted Frog Abundance')
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
      plt.xlabel('Importance Score SHAP')
      plt.title(f'Feature Relevance to {target}')
      plt.tight_layout()

      plt.figure(figsize=(10,6))
      bars = plt.barh(features, shap_importance)
      for bar in bars:
          width = bar.get_width()
          plt.text(width, 
                  bar.get_y() + bar.get_height()/2,
                  f'{width:.3f}',
                  ha='left',
                  va='center',
                  fontsize=10)
      plt.xlabel('Importance Score')
      plt.title(f'Feature Relevance to {target}')
      plt.tight_layout()
      plt.show()

    return model, mse, rmse, r2, y_pred

def random_forest_ensemble(df, target, features, n_estimators=100, test_size=0.2, random_state=42):
    """
    This function performs k-fold cross-validation on the training data, training a separate Random Forest model on each fold and recording its validation score. After all models are trained,
    each one makes predictions on the test set, and these predictions are combined using a weighted average, where the weights are based on the models’ respective validation scores. This results in an ensemble prediction on the test set, with better-performing models (on their own folds) contributing more to the final output.

    # There are big issues with this method. Only a limited number of data are used for training, and they may not generalize well to unseen data.

    Args:
        df: Pandas DataFrame containing the data.
        target: Name of the target variable (dependent variable).
        features: List of independent variable names (features).
        n_estimators: Number of trees in the forest (default: 100)
        test_size: Proportion of dataset to include in the test split (default: 0.2)
        random_state: Random state for reproducibility (default: 42)

    Returns:
        y_pred_sum: Sum of predicted values from the ensemble model.
    """
    # Reshape features to be a 2D array
    # x = features.reshape(-1, 1)
    # y = target
    y = np.array(df[target])
    x = np.array(df[features])


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

def generalised_linear_model(df, target, features, display=False):
    """
    General linear model function.
    """
    X = df[features]
    y = df[target]
    model = sm.GLM(y, sm.add_constant(X)).fit()
    print(model.summary())
    return model

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

def enhanced_multivariate_linear_regression(df, targets, features, log_transform=True, display=True):
    """
    Performs true multivariate linear regression with multiple response variables simultaneously.
    
    Args:
        df: Pandas DataFrame containing the data
        targets: List of target variable names (multiple dependent variables)
        features: List of independent variable names (features)
        log_transform: Whether to apply log transformation to target variables
        display: Whether to display plots and detailed output
    
    Returns:
        dict: Dictionary containing multivariate regression results
    """
    
    print(f"=== TRUE MULTIVARIATE LINEAR REGRESSION ===")
    print(f"Target variables: {targets}")
    print(f"Features: {features}")
    print(f"Sample size: {len(df)}")
    
    # Clean data
    analysis_cols = targets + features
    df_clean = df[analysis_cols].dropna()
    
    if len(df_clean) < len(df):
        print(f"Removed {len(df) - len(df_clean)} rows with missing values")
        print(f"Final sample size: {len(df_clean)}")
    
    # Log transform target variables if requested
    if log_transform:
        print(f"\n=== LOG TRANSFORMATION ===")
        Y_original = df_clean[targets].copy()
        Y_transformed = pd.DataFrame(index=df_clean.index)
        
        for target in targets:
            min_val = df_clean[target].min()
            if min_val <= 0:
                print(f"Using log1p transformation for {target} (has non-positive values)")
                Y_transformed[target] = np.log1p(df_clean[target])
            else:
                print(f"Using natural log transformation for {target}")
                Y_transformed[target] = np.log(df_clean[target])
        
        Y = Y_transformed
    else:
        Y = df_clean[targets]
        Y_original = Y.copy()
    
    # Standardize features
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[features])
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df_clean.index)
    
    # Add constant term
    X_with_const = sm.add_constant(X_scaled_df)
    
    # Fit TRUE multivariate regression using seemingly unrelated regression (SUR)
    # Or matrix form: Y = XB + E where Y is n×p, X is n×k, B is k×p
    
    print(f"\n=== MULTIVARIATE MODEL FITTING ===")
    
    # Method 1: Matrix-based multivariate regression
    X_matrix = X_with_const.values
    Y_matrix = Y.values
    
    # Calculate coefficient matrix: B = (X'X)^(-1)X'Y
    XtX_inv = np.linalg.inv(X_matrix.T @ X_matrix)
    B_matrix = XtX_inv @ X_matrix.T @ Y_matrix
    
    # Predictions
    Y_pred_matrix = X_matrix @ B_matrix
    
    # Residuals
    residuals_matrix = Y_matrix - Y_pred_matrix
    
    # Calculate multivariate statistics
    n, p = Y_matrix.shape  # n samples, p response variables
    k = X_matrix.shape[1]  # k predictors (including intercept)
    
    # Residual sum of squares matrix (E'E)
    E_matrix = residuals_matrix.T @ residuals_matrix
    
    # Total sum of squares matrix
    Y_centered = Y_matrix - Y_matrix.mean(axis=0)
    T_matrix = Y_centered.T @ Y_centered
    
    # Multivariate R-squared (Wilks' Lambda based)
    try:
        wilks_lambda = np.linalg.det(E_matrix) / np.linalg.det(T_matrix)
        multivariate_r2 = 1 - wilks_lambda
    except:
        multivariate_r2 = np.nan
        wilks_lambda = np.nan
    
    print(f"Multivariate R²: {multivariate_r2:.4f}")
    print(f"Wilks' Lambda: {wilks_lambda:.4f}")
    
    # Individual response variable statistics
    individual_r2 = {}
    coefficients_df = pd.DataFrame(
        B_matrix, 
        index=['Intercept'] + features, 
        columns=targets
    )
    
    print(f"\n=== COEFFICIENT MATRIX ===")
    print(coefficients_df.round(4))
    
    print(f"\n=== INDIVIDUAL RESPONSE STATISTICS ===")
    for i, target in enumerate(targets):
        y_true = Y_matrix[:, i]
        y_pred = Y_pred_matrix[:, i]
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        individual_r2[target] = r2
        
        print(f"{target}:")
        print(f"  R²: {r2:.4f}")
        print(f"  Coefficients: {coefficients_df[target].round(4).to_dict()}")
    
    # Multivariate hypothesis testing (MANOVA-style)
    print(f"\n=== MULTIVARIATE HYPOTHESIS TESTING ===")
    
    # Calculate F-statistic for overall model significance
    try:
        # Degrees of freedom
        df_model = k - 1  # excluding intercept
        df_error = n - k
        df_total = n - 1
        
        # Pillai's trace
        try:
            pillai_trace = np.trace(np.linalg.solve(T_matrix, T_matrix - E_matrix))
            print(f"Pillai's Trace: {pillai_trace:.4f}")
        except:
            pillai_trace = np.nan
            print(f"Pillai's Trace: Could not calculate")
        
        # Roy's largest root
        try:
            eigenvals = np.linalg.eigvals(np.linalg.solve(E_matrix, T_matrix - E_matrix))
            roy_root = np.max(eigenvals.real)
            print(f"Roy's Largest Root: {roy_root:.4f}")
        except:
            roy_root = np.nan
            print(f"Roy's Largest Root: Could not calculate")
            
    except Exception as e:
        print(f"Multivariate test statistics could not be calculated: {e}")
    
    if display:
        # Visualization
        n_targets = len(targets)
        
        # 1. Actual vs Predicted for each response
        fig, axes = plt.subplots(2, n_targets, figsize=(5 * n_targets, 8))
        if n_targets == 1:
            axes = axes.reshape(-1, 1)
        
        for i, target in enumerate(targets):
            y_true = Y_matrix[:, i]
            y_pred = Y_pred_matrix[:, i]
            residuals = residuals_matrix[:, i]
            
            # Actual vs Predicted
            axes[0, i].scatter(y_true, y_pred, alpha=0.6)
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[0, i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            title_prefix = f"log({target})" if log_transform else target
            axes[0, i].set_xlabel(f'Actual {title_prefix}')
            axes[0, i].set_ylabel(f'Predicted {title_prefix}')
            axes[0, i].set_title(f'{title_prefix}: R² = {individual_r2[target]:.3f}')
            axes[0, i].grid(True, alpha=0.3)
            
            # Residuals vs Fitted
            axes[1, i].scatter(y_pred, residuals, alpha=0.6)
            axes[1, i].axhline(y=0, color='red', linestyle='--')
            axes[1, i].set_xlabel('Fitted Values')
            axes[1, i].set_ylabel('Residuals')
            axes[1, i].set_title(f'{title_prefix}: Residuals')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle(f'Multivariate Regression Results (Multivariate R² = {multivariate_r2:.3f})')
        plt.tight_layout()
        plt.show()
        
        # 2. Coefficient heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(coefficients_df.T, annot=True, cmap='RdBu_r', center=0,
                   fmt='.3f', cbar_kws={'label': 'Coefficient Value'})
        plt.title('Multivariate Regression Coefficients')
        plt.xlabel('Predictors')
        plt.ylabel('Response Variables')
        plt.tight_layout()
        plt.show()
        
        # 3. Response correlation matrix
        if len(targets) > 1:
            plt.figure(figsize=(8, 6))
            response_corr = pd.DataFrame(Y_matrix, columns=targets).corr()
            sns.heatmap(response_corr, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f', cbar_kws={"shrink": .8})
            title_suffix = " (log-transformed)" if log_transform else ""
            plt.title(f'Response Variable Correlations{title_suffix}')
            plt.tight_layout()
            plt.show()
    
    return {
        'coefficient_matrix': coefficients_df,
        'multivariate_r2': multivariate_r2,
        'individual_r2': individual_r2,
        'wilks_lambda': wilks_lambda,
        'predictions': pd.DataFrame(Y_pred_matrix, columns=targets, index=df_clean.index),
        'residuals': pd.DataFrame(residuals_matrix, columns=targets, index=df_clean.index),
        'scaler': scaler,
        'sample_size': len(df_clean),
        'log_transformed': log_transform,
        'Y_matrix': Y_matrix,
        'X_matrix': X_matrix,
        'n_predictors': k,
        'n_responses': p
    }
    
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

