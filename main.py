import align_coords
import coordinate_extraction
import statistical_modelling
import gis as gis
import matplotlib.pyplot as plt
import numpy as np


def plot_relations(df, target, column, geo = False):
    """
    Plots average canopy openness against CHM for the BC points.
    """
    if geo:
        plt.scatter(df[target], df['point.label']
                    , label=target
                    , color='blue', alpha=0.5)
        plt.scatter(df[column], df['point.label']
                    , label= 'CHM'
                    , color='red', alpha=0.5)
        plt.xlabel(column + ' and ' + target)
        plt.ylabel('point.label')
        plt.legend()
        plt.show()
        
    else:
        plt.scatter(df[target], df[column]
                    , label=target
                    , color='blue', alpha=0.5
                    , s=10)
        plt.xlabel(target)
        plt.ylabel(column)
        plt.legend()
        plt.show()


def analyze_chm_correlations(merged_df, features):
    """Analyze correlations in your CHM features specifically"""
    
    print("=== CHM FEATURE CORRELATION ANALYSIS ===")
    
    corr_matrix = merged_df[features].corr()
    
    # Identify correlation clusters
    print("Correlation matrix:")
    print(corr_matrix.round(3))
    
    # Find redundant features
    redundant_pairs = []
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features[i+1:], i+1):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > 0.9:
                redundant_pairs.append((feat1, feat2, corr))
    
    if redundant_pairs:
        print(f"\n🚨 HIGHLY REDUNDANT FEATURES:")
        for feat1, feat2, corr in redundant_pairs:
            print(f"   {feat1} ↔ {feat2}: {corr:.3f}")
        
        print(f"\nImpact on Random Forest:")
        print(f"• Feature importance unreliable")
        print(f"• CV results highly variable") 
        print(f"• Overfitting more likely")
        print(f"• Poor generalization")
    
    # Suggest feature reduction
    print(f"\n=== FEATURE REDUCTION RECOMMENDATIONS ===")
    
    # Group highly correlated features
    correlation_groups = []
    used_features = set()
    
    for feat1 in features:
        if feat1 in used_features:
            continue
            
        group = [feat1]
        used_features.add(feat1)
        
        for feat2 in features:
            if feat2 != feat1 and feat2 not in used_features:
                corr = abs(corr_matrix.loc[feat1, feat2])
                if corr > 0.8:
                    group.append(feat2)
                    used_features.add(feat2)
        
        if len(group) > 1:
            correlation_groups.append(group)
    
    print(f"Correlation groups found: {len(correlation_groups)}")
    for i, group in enumerate(correlation_groups):
        print(f"  Group {i+1}: {group}")
        
        # Suggest which to keep
        target_corrs = []
        for feat in group:
            target_corr = abs(merged_df['average_canopy_openness'].corr(merged_df[feat]))
            target_corrs.append((feat, target_corr))
        
        best_feature = max(target_corrs, key=lambda x: x[1])
        print(f"    → Keep: {best_feature[0]} (target correlation: {best_feature[1]:.3f})")
        print(f"    → Remove: {[f for f, _ in target_corrs if f != best_feature[0]]}")


def feature_diagnostics(merged_df, target, features):
    print("=== CRITICAL DATASET ANALYSIS ===")
    print(f"Total samples: {len(merged_df)}")
    print(f"Total features: {len(features)}")
    print(f"Sample-to-feature ratio: {len(merged_df)/len(features):.1f}")
    print(f"Features: {features}")
    
    # Check for the exact problem
    if len(merged_df) < 30:
        print("🚨 DATASET TOO SMALL FOR RELIABLE CV!")
        print("Cross-validation becomes meaningless with tiny datasets")
        print("Each CV fold has ~2-5 samples - no statistical power")
    
    # Check feature correlations
    print(f"\n=== FEATURE CORRELATION ANALYSIS ===")
    feature_corr_matrix = merged_df[features].corr()
    high_corr_pairs = []
    
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            corr_val = abs(feature_corr_matrix.iloc[i, j])
            if corr_val > 0.8:
                high_corr_pairs.append((features[i], features[j], corr_val))
    
    if high_corr_pairs:
        print("🚨 HIGHLY CORRELATED FEATURES DETECTED:")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"   {feat1} ↔ {feat2}: {corr:.3f}")
        print("This causes multicollinearity and unstable CV results!")
    
    # Check target-feature relationships
    print(f"\n=== TARGET-FEATURE RELATIONSHIPS ===")
    for feature in features:
        corr = merged_df['average_canopy_openness'].corr(merged_df[feature])
        print(f"{feature}: {corr:.3f}")

def main(paths):

    # coordinate_extraction.extract_corner_coords(paths['coordinates'], paths['result_data'])
    # align_coords.canopy_openness(paths['canopy_openness'], paths['result_data'], paths['canopy_openness_result'], timepoint=2019)
    # gis.zonal_statistics(paths['canopy_openness_result'], "G:/My Drive/UROP/UROP Rerta Palapa June2019 CHM.tif", paths['buffered_points'], paths['CHM'], buffer_geom_path=paths['result_data'])

    merged_df = statistical_modelling.load_data(
            [#('GLI', paths['GLI']),
            #('ExG', paths['ExG']),
            #('DEM', paths['DEM']),
            ('CHM', paths['CHM'])], filter = "OPE|BC")

    # print(merged_df)
    BC_df = merged_df#[merged_df['point.label'].str.contains("BC", case=False, na=False)]
    print(BC_df.head())
    # plot_relations(BC_df,'average_canopy_openness', 'mean_CHM', geo = False)
    # print(merged_df.columns)

    features = [column for column in merged_df.columns if'CHM' in column and column != 'geometry_CHM' and column != 'name_CHM']
    #print(merged_df[features].describe())
    print(merged_df['average_canopy_openness'].describe())
    feature_diagnostics(merged_df, 'average_canopy_openness', features)
    analyze_chm_correlations(merged_df, features)
    # coordinate_extraction.extract_corner_coords(paths['coordinates'], paths['result_data'])
    # gis.create_buffer(gis.gpd.read_file(paths['canopy_openness_result']), paths['buffered_points'], buffer_geom=gis.gpd.read_file(paths['result_data']))

    # print([feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'name','average_canopy_openness']])
    statistical_modelling.random_forest_regression(merged_df, 'average_canopy_openness', [feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'name_CHM','average_canopy_openness']])
    # statistical_modelling.random_forest_ensemble(merged_df, 'average_canopy_openness', [feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'average_canopy_openness']])
    # statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if'CHM' in column and column != 'geometry_CHM' and column != 'name_CHM'], display=False)
    # statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if column not in ['geometry', 'point.label', 'average_canopy_openness']], display=False)


paths = {
'canopy_openness': "Data/3.4-canopy.openness.csv",
'coordinates': "Data/Rerta_veg_plots.gpkg",#"Data/Rerta koordinate 2018_09_24.gpkg",
'result_data': "result_data.gpkg",
'canopy_openness_result': "canopy_openness_result.gpkg",
'buffered_points': "buffered_points.gpkg",
'GLI': "Data/Palapa June2019 GLI statistics.gpkg",
'ExG': "Data/Palapa June2019 ExG statistics.gpkg",
'DEM': "Data/Palapa June2019 DEM statistics.gpkg",
'CHM': "Data/Palapa June2019 CHM statistics.gpkg"
}
main(paths)
