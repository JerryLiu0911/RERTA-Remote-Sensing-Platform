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

def smart_feature_selection_pipeline(merged_df, target, all_possible_features):
    """
    Multi-stage feature selection appropriate for small datasets
    """
    print("=== SMART FEATURE SELECTION FOR SMALL DATASETS ===")
    
    n_samples = len(merged_df)
    print(f"Dataset size: {n_samples} samples")
    
    # STAGE 1: Theory-based pre-filtering
    print(f"\n📚 STAGE 1: THEORY-BASED PRE-FILTERING")
    
    # Based on canopy openness literature, these should be most relevant:
    theory_based_candidates = {
        'height_metrics': [col for col in all_possible_features if 'CHM' in col and any(stat in col for stat in ['mean', 'median', 'max'])],
        'variability_metrics': [col for col in all_possible_features if 'CHM' in col and any(stat in col for stat in ['std', 'range', 'cv'])],
        'greenness_metrics': [col for col in all_possible_features if any(index in col for index in ['ExG', 'NDVI', 'GLI'])],
        'texture_metrics': [col for col in all_possible_features if any(tex in col for tex in ['contrast', 'entropy', 'homogeneity'])]
    }
    
    # Select BEST representative from each category
    stage1_features = []
    for category, candidates in theory_based_candidates.items():
        if candidates:
            print(f"  {category}: {len(candidates)} candidates → selecting best 1-2")
            
            # Calculate target correlations for candidates in this category
            category_corrs = []
            for feat in candidates:
                if feat in merged_df.columns:
                    corr = abs(merged_df[target].corr(merged_df[feat]))
                    category_corrs.append((feat, corr))
            
            # Take top 1-2 from each category
            category_corrs.sort(key=lambda x: x[1], reverse=True)
            selected_from_category = [feat for feat, _ in category_corrs[:2]]
            stage1_features.extend(selected_from_category)
            
            for feat, corr in category_corrs[:2]:
                print(f"    ✅ {feat}: {corr:.3f}")
    
    print(f"  Stage 1 result: {len(stage1_features)} features")
    
    # STAGE 2: Correlation-based refinement
    print(f"\n📊 STAGE 2: CORRELATION-BASED REFINEMENT")
    
    # Remove highly correlated features within our selected set
    if len(stage1_features) > 1:
        feature_corr_matrix = merged_df[stage1_features].corr()
        
        # Find and remove redundant features
        to_remove = set()
        for i, feat1 in enumerate(stage1_features):
            for j, feat2 in enumerate(stage1_features[i+1:], i+1):
                if feat1 not in to_remove and feat2 not in to_remove:
                    corr = abs(feature_corr_matrix.iloc[i, j])
                    if corr > 0.85:  # High correlation threshold
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
    
    # STAGE 3: Sample size validation
    print(f"\n⚖️ STAGE 3: SAMPLE SIZE VALIDATION")
    
    ratio = n_samples / len(stage2_features) if stage2_features else 0
    print(f"  Sample-to-feature ratio: {ratio:.1f}:1")
    
    if ratio < 5:
        print(f"  🚨 Still too many features for dataset size!")
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
            print(f"    ✅ {feat}: {corr:.3f}")
    else:
        stage3_features = stage2_features
        print(f"  ✅ Feature count appropriate for dataset size")
    
    print(f"\n🎯 FINAL RESULT: {len(stage3_features)} high-quality features")
    print(f"   Features: {stage3_features}")
    print(f"   Final ratio: {n_samples/len(stage3_features):.1f}:1")
    
    return stage3_features

def main(paths):

    ### Extracting and aligning coordinates ###

    # coordinate_extraction.extract_corner_coords(paths['coordinates'], paths['result_data'])
    # align_coords.canopy_openness(paths['canopy_openness'], paths['result_data'], paths['canopy_openness_result'], timepoint=2019)
    # zonal_gdf, figures = gis.zonal_statistics(gpkg_path=paths['canopy_openness_result'],
    #                     raster_path="G:/My Drive/UROP/UROP Rerta Palapa June2019 CHM.tif",
    #                     output_buffer_path=paths['buffered_points'], 
    #                     filtering_logic=gis.clip_below_zero,
    #                     output_zonal_gpkg=paths['CHM'],
    #                     buffer_geom_path=paths['result_data'],
    #                     save_plots=True)


    # zonal_gdf, figures = gis.zonal_statistics(gpkg_path=paths['canopy_openness_result'],
    #                     raster_path="G:/My Drive/UROP/UROP RERTA Palapa June2019 ExG.tif",
    #                     output_buffer_path=paths['buffered_points'], 
    #                     #filtering_logic=gis.clip_below_zero,
    #                     output_zonal_gpkg=paths['ExG'],
    #                     buffer_geom_path=paths['result_data'])


    # gis.raster_calculation_zonal(gpkg_path=paths['canopy_openness_result'],
    #                               raster_path="G:/My Drive/UROP/imagery/rerta_palapa_ortho20190501.tif",
    #                               output_buffer_path=paths['buffered_points'],
    #                               output_zonal_gpkg=paths['GLI'],
    #                               buffer_geom_path=paths['result_data'],
    #                               calculation_func=gis.create_calculation_functions()['Green_Leaf_Index'],
    #                               calculation_name='Green_Leaf_Index')


    ### Combining and analyzing data into dataframes ###

    merged_df = statistical_modelling.load_data(
            [#('GLI', paths['GLI'])
            ('ExG', paths['ExG']),
            #('DEM', paths['DEM']),
            ('CHM', paths['CHM'])], filter = "OPE|BC")

    print(f"Finished merging columns : {merged_df.columns}")

    # BC_df = merged_df[merged_df['point.label'].str.contains("BC", case=False, na=False)]
    # print(BC_df.head())
    # plot_relations(BC_df,'average_canopy_openness', 'mean_CHM', geo = False)


    features = [column for column in merged_df.columns if column not in ['geometry', 'point.label', 'treatment']]
    feature_diagnostics(merged_df, 'average_canopy_openness', features)
    print("\n \n \n \n \n \n \n")
    analyze_chm_correlations(merged_df, features)
    selected_features = smart_feature_selection_pipeline(merged_df, 'average_canopy_openness', features)



    ### Statistical Modelling ###

    #statistical_modelling.random_forest_regression(merged_df, 'average_canopy_openness', ['canopy_openness_CHM', 'mean_ExG', 'mean_CHM'])
    # statistical_modelling.random_forest_ensemble(merged_df, 'average_canopy_openness', [feature for feature in merged_df.columns if feature not in ['geometry', 'point.label', 'average_canopy_openness']])
    # statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if'CHM' in column and column != 'geometry_CHM' and column != 'name_CHM'], display=False)
    # statistical_modelling.multi_linear_regression_display(merged_df, 'average_canopy_openness', [column for column in merged_df.columns if column not in ['geometry', 'point.label', 'average_canopy_openness']], display=False)


paths = {
'canopy_openness': "Data/3.4-canopy.openness.csv",
'frog_biodiversity': "Data/3.4-frog.biodiversity.csv",
'coordinates': "Data/Palapa_veg_plots.gpkg", #"Data/Rerta koordinate 2018_09_24.gpkg",
'abcd_coordinates': "Data/Palapa_ABCD.gpkg", 
'result_data': "result_data.gpkg",
'canopy_openness_result': "canopy_openness_result.gpkg",
'buffered_points': "buffered_points.gpkg",
'GLI': "Data/Palapa June2019 GLI statistics.gpkg",
'ExG': "Data/Palapa June2019 ExG statistics.gpkg",
'DEM': "Data/Palapa June2019 DEM statistics.gpkg",
'CHM': "Data/Palapa June2019 CHM statistics.gpkg"
}
main(paths)