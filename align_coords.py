import pandas as pd
import geopandas as gpd
import fiona
import re

''' 
Attaches geometry coordinates extracted from coordinate_extraction to the given csv files. Ideally, this would be changed as we obtain the actually geometry data from the csv files.
As each csv file is different, this function will need to be edited for each csv file, especially for those without geometry data.
'''

def canopy_openness(canopy_path, coordinates_path, destination_path, timepoint="post3"):
  ''' 
  Extracts canopy openness data from a CSV file, calculates the average openness, and filters by timepoint.
  Default timepoint is "post1", but for more recent data "post 3" should be considered.
  
  '''
  # coordinates_path = "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/result_data.gpkg"
  try:
    canopy_df_filtered = pd.read_csv(canopy_path)
    coordinates_gdf = gpd.read_file(coordinates_path)

    print(f"File loaded from: {canopy_path}")

    # Define the columns to average
    openness_cols = ['canopy.openness.to.river', 'canopy.openness.from.river', 'canopy.openness.right', 'canopy.openness.left']

    # Convert columns to numeric, coercing errors to NaN
    for col in openness_cols:
        if col in canopy_df_filtered.columns:
            canopy_df_filtered[col] = pd.to_numeric(canopy_df_filtered[col], errors='coerce')
        else:
            print(f"Warning: Column '{col}' not found in DataFrame.")

    # Calculate the average of the specified canopy openness columns
    # Only attempt to average if all columns are present
    if all(col in canopy_df_filtered.columns for col in openness_cols):
        canopy_df_filtered['average_canopy_openness'] = canopy_df_filtered[openness_cols].mean(axis=1)
    else:
        print("Error: Not all required columns for averaging were found.")
    


    # Filter by 'timepoint' after calculating the average
    if type(timepoint) is str:
        canopy_df_filtered = canopy_df_filtered[canopy_df_filtered['timepoint'].str.contains(timepoint, case=False, na=False)]
    elif type(timepoint) is int:
        canopy_df_filtered = canopy_df_filtered[canopy_df_filtered['date'].str.contains(str(timepoint), case=False, na=False)]

    print(f"Filter for timeframe {timepoint}")
    #print(canopy_df_filtered)  # Display the first few rows of the filtered DataFrame

    def standardize_names_for_canopy_openness(name):
        """
        Standardizes the 'point.label  ' column into the format 'treatment-EAST/WEST-transect-BC/OPE/OPC'
        from the GeoDataFrame by replacing specific patterns. 
        ***SPECIFICALLY FOR 3.4-canopy-openness.csv***
        """
        identifier = name.split()
        if len(identifier)>=2:
            identifier[1] = identifier[1].upper()
        else:
            identifier = name.split('-')
            identifier[1] = identifier[1].upper()
            if re.search("Opc",identifier[3], re.IGNORECASE): # Typo for OPc. 
                identifier[3] = "OPC"
        identifier[2] = re.findall(r'\d+', identifier[2])[0]  #Only keep the numeric part
        
        name = '-'.join(identifier)
        return name

    # Standardize names in the 'point.label' column
    canopy_df_filtered['point.label'] = canopy_df_filtered['point.label'].apply(standardize_names_for_canopy_openness)

    # Average across time points for a given point.label, considering the maximum value to reduce random effects of visibility
    # Maybe consider other methods of averaging if there are multiple time points?
    canopy_df_filtered = canopy_df_filtered.groupby('point.label').agg({'average_canopy_openness': 'max','treatment': 'first'}).reset_index()

    # Merge with coordinates_gdf to attach geometry
    canopy_df_filtered = canopy_df_filtered.merge(coordinates_gdf, left_on='point.label', right_on='name', how='inner')
    canopy_df_filtered = canopy_df_filtered.drop(columns='name')  # Drop 'name' columns to prevent redundancy with'point.label 

    print('Coordinates merged')
    print('Final dataframe : \n', canopy_df_filtered.head())  # Display the first few rows of the final DataFrame

    # Create a GeoDataFrame and save as a gpkg file
    merged_gdf = gpd.GeoDataFrame(canopy_df_filtered, geometry='geometry')
    merged_gdf.to_file("canopy_openness_result.gpkg", driver="GPKG")

    return merged_gdf

  except FileNotFoundError:
    print(f"Error: The file was not found at {canopy_path}")
  except Exception as e:
    print(f"An error occurred: {e}")

def frogs(frogs_path, coordinates_path, destination_path, timepoint=None):
    '''
    Extracts frog data from a CSV file, calculates the average frog count, and filters by timepoint.
    '''

    frogs_df = pd.read_csv(frogs_path)
    coordinates_gdf = gpd.read_file(coordinates_path)

    print(f"File loaded from: {frogs_path}")

    def standardize_names_for_frogs(name):
        """
        Standardizes the 'point.label  ' column into the format 'treatment-EAST/WEST-transect-BC/OPE/OPC'
        from the GeoDataFrame by replacing specific patterns. 
        ***SPECIFICALLY FOR 4.3_Frogs.csv***
        """
        identifier = name.split('-')
        if identifier[1] == 'E':
            identifier[1] = 'EAST'
            # Accounting for the mislabelled data on the east side. 
            if identifier[2] == '150':
                identifier[2] = '50'
            elif identifier[2] == '350':
                identifier[2] = '250'
        elif identifier[1] == 'W':
            identifier[1] = 'WEST'
        if len(identifier) >= 2:
            identifier[1] = identifier[1].upper()
        if len(identifier) <= 3:
            identifier.extend(i for i in identifier.pop().split(' '))
            print(identifier)
        identifier[2] = re.findall(r'\d+', identifier[2])[0]  # Only keep the numeric part

        name = '-'.join(identifier)
        return name
    
    frogs_df['Line_transect'] = frogs_df['Line_transect'].apply(standardize_names_for_frogs)

    if timepoint is not None:
        # Filter by 'timepoint' after calculating the average
        if type(timepoint) is str:
            frogs_df = frogs_df[frogs_df['timepoint'].str.contains(timepoint, case=False, na=False)]
        elif type(timepoint) is int:
            frogs_df = frogs_df[frogs_df['date'].str.contains(str(timepoint), case=False, na=False)]

    for col in ['Frog.abundance', 'Frog.richness']:
        if col in frogs_df.columns:
            frogs_df[col] = pd.to_numeric(frogs_df[col], errors='coerce')
        else:
            print(f"Warning: {col} not found in DataFrame columns.")
    frogs_df = frogs_df.groupby('Line_transect').agg({'Frog.abundance': 'mean','Frog.richness': 'mean','treatment': 'first'}).reset_index()

    print(frogs_df)  # Display the first few rows of the DataFrame

    # Merge with coordinates_gdf to attach geometry
    frogs_df = frogs_df.merge(coordinates_gdf, left_on='Line_transect', right_on='name', how='inner')
    frogs_df = frogs_df.rename(columns={'Line_transect': 'point.label'})
    frogs_df = frogs_df.drop([column for column in frogs_df.columns if column not in ['point.label', 'Frog.abundance', 'Frog.richness', 'treatment', 'geometry']], axis=1)  # Drop 'name' columns to prevent redundancy with 'point.label'

    print('Coordinates merged')
    print('Final dataframe : \n', frogs_df)  # Display the first few rows of the final DataFrame

    # Create a GeoDataFrame and save as a gpkg file
    merged_gdf = gpd.GeoDataFrame(frogs_df, geometry='geometry')
    merged_gdf.to_file(destination_path, driver="GPKG")
    print(f"Saved to {destination_path}")

    return merged_gdf

def erosion_sticks(erosion_sticks_path, coordinates_path, destination_path, timepoint=None):
    '''
    Extracts soil data from a CSV file, calculates the average of relevant soil metrics, and merges with coordinates.
    '''

    erosion_sticks_df = pd.read_csv(erosion_sticks_path)
    coordinates_gdf = gpd.read_file(coordinates_path)

    print(f"File loaded from: {erosion_sticks_path}")

    def standardize_names_for_soil(row):
        """
        Standardizes the 'point.label' column into the format 'treatment-EAST/WEST-transect-BC/OPE/OPC'
        from the GeoDataFrame by replacing specific patterns.
        ***SPECIFICALLY FOR 1.2 Erosion-sticks.csv***
        """
        identifier = [row['treatment'], row['side'].upper(), str(row['transect']), row['point'].split('.')[0]]

        if identifier[3] == 'buffer':
            identifier[3] = 'BC'
        elif identifier[3] == 'OP':
            identifier[3] = 'OPC'
        else:
            print(f"ERROR: Unknown point label format at f{identifier}")

        name = '-'.join(identifier)
        return name

    erosion_sticks_df['point.label'] = erosion_sticks_df.apply(standardize_names_for_soil, axis=1)

    # Filter by timepoint if provided
    if timepoint is not None:
        if 'timepoint' in erosion_sticks_df.columns and isinstance(timepoint, str):
            erosion_sticks_df_filtered = erosion_sticks_df[erosion_sticks_df['timepoint'].str.contains(timepoint, case=False, na=False)]
        elif 'date' in erosion_sticks_df.columns and isinstance(timepoint, int):
            erosion_sticks_df_filtered = erosion_sticks_df[erosion_sticks_df['date'].str.contains(str(timepoint), case=False, na=False)]

    # Convert relevant columns to numeric
    soil_cols = [col for col in erosion_sticks_df_filtered.columns if col in ['original.mm', 'measure.mm']]
    for col in soil_cols:
        erosion_sticks_df_filtered[col] = pd.to_numeric(erosion_sticks_df_filtered[col], errors='coerce')

    # Quantify change from data
    erosion_sticks_df_filtered['change.mm'] = erosion_sticks_df_filtered['measure.mm'] - erosion_sticks_df_filtered['original.mm']

    
    erosion_sticks_df_filtered = erosion_sticks_df_filtered.groupby(['point.label','position']).agg({'change.mm': 'mean', 'treatment': 'first'}).reset_index() # Averaging the erosion height according to their position
    erosion_sticks_df_filtered = erosion_sticks_df_filtered.pivot(index='point.label', columns='position', values='change.mm').reset_index() # Rotating to turn them into a column
    erosion_sticks_df_filtered = erosion_sticks_df_filtered.rename(columns={'Circle': 'Circle change.mm', 'Harvesting path': 'Harvesting path change.mm', 'Windrow': 'Windrow change.mm'})

    # # Aggregate soil metrics by point.label
    # agg_dict = {col: 'mean' for col in soil_cols}
    # if 'treatment' in erosion_sticks_df.columns:
    #     agg_dict['treatment'] = 'first'
    # erosion_sticks_df = erosion_sticks_df.groupby('point.label').agg(agg_dict).reset_index()

    # # Merge with coordinates
    # merged_df = erosion_sticks_df.merge(coordinates_gdf, left_on='point.label', right_on='name', how='inner')
    # merged_df = merged_df.drop(columns='name')

    # print('Coordinates merged')
    # print('Final dataframe : \n', merged_df.head())

    # # Save as GeoPackage
    # merged_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry')
    # merged_gdf.to_file(destination_path, driver="GPKG")
    # print(f"Saved to {destination_path}")

    # return merged_gdf

erosion_sticks("Data/1.2_Erosion-sticks.csv", "Data/Palapa_transects_buffer.gpkg", "Data/1.2_Erosion-sticks_aligned.gpkg", timepoint="post3")