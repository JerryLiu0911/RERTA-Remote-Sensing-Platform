import pandas as pd
import geopandas as gpd
import fiona
import re

''' 
Attaches geometry coordinates extracted from coordinate_extraction to the given csv files. Ideally, this would be changed as we obtain the actually geometry data from the csv files.
As each csv file is different, this function will need to be edited for each csv file, especially for those without geometry data.
'''

def load_canopy_openness(canopy_path, timepoint= None):
  ''' 
  Extracts canopy openness data from a CSV file, calculates the average openness, and filters by timepoint.
  Default timepoint is "post1", but for more recent data "post 3" should be considered.
  
  '''
  # coordinates_path = "G:/My Drive/UROP/UROP RERTA Remote Sensing Platform/RERTA-Remote-Sensing-Platform/result_data.gpkg"
  try:
    canopy_df_filtered = pd.read_csv(canopy_path)
    # coordinates_gdf = gpd.read_file(coordinates_path)

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

    # # Merge with coordinates_gdf to attach geometry
    # canopy_df_filtered = canopy_df_filtered.merge(coordinates_gdf, left_on='point.label', right_on='name', how='inner')
    # canopy_df_filtered = canopy_df_filtered.drop(columns='name')  # Drop 'name' columns to prevent redundancy with'point.label 

    # print('Coordinates merged')
    # print('Final dataframe : \n', canopy_df_filtered.head())  # Display the first few rows of the final DataFrame

    # # Create a GeoDataFrame and save as a gpkg file
    # merged_gdf = gpd.GeoDataFrame(canopy_df_filtered, geometry='geometry')
    # merged_gdf.to_file("canopy_openness_result.gpkg", driver="GPKG")

    # return merged_gdf
    print("Filtered canopy_openness dataframe :\n", canopy_df_filtered.head())
    return canopy_df_filtered

  except FileNotFoundError:
    print(f"Error: The file was not found at {canopy_path}")
  except Exception as e:
    print(f"An error occurred: {e}")

def load_frogs(frogs_path, timepoint=None):
    '''
    Extracts frog data from a CSV file, calculates the average frog count, and filters by timepoint.
    '''

    try:
        frogs_df = pd.read_csv(frogs_path)

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
        frogs_df = frogs_df.rename(columns={'Line_transect': 'point.label'})
        frogs_df = frogs_df.drop([column for column in frogs_df.columns if column not in ['point.label', 'Frog.abundance', 'Frog.richness', 'treatment']], axis=1)  # Drop 'name' columns to prevent redundancy with 'point.label'

        print('Filtered frogs dataframe : \n', frogs_df.head())  # Display the first few rows of the final DataFrame
        return frogs_df
    
    except FileNotFoundError:
        print(f"Error: The file was not found at {frogs_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_erosion_sticks(erosion_sticks_path, timepoint=None):
    '''
    Extracts soil data from a CSV file, calculates the average of relevant soil metrics, and merges with coordinates.
    '''
    
    try:

        erosion_sticks_df = pd.read_csv(erosion_sticks_path)

        print(f"File loaded from: {erosion_sticks_path}")

        def standardize_names_for_soil(row):
            """
            Standardizes the 'point.label' column into the format 'treatment-EAST/WEST-transect-BC/OPE/OPC'
            from the GeoDataFrame by replacing specific patterns.
            ***SPECIFICALLY FOR 1.2 Erosion-sticks.csv***
            """
            identifier = [row['treatment'], row['side'].upper(), str(row['transect']), row['point'].split('.')[0]]

            # Only 2 sample sites: buffer and OP
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

        print("Filtered erosion_sticks dataframe :\n", erosion_sticks_df_filtered.head())  # Display the first few rows of the DataFrame
        return erosion_sticks_df_filtered

    except FileNotFoundError:
        print(f"Error: The file was not found at {erosion_sticks_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def load_seed_removal(seed_removal_path, timepoint = None):
    """
    Loads the seed removal data from the specified paths.

    Args:
        seed_removal_paths (dict): The paths to the seed removal data files.
        timepoint (str, optional): The timepoint for filtering the data.

    Returns:
        pd.DataFrame: The loaded seed removal data.
    """
    try:

        seed_removal_df = pd.read_csv(seed_removal_path)

        print(f"File loaded from: {seed_removal_path}")

        def standardize_names_for_seed(row):
            """
            Standardizes the 'point.label' column into the format 'treatment-EAST/WEST-transect-BC/OPE/OPC'
            from the GeoDataFrame by replacing specific patterns.
            ***SPECIFICALLY FOR 1.2 Erosion-sticks.csv***
            """
            identifier = [row['treatment'], row['side'].upper(), re.findall(r'\d+', row['transect'])[0], row['point']]

            if identifier[3] == 'buffer.core':
                identifier[3] = 'BC'
            elif identifier[3] == 'OP.core':
                identifier[3] = 'OPC'
            elif identifier[3] == 'OP.edge':
                identifier[3] = 'OPE'
            else:
                print(f"ERROR: Unknown point label format at f{identifier}")

            name = '-'.join(identifier)
            return name

        seed_removal_df['point.label'] = seed_removal_df.apply(standardize_names_for_seed, axis=1)

        # Filter by timepoint if provided
        if timepoint is not None:
            if 'timepoint' in seed_removal_df.columns and isinstance(timepoint, str):
                seed_df_filtered = seed_removal_df[seed_removal_df['timepoint'].str.contains(timepoint, case=False, na=False)]
            elif 'date' in seed_removal_df.columns and isinstance(timepoint, int):
                seed_df_filtered = seed_removal_df[seed_removal_df['date'].str.contains(str(timepoint), case=False, na=False)]

        # Convert relevant columns to numeric and quantify data
        seed_cols = [col for col in seed_df_filtered.columns if col in ['seeds.remaining.plate.1', 'seeds.remaining.plate.2','seeds.remaining.plate.3']]
        for col in seed_cols:
            seed_df_filtered[col] = pd.to_numeric(seed_df_filtered[col], errors='coerce')
            seed_df_filtered[col] = seed_df_filtered[col] / 10 # According to protocol, initially all plates had 10 seeds. Transform data to proportions (might need to arcsin sqrt transform later)
        
        if all(col in seed_df_filtered.columns for col in seed_cols):
            seed_df_filtered['average_canopy_openness'] = seed_df_filtered[seed_cols].mean(axis=1)

        print("Filtered seed dataframe :\n", seed_df_filtered.head())

        agg_dict = {f'{col}.proportions': 'mean' for col in seed_cols}
        agg_dict.update({'treatment': 'first'})
        seed_df_filtered = seed_df_filtered.groupby('point.label').agg(agg_dict).reset_index() # Averaging the erosion height according to their position

        print("Filtered seed dataframe :\n", seed_df_filtered.head())  # Display the first few rows of the DataFrame
        return seed_df_filtered

    except FileNotFoundError:
        print(f"Error: The file was not found at {seed_removal_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def align_coords(dataframes, coordinates_gdf_path, destination_path):
    """
    Aligns the data in df with the provided coordinates GeoDataFrame.

    Args:
        dataframes (list(pd.DataFrame)): The erosion sticks data to align.
        coordinates_gdf (string): The path to the GeoDataFrame containing the coordinates.

    Returns:
        result_gdf (gpd.GeoDataFrame): The aligned erosion sticks data.
    """
    try:

        coordinates_gdf = gpd.read_file(coordinates_gdf_path)
        merged_df = coordinates_gdf
        if 'name' not in coordinates_gdf.columns:
            print(f"Error: 'name' column not found in coordinates GeoDataFrame")
            return None
        else:
            merged_df = merged_df.rename(columns={'name': 'point.label'})# Drop 'name' columns to prevent redundancy with 'point.label'

        for i, df in enumerate(dataframes):
            # Perform spatial join or coordinate alignment here
            # This is a placeholder for the actual alignment logic
            if 'point.label' not in df.columns:
                print(f"Error: 'point.label' column not found in DataFrame {i}")
                continue

            merged_df = merged_df.merge(df, on='point.label', how='left')

        print('Coordinates merged')
        print('Final dataframe : \n', merged_df[:10])  # Display the first few rows of the final DataFrame

        # Create a GeoDataFrame and save as a gpkg file
        merged_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry')
        merged_gdf.to_file(destination_path, driver="GPKG")
        print(f"Saved to {destination_path}")

        return merged_df
    
    except FileNotFoundError:
        print(f"Error: The file was not found at {coordinates_gdf_path}")
    except Exception as e:
        print(f"An error occurred during alignment: {e}")
        return None

load_seed_removal("Data/6.5_Seed-removal.csv", timepoint="post2")
# load_erosion_sticks("Data/1.2_Erosion-sticks.csv", "Data/1.2_Erosion-sticks_aligned.gpkg", timepoint="post3")
# align_coords([load_canopy_openness("Data/3.4-canopy.openness.csv", "Data/result_data.gpkg", "Data/canopy_openness_result.gpkg", timepoint="post3"), load_erosion_sticks("Data/1.2_Erosion-sticks.csv", "Data/1.2_Erosion-sticks_aligned.gpkg", timepoint="post3")], "Data/Palapa_veg_plots_corners.gpkg")
# align_coords([load_frogs("Data/4.3_Frogs.csv", timepoint="post3")], "Data/Palapa_transects_buffer.gpkg", "Data/Palapa_transects_buffer_results.gpkg")