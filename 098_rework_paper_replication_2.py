### information
# reads the json all information file
# extracts information from c1_sampleStart to v1_sampleStart + v1_sampleDur
# name_sampleStart is the same as c1_sampleStart
# from there, extracts information from the sensory ttip, tbo, llip, ulip - all values

import pandas as pd
import os
from math import floor, ceil
import time
import sys

start_time = time.time()

# Read json_all_information.csv (Adjust the path accordingly)
json_df = pd.read_csv(r'D:\00000_master_thesis_new\csv files\json annotation\first_datapoint_filter_to_2083_datapoints.csv')

# Constants for calculations
sr = 48000
csv_conversion = 250

# Initialize counter for skipped rows
skipped_rows_counter = 0

# List to collect individual DataFrames generated in each iteration
dfs_to_concat = []

current_group = None # for tracking / script feedback

# Iterate over each row in json_df
for idx, row in json_df.iterrows():
    # Extracting relevant columns
    group = row['group']
    patient = row['patient']
    bundle = row['bundle']
    
    if row['group'] != current_group:
        current_group = row['group']
        print(f'working on group: {current_group}')

    # Check for NaN values in the relevant columns
    if pd.isna(row['name_sampleStart']) or pd.isna(row['V1_sampleStart']) or pd.isna(row['V1_sampleDur']):
        skipped_rows_counter += 1
        continue
    
    # Perform calculations for start and end index
    start = floor((row['name_sampleStart'] / sr) * csv_conversion) - 1
    end = ceil(((row['V1_sampleStart']) / sr) * csv_conversion)
    time_in_ms = (row['V1_sampleStart'] - row['name_sampleStart']) / sr * 1000 ###################### adding time in ms
    #print(time_in_ms)
    
    # print(start, end)
    # Placeholder DataFrame to store the extracted rows for this particular patient and bundle
    extracted_df = pd.DataFrame()
    
    # Construct the folder path
    folder_path = os.path.join('D:\\00000_master_thesis_new\\all_data', group, patient, bundle)
    
    # Iterate through the 4 CSV files in the folder
    for filename in os.listdir(folder_path):
        if any(x in filename for x in ['llip', 'tbo', 'ttip', 'ulip']) and filename.endswith('.csv'):
            # Read the CSV file
            csv_path = os.path.join(folder_path, filename)
            csv_df = pd.read_csv(csv_path)
            


            # Extract the rows based on calculated start and end index
            extracted_rows = csv_df.iloc[int(start):int(end)]
            
            # Reset index for horizontal concatenation
            extracted_rows.reset_index(drop=True, inplace=True)
            
            # Concatenate the extracted rows horizontally
            extracted_df = pd.concat([extracted_df, extracted_rows], axis=1)

        # After extracting and concatenating the rows, add the Group, Patient, Bundle, and Name information
    extracted_df['group'] = row['group']
    extracted_df['patient'] = row['patient']
    extracted_df['bundle'] = row['bundle']
    extracted_df['name'] = row['name']
    extracted_df['group2'] = row['group2']
    extracted_df['time_in_ms'] = time_in_ms
    
    # Remove 'Unnamed: 0' column if it exists
    extracted_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)


    # Add the extracted DataFrame to the list
    dfs_to_concat.append(extracted_df)

# Perform a single concatenation operation to combine these DataFrames into final_df
final_df = pd.concat(dfs_to_concat, axis=0)

# Rearrange columns to have 'group', 'patient', 'bundle', and 'name' as the first four columns
cols = ['group', 'group2', 'patient', 'bundle', 'name'] + [col for col in final_df.columns if col not in ['group', 'group2', 'patient', 'bundle', 'name']]
final_df = final_df[cols]

# Specify the output folder (Adjust the path accordingly)
output_folder = 'D:\\00000_master_thesis_new\\csv files\\c1_to_v1_paper_replication_dataset\\c1_to_v1'

# Save the final DataFrame to CSV and Excel formats
# final_df.to_csv(os.path.join(output_folder, 'c1_v1_filtered_datapoints_paper_replication_keypoint_values_first_syllable.csv'), index=False)
# final_df.to_excel(os.path.join(output_folder, 'c1_v1_filtered_datapoints_paper_replication_keypoint_values_first_syllable.xlsx'), index=False)

# Display the number of skipped rows
print(f"Number of skipped rows due to NaN values: {skipped_rows_counter}")



print("--- %s seconds ---" % (time.time() - start_time))