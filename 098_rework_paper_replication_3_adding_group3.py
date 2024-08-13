import pandas as pd
import os
from math import floor, ceil
import time
import sys

start_time = time.time()

# Read json_all_information.csv (Adjust the path accordingly)
json_df = pd.read_csv(r'D:\00000_master_thesis_new\csv files\json annotation\first_datapoint_filter_to_2083_datapoints.csv')

# Read the new CSV file
updrs_df = pd.read_csv(r'D:\00000_master_thesis_new\csv files\group3 data\filtered_patient_data_UPDRS_25.csv')

# Create a dictionary mapping patient names to their Total_score and Group for group3 data
patient_info = {patient[:6]: data for patient, data in updrs_df.set_index('Patient')[['Total_score', 'Group']].to_dict('index').items()}
print(patient_info)
for patient, data in patient_info.items():
    print(patient, data)



print(json_df['patient'].unique())
print(updrs_df['Patient'].unique())


# sys.exit()
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
    end = ceil(((row['V1_sampleStart'] + row['V1_sampleDur']) / sr) * csv_conversion)
    time_in_ms = (row['V1_sampleStart'] - row['name_sampleStart']) / sr * 1000
    
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
    
    # Add Total_score and Group from the new CSV if available
    patient_key = patient[:6]  # Take first 6 characters of patient name
    if patient_key in patient_info:
        extracted_df['Total_score'] = patient_info[patient_key]['Total_score']
        extracted_df['group3'] = patient_info[patient_key]['Group']
    else:
        extracted_df['Total_score'] = None
        extracted_df['group3'] = None
    
    # Remove 'Unnamed: 0' column if it exists
    extracted_df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

    # Add the extracted DataFrame to the list
    dfs_to_concat.append(extracted_df)

# Perform a single concatenation operation to combine these DataFrames into final_df
final_df = pd.concat(dfs_to_concat, axis=0)

# Rearrange columns in the desired order
cols = ['group', 'patient', 'bundle', 'group2', 'group3', 'Total_score'] + [col for col in final_df.columns if col not in ['group', 'patient', 'bundle', 'group2', 'group3', 'Total_score']]
final_df = final_df[cols]

# Specify the output folder (Adjust the path accordingly)
output_folder = 'D:\\00000_master_thesis_new\\csv files\\group3 data\\paper_replication_group3_filter_updrs_25'

# Save the final DataFrame to CSV and Excel formats
final_df.to_csv(os.path.join(output_folder, 'c1_v1_plus_duration_group3.csv'), index=False)
final_df.to_excel(os.path.join(output_folder, 'c1_v1_plus_duration_group3.xlsx'), index=False)

# Display the number of skipped rows
print(f"Number of skipped rows due to NaN values: {skipped_rows_counter}")

print("--- %s seconds ---" % (time.time() - start_time))