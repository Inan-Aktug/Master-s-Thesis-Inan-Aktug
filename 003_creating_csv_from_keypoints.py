
import os
import pandas as pd
import glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import sys

start_time = time.time()

# check here which annotation file is the latest
df = pd.read_excel(r'D:\00000_master_thesis_new\0_sentence_annotation\sentences annotated\0_only_good_updated_annotations-3.xlsx')

df_master = pd.DataFrame()

# current_group = 'controls_focus_emuDB'
# current_group = 'postoperative_focus_emuDB'
# current_group = 'preoperative_focus_emuDB'
# current_group = 'RBD_focus_emuDB'



# Calculate new start and end points
df['start_point'] = (df['start_point'] / 48000) * 250
df['end_point'] = (df['end_point'] / 48000) * 250

# Handle NaN values
df['start_point'] = df['start_point'].fillna(0)
df['end_point'] = df['end_point'].fillna(0)

# Round them floor and ceil
df['start_point'] = np.floor(df['start_point']).astype(np.int64)
df['end_point'] = np.ceil(df['end_point']).astype(np.int64)
print(df)

# Path to the main directory
dir_path = r"D:\00000_master_thesis_new\all_data_trimmed"

# Initialize an empty DataFrame for the combined data
df_master = pd.DataFrame()

counter = 0


missing_directories = []

all_groups = ['controls_focus_emuDB', 'postoperative_focus_emuDB', 'preoperative_focus_emuDB', 'RBD_focus_emuDB']
start_dur = 1.55
end_dur = 3.6


for cur_group in all_groups:
    current_group = cur_group
    df_master = pd.DataFrame()

    # Loop through all the rows in your DataFrame
    for i, row in df.iterrows():
        
        if row['group'] == current_group and start_dur <= row['duration'] <= end_dur: # use 1.55 and 3.6 after

        
            # Combine parts of the path
            path_to_folder = os.path.join(dir_path, row['group'], row['patient'], row['bundle'])
            # print(path_to_folder)



            # Check if the directory exists
            if not os.path.exists(path_to_folder):
                # Log the missing directory and skip the rest of the loop
                print(f"Missing directory: {path_to_folder}")
                missing_directories.append(path_to_folder)
                continue  # Skip to the next iteration



            # Get all .csv files in the folder
            csv_files = glob.glob(path_to_folder + "/*.csv")
            # print(type(csv_files))


            # Initialize an empty DataFrame for the combined data of one row
            df_row_combined = pd.DataFrame()

            
            # Process each csv file
            for csv_file in csv_files:
                df_file = pd.read_csv(csv_file)

                # Extract the data between start_point and end_point
                df_extract = df_file.loc[row['start_point'] -1 :row['end_point'] -1] #it is shifted by one so -1 will correct for this

                # Combine the extracted data horizontally (column-wise)
                df_row_combined = pd.concat([df_row_combined, df_extract], axis=1)

            # Add 'group', 'patient', 'bundle', 'name', 'sentence' columns
            df_row_combined.insert(0, 'group', row['group'])
            df_row_combined.insert(1, 'patient', row['patient'])
            df_row_combined.insert(2, 'bundle', row['bundle'])
            df_row_combined.insert(3, 'name', row['name'])
            df_row_combined.insert(4, 'sentence', row['sentence'])
            # df_row_combined.insert(5, 'quality', row['quality'])
            # df_row_combined.insert(6, 'info', row['info'])
            df_row_combined.insert(5, 'start_point', row['start_point'])
            df_row_combined.insert(6, 'end_point', row['end_point'])
            df_row_combined.insert(7, 'duration', row['duration'])  
            

            # print(df_row_combined)


            # Append the combined data of one row to df_master
            df_master = pd.concat([df_master, df_row_combined])
            counter += 1
            print(counter)

    # Reset the index of df_master
    df_master = df_master.reset_index(drop=True)
    # print(df_master)


    df_master = df_master.loc[:, ~df_master.columns.str.startswith('Unnamed:')]
    print(df_master)
    df_master.to_excel(f'D:\\00000_master_thesis_new\\csv files\\sentences_dataset\\{current_group}_{start_dur}_{end_dur}_filtered_sentences_master_file_no_unnamed_col.xlsx')
    df_master.to_csv(f'D:\\00000_master_thesis_new\\csv files\\sentences_dataset\\{current_group}_{start_dur}_{end_dur}_filtered_sentences_master_file_no_unnamed_col.csv')


    



    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total execution time: {total_time} seconds for {current_group}")