####################################################################################################
# Rework of the original script
# this script contains:
# - centering of the data
# - min max normalization per patient 
# - exporting the results to csv and excel files
####################################################################################################

import statistic_modules as sm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import random
import sys
import os




features = ['llip_x', 'llip_y', 'tbo_x', 'tbo_y', 'ttip_x', 'ttip_y', 'ulip_x', 'ulip_y']

path = r'D:\00000_master_thesis_new\csv files\c1_to_v1_paper_replication_dataset\c1_to_v1_plus_duration'
path_to_save = r'D:\00000_master_thesis_new\csv files\c1_to_v1_paper_replication_dataset\\'


# centered_dir = os.path.join(path_to_save + '_centered')
# centered_and_min_max_per_patient_dir = os.path.join(path_to_save + '_centered_and_min_max_per_patient')
# centered_and_min_max_per_group_dir = os.path.join(path_to_save + '_centered_and_min_max_per_group')
centered_and_min_max_per_sample_dir = os.path.join(path_to_save + '_centered_and_min_max_per_sample')


# os.makedirs(centered_dir, exist_ok=True)
# os.makedirs(centered_and_min_max_per_patient_dir, exist_ok=True)
# os.makedirs(centered_and_min_max_per_group_dir, exist_ok=True)
os.makedirs(centered_and_min_max_per_sample_dir, exist_ok=True)

processed_dfs = [] # to store the processed dataframes

for i in os.listdir(path):
    if i.endswith('.csv') and not i.startswith('all_groups'):
        current_df = pd.read_csv(path + '\\' + i)
        current_df = current_df.rename(columns={'session': 'patient'}) 
        new_filename = i[:-4] + "_centered"
        print(new_filename)
        # print(i)
        
        ### calling the function to center the data
        df_centered = sm.center_features(current_df, features)
        # print(i[-1])
        # sys.exit()
        

        # csv_path = os.path.join(centered_dir, new_filename + '.csv')
        # df_centered.to_csv(csv_path, index=False)
        
        # excel_path = os.path.join(centered_dir, new_filename + '.xlsx')
        # df_centered.to_excel(excel_path, index=False)

        # ## calling the function to min max normalize the data after centering
        # df_centered_min_max_per_patient = sm.minmax_scale_features_by_patient(df_centered, features)
        # new_filename = i[:-4] + "_centered_min_max_per_patient"

        # calling the function to min max normalize the data after centering for each sample
        df_centered_min_max_per_sample = sm.minmax_scale_features_per_sample(df_centered, features)
        new_filename = i[:-4] + "_centered_min_max_per_sample_bundle"

        # csv_path = os.path.join(centered_and_min_max_per_patient, new_filename + '.csv')
        # df_centered_min_max_per_patient.to_csv(csv_path, index=False)

        # excel_path = os.path.join(centered_and_min_max_per_patient, new_filename + '.xlsx')
        # df_centered_min_max_per_patient.to_excel(excel_path, index=False)

        





        # df_centered_min_max_per_group = sm.minmax_scale_features_by_group(df_centered, features)
        # new_filename = i[:-4] + "_centered_min_max_per_group"


        
        csv_path = os.path.join(centered_and_min_max_per_sample_dir, new_filename + '.csv')
        df_centered_min_max_per_sample.to_csv(csv_path, index=False)

        excel_path = os.path.join(centered_and_min_max_per_sample_dir, new_filename + '.xlsx')
        df_centered_min_max_per_sample.to_excel(excel_path, index=False)


        processed_dfs.append(df_centered_min_max_per_sample)


# After the loop, merge all processed dataframes
merged_df = pd.concat(processed_dfs, ignore_index=True)

# Save the merged dataframe
merged_csv_path = os.path.join(centered_and_min_max_per_sample_dir, "all_groups_centered_min_max_per_sample_bundle.csv")
merged_df.to_csv(merged_csv_path, index=False)

merged_excel_path = os.path.join(centered_and_min_max_per_sample_dir, "all_groups_centered_min_max_per_sample_bundle.xlsx")
merged_df.to_excel(merged_excel_path, index=False)

print("All files processed and merged successfully.")