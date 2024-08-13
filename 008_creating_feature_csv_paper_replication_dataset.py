'''
IMPORTANT NOTE:
THIS SCRIPT IS MODIFIED FOR THE PAPER REPLICATION DATASET
- I extracted the features from the paper replication dataset
- The features are the same as for the names and sentences dataset
- This script also centers and scales the original paper replication dataset and exports it to:
D:\00000_master_thesis_new\csv files\c1_to_v1_paper_replication_dataset

'''



import numpy as np
import pandas as pd
import random_forest_features as rff
import os
import sys
import openpyxl


# load the data


###names dataset
# output_path = r'D:\00000_master_thesis_new\csv files\rf_features\names_dataset'

# input_folder = r'D:\00000_master_thesis_new\csv files\names_dataset'

### group3 dataset
output_path = r'D:\00000_master_thesis_new\csv files\rf_features\paper_replication_dataset\features_like_other_datasets'

input_folder = r'D:\00000_master_thesis_new\csv files\c1_to_v1_paper_replication_dataset\c1_to_v1_plus_duration'
input_file = r'c1_v1_plus_duration_filtered_datapoints_paper_replication_keypoint_values_first_syllable.csv'


# sentences_annotation = r'D:\00000_master_thesis_new\annotations from json files\sentences_dataset_all_annotations.csv'

####################################################################################################
# Centering of the data
def center_features(df, features):
    '''
    Center the specified features around 0 within each group defined by 'group', 'patient', and 'bundle'.

    Args:
        df (pd.DataFrame): The input DataFrame containing the features to be centered.
        features (List[str]): A list of feature names that need to be centered.

    Returns:
        pd.DataFrame: A new DataFrame with the specified features centered around 0 within each group.
    '''
    # Create a copy of the input DataFrame
    df_centered = df.copy()
    
    # Group by 'group', 'patient', and 'bundle'
    grouped = df_centered.groupby(['group2', 'patient', 'bundle'])
    
    # Iterate over each feature
    for feature in features:
        # Calculate the mean of the feature within each group
        feature_mean = grouped[feature].transform('mean')
        
        # Subtract the mean from all values to center them around 0
        df_centered[feature] = df_centered[feature] - feature_mean
    
    return df_centered
# centerin of the data - end
####################################################################################################



####################################################################################################
# min-max scale features for each patient, regardless of the group
def minmax_scale_features_by_patient(df, features): ### change this to minmax_scale_features_by_patient
    '''
    Scale the specified features within each group and patient using Min-Max scaling.

    Args:
        df (pd.DataFrame): The input DataFrame containing the features to be scaled.
        features (List[str]): A list of feature names that need to be scaled.

    Returns:
        pd.DataFrame: A new DataFrame with the specified features scaled within each group and patient.
    '''
    # Create a copy of the input DataFrame
    df_scaled = df.copy()

    # Group by 'group' and 'patient'
    grouped = df_scaled.groupby(['group2', 'patient'])

    # Iterate over each feature
    for feature in features:
        # Calculate the minimum and maximum values of the feature within each group and patient
        feature_min = grouped[feature].transform('min')
        feature_max = grouped[feature].transform('max')

        # Scale the feature within each group and patient using Min-Max scaling
        df_scaled[feature] = (df_scaled[feature] - feature_min) / (feature_max - feature_min)

    return df_scaled

# min-max scale features for each patient, regardless of the group - end
####################################################################################################




df = pd.read_csv(os.path.join(input_folder, input_file))
df.drop(columns=['time_in_ms'], inplace=True)


keywords = ['llip_x', 'llip_y',
            'ulip_x', 'ulip_y',
            'tbo_x', 'tbo_y',
            'ttip_x', 'ttip_y']

all_vel_col = [
    'llip_xvel', 'llip_yvel', 'llip_tvel',
    'tbo_xvel', 'tbo_yvel', 'tbo_tvel',
    'ulip_xvel', 'ulip_yvel', 'ulip_tvel',
    'ttip_xvel', 'ttip_yvel', 'ttip_tvel'
]

all_acc_col = [
    'llip_xacc', 'llip_yacc', 'llip_tacc',
    'tbo_xacc', 'tbo_yacc', 'tbo_tacc',
    'ulip_xacc', 'ulip_yacc', 'ulip_tacc',
    'ttip_xacc', 'ttip_yacc', 'ttip_tacc'
]

df_centered = center_features(df, keywords)
df_centered_and_scaled = minmax_scale_features_by_patient(df_centered, keywords)

df = df_centered_and_scaled
print(df)
print(df.columns)
print(df.shape)


print(df_centered)
print(df_centered_and_scaled)

centered_filename = 'centered_c1_to_v1_plus_duration'
centered_and_min_max_filename = 'centered_and_min_max_per_patient_c1_to_v1_plus_duration'



##### Exporting the centered and min-max scaled data for paper replication dataset ####
# df_centered.to_csv(os.path.join(r'D:\00000_master_thesis_new\csv files\c1_to_v1_paper_replication_dataset\centered_c1_to_v1_plus_duration', centered_filename + '.csv'), index=False)
# df_centered.to_excel(os.path.join(r'D:\00000_master_thesis_new\csv files\c1_to_v1_paper_replication_dataset\centered_c1_to_v1_plus_duration', centered_filename + '.xlsx'), index=False)


# df_centered_and_scaled.to_csv(os.path.join(r'D:\00000_master_thesis_new\csv files\c1_to_v1_paper_replication_dataset\centered_and_min_max_per_patient_c1_to_v1_plus_duration', centered_and_min_max_filename + '.csv'), index=False)
# df_centered_and_scaled.to_excel(os.path.join(r'D:\00000_master_thesis_new\csv files\c1_to_v1_paper_replication_dataset\centered_and_min_max_per_patient_c1_to_v1_plus_duration', centered_and_min_max_filename + '.xlsx'), index=False) 
# sys.exit()



# Run your functions on the DataFrame
df_peak_vel_pos = rff.peak_velocity_positive(df, all_vel_col)
df_peak_vel_neg = rff.peak_velocity_negative(df, all_vel_col)
df_avg_vel = rff.average_velocity(df, all_vel_col)
df_mean_speed = rff.mean_speed(df, all_vel_col)
df_max_speed = rff.max_speed(df, all_vel_col)
df_peak_acc_pos = rff.peak_acceleration_positive(df, all_acc_col)
df_peak_acc_neg = rff.peak_acceleration_negative(df, all_acc_col)
df_avg_acc = rff.average_acceleration(df, all_acc_col)
df_range = rff.range_feature(df, keywords)
df_total_length = rff.total_length(df, 'tbo_y')
df_time_dip = rff.time_till_dip(df, keywords)
df_time_peak = rff.time_till_peak(df, keywords)
df_turning_points = rff.calculate_turning_points(df, keywords)

# Merge the results into a single DataFrame
df_rf_features = pd.merge(df_peak_vel_pos, df_peak_vel_neg, on=['group', 'patient', 'bundle'])
df_rf_features = pd.merge(df_rf_features, df_avg_vel, on=['group', 'patient', 'bundle'])
df_rf_features = pd.merge(df_rf_features, df_mean_speed, on=['group', 'patient', 'bundle'])
df_rf_features = pd.merge(df_rf_features, df_max_speed, on=['group', 'patient', 'bundle'])
df_rf_features = pd.merge(df_rf_features, df_peak_acc_pos, on=['group', 'patient', 'bundle'])
df_rf_features = pd.merge(df_rf_features, df_peak_acc_neg, on=['group', 'patient', 'bundle'])
df_rf_features = pd.merge(df_rf_features, df_avg_acc, on=['group', 'patient', 'bundle'])
df_rf_features = pd.merge(df_rf_features, df_range, on=['group', 'patient', 'bundle'])
df_rf_features = pd.merge(df_rf_features, df_total_length, on=['group', 'patient', 'bundle'])
df_rf_features = pd.merge(df_rf_features, df_time_dip, on=['group', 'patient', 'bundle'])
df_rf_features = pd.merge(df_rf_features, df_time_peak, on=['group', 'patient', 'bundle'])
df_rf_features = pd.merge(df_rf_features, df_turning_points, on=['group', 'patient', 'bundle'])


print(df_rf_features)
print(df_rf_features.columns)
print(df_rf_features.shape) 
print(df.columns)
df_group2_info = df[['group', 'patient', 'bundle', 'group2']]


print(df_group2_info)
# Merge df_rf_features with df based on 'group', 'patient', and 'bundle' columns
df_rf_features = pd.merge(df_rf_features, df[['group', 'patient', 'bundle', 'group2']].drop_duplicates(),
                          on=['group', 'patient', 'bundle'], how='left')


column_order = ['group', 'group2', 'patient', 'bundle']
df_rf_features = df_rf_features[column_order + [col for col in df_rf_features.columns if col not in column_order]]
print(df_rf_features)

print(df_rf_features.columns)




# Save the merged DataFrame to the CSV file
output_file_csv = os.path.join(output_path, input_file.replace('.csv', '_centered_min_max_per_patient_rf_features.csv'))
output_file_excel = os.path.join(output_path, input_file.replace('.csv', '_centered_min_max_per_patient_rf_features.xlsx'))


df_rf_features.to_csv(output_file_csv, index=False)
df_rf_features.to_excel(output_file_excel, index=False)


# # Create the output folder if it doesn't exist
# os.makedirs(output_path, exist_ok=True)

# constant_columns_df = None

# for folder_name in os.listdir(input_folder):
#     # if folder_name.startswith("centered"):
#     if "centered" in folder_name:
#         folder_full_path = os.path.join(input_folder, folder_name)
#         if os.path.isdir(folder_full_path):
#             # Find the largest CSV file starting with "all_groups"
#             largest_csv_file = None
#             largest_csv_size = 0
            
#             for file_name in os.listdir(folder_full_path):
#                 if file_name.startswith("all_groups") and file_name.endswith(".csv"):
#                     file_full_path = os.path.join(folder_full_path, file_name)
#                     file_size = os.path.getsize(file_full_path)
                    
#                     if file_size > largest_csv_size:
#                         largest_csv_file = file_full_path
#                         largest_csv_size = file_size
             
#             if largest_csv_file:
#                 # Read the largest CSV file using pandas
#                 df = pd.read_csv(largest_csv_file)

#                 # print(df.shape)
#                 # print(df_sentences_annotation.shape)
#                 # create list of column names from df
#                 df_columns = df.columns
                
#                 # Create list of column names from df
#                 df_columns = df.columns.tolist()
#                 print(df_columns)
#                 # sys.exit()
#                 # Identify missing columns from df_sentences_annotation
#                 missing_columns = [col for col in annotation_filter_list if col not in df_columns]

#                 print(missing_columns)
#                 # sys.exit()
#                 # If there are missing columns, merge the data
#                 if missing_columns:
#                     # Merge based on common keys ('group', 'patient', 'bundle')
#                     merged_df = pd.merge(df, df_sentences_annotation[['group', 'patient', 'bundle'] + missing_columns], 
#                                          on=['group', 'patient', 'bundle'], how='left')

#                 df = merged_df


#                 print(df)
#                 print(df.head)
#                 print(df.columns)
#                 # sys.exit()
                
                
                
#                 df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#                 # print(df.head())
#                 # # sys.exit

#                 # print(df.columns)
#                 # sys.exit()

#                 file_name = os.path.basename(largest_csv_file)
#                 name, extension = os.path.splitext(file_name)
#                 modified_name = name + "_rf_features" + extension
#                 output_file_csv = os.path.join(output_path, modified_name)
#                 output_file_excel = os.path.join(output_path, name + "_rf_features.xlsx")

                
                

#                 if constant_columns_df is None:
#                     # extracts the constant columns of acc and vel (acceleration and velocity) from the first file
#                     constant_columns = ['group', 'patient', 'bundle'] + [col for col in df.columns if 'acc' in col.lower() or 'vel' in col.lower()]
#                     # print('constant columns')
#                     # print(constant_columns)
#                     constant_columns_df = df[constant_columns]
#                     extra_columns = ['name', 'sentence', 'med_condition', 'condition']
#                     extra_columns_df = df[['group', 'patient', 'bundle'] + extra_columns].drop_duplicates()


#                 else:
#                     missing_columns = [col for col in constant_columns if col not in df.columns]
#                     if missing_columns:
#                         # Assign the values from constant_columns_df to the missing columns
#                         for col in missing_columns:
#                             df[col] = constant_columns_df[col]



#                 print(file_full_path)
#                 print(largest_csv_file)
#                 # print("-------------------")
#                 # # print(df.shape)
#                 # print(df.columns)
                
#                 # # print([col for col in df.columns if 'acc' in col.lower() or 'vel' in col.lower()])
#                 # print()
#                 # sys.exit()
#                 # Run your functions on the DataFrame
                
#                 # Run your functions on the DataFrame
#                 df_peak_vel_pos = rff.peak_velocity_positive(df, all_vel_col)
#                 df_peak_vel_neg = rff.peak_velocity_negative(df, all_vel_col)
#                 df_avg_vel = rff.average_velocity(df, all_vel_col)
#                 df_mean_speed = rff.mean_speed(df, all_vel_col)
#                 df_max_speed = rff.max_speed(df, all_vel_col)
#                 df_peak_acc_pos = rff.peak_acceleration_positive(df, all_acc_col)
#                 df_peak_acc_neg = rff.peak_acceleration_negative(df, all_acc_col)
#                 df_avg_acc = rff.average_acceleration(df, all_acc_col)
#                 df_range = rff.range_feature(df, keywords)
#                 df_total_length = rff.total_length(df, 'tbo_y')
#                 df_time_dip = rff.time_till_dip(df, keywords)
#                 df_time_peak = rff.time_till_peak(df, keywords)
#                 df_turning_points = rff.calculate_turning_points(df, keywords)
                
#                 # Merge the results into a single DataFrame
#                 df_rf_features = pd.merge(df_peak_vel_pos, df_peak_vel_neg, on=['group', 'patient', 'bundle'])
#                 df_rf_features = pd.merge(df_rf_features, df_avg_vel, on=['group', 'patient', 'bundle'])
#                 df_rf_features = pd.merge(df_rf_features, df_mean_speed, on=['group', 'patient', 'bundle'])
#                 df_rf_features = pd.merge(df_rf_features, df_max_speed, on=['group', 'patient', 'bundle'])
#                 df_rf_features = pd.merge(df_rf_features, df_peak_acc_pos, on=['group', 'patient', 'bundle'])
#                 df_rf_features = pd.merge(df_rf_features, df_peak_acc_neg, on=['group', 'patient', 'bundle'])
#                 df_rf_features = pd.merge(df_rf_features, df_avg_acc, on=['group', 'patient', 'bundle'])
#                 df_rf_features = pd.merge(df_rf_features, df_range, on=['group', 'patient', 'bundle'])
#                 df_rf_features = pd.merge(df_rf_features, df_total_length, on=['group', 'patient', 'bundle'])
#                 df_rf_features = pd.merge(df_rf_features, df_time_dip, on=['group', 'patient', 'bundle'])
#                 df_rf_features = pd.merge(df_rf_features, df_time_peak, on=['group', 'patient', 'bundle'])
#                 df_rf_features = pd.merge(df_rf_features, df_turning_points, on=['group', 'patient', 'bundle'])
                
#                 # print(df_rf_features.head())
#                 # print(df_rf_features.shape)
#                 # print(df_rf_features.columns)

#                 # adds the extra columns to the final dataframe -> name, med_condition, condition
#                 df_rf_features = pd.merge(df_rf_features, extra_columns_df, on=['group', 'patient', 'bundle'])

#                 # columns reordering
#                 column_final_order = ['group', 'patient', 'bundle', 'name', 'sentence', 'med_condition', 'condition'] 
#                 column_order = column_final_order + [col for col in df_rf_features.columns if col not in column_final_order]
#                 df_rf_features = df_rf_features[column_order]
                
#                 # # df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#                 # print(df_rf_features.head())
#                 # print(df_rf_features.shape)
#                 # print(df_rf_features.columns)
#                 # sys.exit()      


#                 # # Save the merged DataFrame to the CSV file
#                 # df_rf_features.to_csv(output_file_csv, index=False)

#                 # # Save the merged DataFrame to the Excel file
#                 # df_rf_features.to_excel(output_file_excel, index=False)

#                 # print(f"Saved the features to {output_file_csv}")