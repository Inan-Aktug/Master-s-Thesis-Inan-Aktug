import numpy as np
import pandas as pd
import random
import sys

'''
The following paths are just for testing
delete them later
until ### delete above
'''



# # Load the data
# path1 = r'C:\Users\iaktug\Desktop\Master_thesis\csv files\rerun_of_second_with_new_graphs\center_values\all_groups_for_names_only_centered_values_RIGHT_VERSION.csv'
# df1 = pd.read_csv(path1)
# df_4class = df1[['group', 'name', 'patient', 'bundle', 'llip_x', 'llip_y', 'ulip_x', 'ulip_y', 'tbo_x', 'tbo_y', 'ttip_x', 'ttip_y']]
# # print(df_4class['group'].unique())

# # path = r'C:\Users\iaktug\Desktop\Master_thesis\csv files\rerun_of_second_with_new_graphs\center_values\all_groups_for_names_only_centered_values_RIGHT_VERSION.csv'
# path2 = r'C:\Users\iaktug\Desktop\Master_thesis\csv files\paper_replication\filtered_datapoints_c1_to_v1_plus_v1_duration\filtered_datapoints_paper_replication_keypoint_values_first_syllable.csv'
# df2 = pd.read_csv(path2)

# df_2class = df2[['group2', 'name', 'patient', 'bundle', 'llip_x', 'llip_y', 'ulip_x', 'ulip_y', 'tbo_x', 'tbo_y', 'ttip_x', 'ttip_y']]
### delete above


###### Resampling sequences
def resample_sequence(df, target_length=100):
    '''
    resample_senquence resamples a given sequence from a df to the target length
    df: takes dataframe as input
    target_length: int value of the target_length
    '''
    original_length = df.shape[0]
    if original_length == target_length:
        return df
    else:
        # Resample indices and round them to integers
        resampled_indices = np.linspace(0, original_length-1, target_length).astype(int)
        # Get corresponding rows
        resampled_df = df.iloc[resampled_indices].reset_index(drop=True)
        return resampled_df
    
####### How to call this function:
# ### resample from data_preprocessing.py -> first groupby then .apply(function + rest of parameters) then reset_index(drop = True)
# # .apply() is directly applied to the dataframe which is named prior to that function 
# resample_len = 200
# resampled_df = df_keypoints.groupby(['group', 'patient', 'bundle']).apply(data_preprocessing.resample_sequence, resample_len).reset_index(drop=True)
###### Resampling sequences END










######### train and test split per patient to prevent data leakage
#### what still needs to be added:
# if postoperative is taken:
# take those same patients from preoperative as well
# if preop is taken -> skip ?!
def patient_based_train_test_split(df, group_column = 'group', patient_column='patient', split_ratio=0.7, random_seed = None):
    '''
    patient_based_train_test_split creates 2 dataframes as output, train_df and test_df
    df: input dataframe
    group_column: provide group column name; 
        'group' for control_focus, rbd_focus, preoperative_focus, postoperative_focus
        'group2' for con, rbd, severe PD, mild PD
    patient_column: provide column with patient names, default = 'patient'
    split_ratio: split ratio of train and test dataframes, default = '0.7'
    random_seed: int value, select an integer as random seed which ensures reproducibility
    '''


    ''' 
    Idea for splitting patients randomly or not randomly:
    Put a boolean random_patiens = True 
    if random_patients == True:
        put the random shuffling in there
    '''
def patient_based_train_test_split(df, group_column='group', patient_column='patient', split_ratio=0.7, random_seed=None):
    unique_groups = df[group_column].unique()
    df_train_final = pd.DataFrame()
    df_test_final = pd.DataFrame()
    postoperative_group = None
    preoperative_group = None
    processed_patients = set()

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    for group in unique_groups:
        if 'postoperative' in group:
            postoperative_group = group
        if 'preoperative' in group:
            preoperative_group = group

    if postoperative_group in unique_groups:
        postop_df = df[df[group_column] == postoperative_group]
        postop_unique_patients = postop_df[patient_column].unique()
        modified_patient_names = [name.replace('_2_', '_') for name in postop_unique_patients]

        np.random.shuffle(modified_patient_names)
        split_idx = int(split_ratio * len(modified_patient_names))
        train_patients = modified_patient_names[:split_idx]
        test_patients = modified_patient_names[split_idx:]

        patient_name_map = {name[:6]: name for name in postop_unique_patients}

        train_patients_original = [patient_name_map.get(name[:6], '') for name in train_patients]
        test_patients_original = [patient_name_map.get(name[:6], '') for name in test_patients]

        train_df_postop = postop_df[postop_df[patient_column].isin(train_patients_original)]
        test_df_postop = postop_df[postop_df[patient_column].isin(test_patients_original)]

        preop_df = df[df[group_column] == preoperative_group]
        train_df_preop = preop_df[preop_df[patient_column].isin(train_patients)]
        test_df_preop = preop_df[preop_df[patient_column].isin(test_patients)]

        df_train_final = pd.concat([df_train_final, train_df_postop, train_df_preop])
        df_test_final = pd.concat([df_test_final, test_df_postop, test_df_preop])

        processed_patients.update(train_patients)
        processed_patients.update(test_patients)

        remaining_preop_df = preop_df[~preop_df[patient_column].isin(processed_patients)]
        if not remaining_preop_df.empty:
            remaining_preop_patients = remaining_preop_df[patient_column].unique()
            np.random.shuffle(remaining_preop_patients)
            split_idx = int(split_ratio * len(remaining_preop_patients))
            train_remaining_preop = remaining_preop_patients[:split_idx]
            test_remaining_preop = remaining_preop_patients[split_idx:]

            train_df_remaining_preop = remaining_preop_df[remaining_preop_df[patient_column].isin(train_remaining_preop)]
            test_df_remaining_preop = remaining_preop_df[remaining_preop_df[patient_column].isin(test_remaining_preop)]

            df_train_final = pd.concat([df_train_final, train_df_remaining_preop])
            df_test_final = pd.concat([df_test_final, test_df_remaining_preop])

            processed_patients.update(train_remaining_preop)
            processed_patients.update(test_remaining_preop)

    for group in unique_groups:
        if group in [postoperative_group, preoperative_group]:
            continue

        df_current_group = df[df[group_column] == group]
        
        if 'postoperative' in group:
            df_current_group.loc[:, 'processed_patient'] = df_current_group[patient_column].str.replace('_2_', '_')
            unique_patients = df_current_group['processed_patient'].unique()
        else:
            unique_patients = df_current_group[patient_column].unique()

        unique_patients = [p for p in unique_patients if p not in processed_patients]
        
        random.shuffle(unique_patients)
        split_idx = int(split_ratio * len(unique_patients))
        train_patients = unique_patients[:split_idx]
        test_patients = unique_patients[split_idx:]

        df_train = df_current_group[df_current_group[patient_column].isin(train_patients)]
        df_test = df_current_group[df_current_group[patient_column].isin(test_patients)]
        
        df_train_final = pd.concat([df_train_final, df_train]).reset_index(drop=True)
        df_test_final = pd.concat([df_test_final, df_test]).reset_index(drop=True)

        processed_patients.update(train_patients)
        processed_patients.update(test_patients)

    return df_train_final, df_test_final





# ######### train and test split per patient to prevent data leakage
# #### what still needs to be added:
# # if postoperative is taken:
# # take those same patients from preoperative as well
# # if preop is taken -> skip ?!
# def patient_based_train_test_split(df, group_column = 'group', patient_column='patient', split_ratio=0.7, random_seed = None):
#     '''
#     patient_based_train_test_split creates 2 dataframes as output, train_df and test_df
#     df: input dataframe
#     group_column: provide group column name; 
#         'group' for control_focus, rbd_focus, preoperative_focus, postoperative_focus
#         'group2' for con, rbd, severe PD, mild PD
#     patient_column: provide column with patient names, default = 'patient'
#     split_ratio: split ratio of train and test dataframes, default = '0.7'
#     random_seed: int value, select an integer as random seed which ensures reproducibility
#     '''


#     ''' 
#     Idea for splitting patients randomly or not randomly:
#     Put a boolean random_patiens = True 
#     if random_patients == True:
#         put the random shuffling in there
#     '''
#     # Get unique groups
#     unique_groups = df[group_column].unique()
#     df_train_final = pd.DataFrame()
#     df_test_final = pd.DataFrame()
#     postoperative_group = None
#     preoperative_group = None
#     # print(unique_groups) # debug

#     if random_seed is not None:
#         random.seed(random_seed)
#         np.random.seed(random_seed)


#     # sys.exit() # debug code
#     # Checks if the postoperative string is present in the unique_groups
#     for group in unique_groups:
#         if 'postoperative' in group:
#             postoperative_group = group
#         if 'preoperative' in group:
#             preoperative_group = group
#             # print(preoperative_group)
#     #         print('here')
#     #         print(postoperative_group)
#     # sys.exit()

#     if postoperative_group in unique_groups:
        
#         # print those statements one by one
#         '''
#         Everything working as intended!
#         '''
#         postop_df = df[df[group_column] == postoperative_group]
#         postop_unique_patients = postop_df[patient_column].unique()
#         # print(len(postop_unique_patients))
#         modified_patient_names = [name.replace('_2_', '_') for name in postop_unique_patients] ################
#         # print(len(modified_patient_names))
#         # print(modified_patient_names)


#         #Shuffle and split
#         np.random.shuffle(modified_patient_names) ###########################
#         split_idx = int(split_ratio * len(modified_patient_names)) #####################
#         # print(split_idx)
#         train_patients = modified_patient_names[:split_idx] #################
#         test_patients = modified_patient_names[split_idx:] ##################
        
#         # print('test patients')
#         # print(test_patients)
#         # print(postop_df)
#         # sys.exit()

#         # Create a mapping from the modified patient names to original names
#         patient_name_map = {name[:6]: name for name in postop_unique_patients}
#         # print('patient map name')
#         # print(patient_name_map)


#         # Map the first 6 characters of the train and test patients back to the original patient names
#         train_patients_original = [patient_name_map.get(name[:6], '') for name in train_patients]
#         test_patients_original = [patient_name_map.get(name[:6], '') for name in test_patients]
#         # print('train patients original')
#         # print(train_patients_original)

#         # # sys.exit()

#         # Filter the dataframe using the original patient names
#         train_df_postop = postop_df[postop_df[patient_column].isin(train_patients_original)]
#         test_df_postop = postop_df[postop_df[patient_column].isin(test_patients_original)]

#         # print(train_df_postop)
#         # print(test_df_postop)
#         # # sys.exit()
#         preop_df = df[df[group_column] == preoperative_group]
#         train_df_preop = preop_df[preop_df[patient_column].isin(train_patients)]
#         test_df_preop = preop_df[preop_df[patient_column].isin(test_patients)]

#         # print('train df')
#         # print(train_df_preop)
#         # print(test_patients)
#         # # sys.exit()


#         # Concatenate the preoperative and postoperative train DataFrames to the final train DataFrame
#         df_train_final = pd.concat([df_train_final, train_df_postop, train_df_preop])
#         df_test_final = pd.concat([df_test_final, test_df_postop, test_df_preop])

#         # print(df_train_final)
#         # sys.exit()

#         # remove the postop and preop group from unique groups
#         # print(type(unique_groups))
#         # Boolean mask to filter out the specified groups
#         mask = ~((unique_groups == postoperative_group) | (unique_groups == preoperative_group))
#         unique_groups = unique_groups[mask]
#         # print(unique_groups)
        
        

#     for group in unique_groups:
#         df_current_group = df[df[group_column] == group]
        
#         # sys.exit()
        
#         # Special handling for '_2_' in postoperation patients
#         if 'postoperative' in group:
#             df_current_group.loc[:, 'processed_patient'] = df_current_group[patient_column].str.replace('_2_', '_')
#             unique_patients = df_current_group['processed_patient'].unique()
#         else:
#             unique_patients = df_current_group[patient_column].unique()
#     # Shuffle and split
#         random.shuffle(unique_patients)
#         split_idx = int(split_ratio * len(unique_patients))
#         train_patients = unique_patients[:split_idx]
#         test_patients = unique_patients[split_idx:]
        
#         # print(group)
#         # print(len(unique_patients))
        

#         # Create train and test dataframes
#         df_train = df_current_group[df_current_group[patient_column].isin(train_patients)]
#         df_test = df_current_group[df_current_group[patient_column].isin(test_patients)]
        
#         # Concatenate to final DataFrames
#         df_train_final = pd.concat([df_train_final, df_train]).reset_index(drop=True)
#         df_test_final = pd.concat([df_test_final, df_test]).reset_index(drop=True)
    
    
#     return df_train_final, df_test_final

    

# # new_df_2class = patient_based_train_test_split(df_2class, 'group2', 'patient', 0.8)
# train_new_df_4class, test_new_4class = patient_based_train_test_split(df1, 'group', 'patient', 0.8)
# train_2class, test_2class = patient_based_train_test_split(df2, 'group2', 'patient', 0.8)

# print(len(train_new_df_4class))
# print(len(test_new_4class))
# print(train_new_df_4class['group'].unique())
# print(train_2class)
######### train and test split per patient to prevent data leakage END



### might need to change the features_only line so that the the features should be provided as a list
### is the features is provided as an input list, the number_features variable can be the len(feature_list)
def reshape_data(grouped_data, no_rows_per_sample, number_features = 8):
    '''
    The function reshapes the data into an usable 3D matrix for LSTM
    grouped_data: takes the dataframe as grouby input; 
        example: grouped_train = df_train.groupby(['group', 'patient', 'bundle'])
    no_rows_per_sample: define the number of rows per sample
        example: if resample, take number of resampled values;
                 if padding, take number of max padded sequence
    number_features: provide the number of features, default = 8
    '''
    reshaped_data_list = []
    labels_list = []
    target_labels_list = []
    
    for (group, patient, bundle), group_df in grouped_data:
        if group_df.shape[0] == no_rows_per_sample:
            features_only = group_df[['llip_x', 'llip_y', 'ulip_x', 'ulip_y', 'tbo_x', 'tbo_y', 'ttip_x', 'ttip_y']]
            features_np = features_only.to_numpy()
            reshaped_group = features_np.reshape(1, no_rows_per_sample, number_features)
            reshaped_data_list.append(reshaped_group)
            labels_list.append((group, patient, bundle))
            target_labels_list.append(group)

    reshaped_data = np.vstack(reshaped_data_list)
    return reshaped_data, labels_list, target_labels_list











##################################################################################################
'''Validation set'''



def create_validation_set(df, group_column='group', patient_column='patient', patient_per_group=0.2, random_seed = None):
    '''
    The function creates two outputs as dataframes: 
    df_validation with the samples for the validation set.
    df_train where the patients of the validation set are excluded.

    inputs for this function:
    df: use the df_test here 
    group_column: default 'group', change this according to group ('group' or 'group2')
    patient_column: deffault 'patient', column with the patient names
    patient_per_group: default 0.2, takes 20 percent of each group for validation
    random_seed: int value, use int value for less randomness

    '''
    unique_groups = df[group_column].unique()
    df_validation_final = pd.DataFrame()
    df_training_final = pd.DataFrame()

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        # print(random_seed)



    # Identifying postoperative and preoperative groups
    postoperative_group = None
    preoperative_group = None
    for group in unique_groups:
        if 'postoperative' in group:
            postoperative_group = group
        if 'preoperative' in group:
            preoperative_group = group

    # Handling postoperative and preoperative groups
    if postoperative_group in unique_groups:
        postop_df = df[df[group_column] == postoperative_group]
        postop_unique_patients = postop_df[patient_column].unique()
        # modified_patient_names_for_preop = [name.replace('_2_', '_') for name in postop_unique_patients]
        # print(postop_unique_patients)


        # Shuffle and select a specific number of patients for validation
        np.random.shuffle(postop_unique_patients)
        split_index = int(round(patient_per_group * len(postop_unique_patients)))
        validation_patients = postop_unique_patients[:split_index]
        training_patients = postop_unique_patients[split_index:]


        # print(validation_patients)
        # print(training_patients)
        # sys.exit()

        # Create training and validation sets for postoperative and preoperative groups
        validation_df_postop = postop_df[postop_df[patient_column].isin(validation_patients)]
        train_df_postop = postop_df[postop_df[patient_column].isin(training_patients)]
        
        # print(validation_df_postop)
        # sys.exit()
        '''
        if statement until here works fine
        now from the validation patients, replace those _2_ with '_' and nothing else
        then take those patients from the preop
        '''



        validation_preop_patients = [name.replace('_2_', '_') for name in validation_patients]
        training_preop_patients = [name.replace('_2_', '_') for name in training_patients]
        print(validation_preop_patients)
        # sys.exit()

        '''dont need this mapping'''
        # Create a mapping from the modified patient names to original names
        # patient_name_map = {name[:6]: name for name in validation_patients}
        # print(patient_name_map)
        # sys.exit()
        # # Map the first 6 characters of the train and test patients back to the original patient names
        # train_patients_original = [patient_name_map.get(name[:6], '') for name in train_patients]
        # test_patients_original = [patient_name_map.get(name[:6], '') for name in test_patients]
        ''' until here, above not needed '''



        ### from hhere on the preop
        preop_df = df[df[group_column] == preoperative_group]
        validation_df_preop = preop_df[preop_df[patient_column].isin(validation_preop_patients)]
        train_df_preop = preop_df[preop_df[patient_column].isin(training_preop_patients)]
        
        print()

        # Combine into final sets
        df_training_final = pd.concat([df_training_final, train_df_postop, train_df_preop])
        df_validation_final = pd.concat([df_validation_final, validation_df_postop, validation_df_preop])

        # Exclude processed groups from unique_groups
        unique_groups = unique_groups[~np.isin(unique_groups, [postoperative_group, preoperative_group])]



    # Process other groups
    for group in unique_groups:
        # print(group)
        group_df = df[df[group_column] == group]
        group_unique_patients = group_df[patient_column].unique()
        
        # print(group)
        # print(group_unique_patients)

        np.random.shuffle(group_unique_patients)
        split_index = int(round(patient_per_group * len(group_unique_patients)))
        group_validation_patients = group_unique_patients[:split_index]
        group_training_patients = group_unique_patients[split_index:]

        # print('validation patients')
        # print(group_validation_patients)
        # print('training patients')
        # print(group_training_patients)

        group_train_df = group_df[group_df[patient_column].isin(group_training_patients)]
        group_validation_df = group_df[group_df[patient_column].isin(group_validation_patients)]
        
        # print(group_train_df)

        df_training_final = pd.concat([df_training_final, group_train_df])
        df_validation_final = pd.concat([df_validation_final, group_validation_df])

        # print(df_training_final)

    return df_validation_final, df_training_final





# print(df_train['patient'].unique())
# print(len(df_train))
# df_validation, df_train = create_validation_set(df_train)

# print(len(df_validation))
# print(len(df_train))
# print(len(df_validation) + len(df_train))
# print(df_validation)
# print(len(df_train['patient'].unique()))

##################################################################################################


