import sys
import random
import string
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d


################################################################################################################################################################
# test_df = pd.read_csv(r'D:\00000_master_thesis_new\csv files\names_dataset\centered\all_groups_C1toV2_annotated_centered.csv')
################################################################################################################################################################




def random_id_generator(length = 5):
    '''
    This function creates random ids including strings and digits
    
    argument1 = length as int
    '''
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for i in range(length))


# def min_max_normalization(df, columns_to_normalize, group_column_name, patient_column_name = 'patient', per_patient = True):
#     '''
#     This function normalizes column in a DataFrame either per patient or per sample.
#     It's the min max normalization

#     :param df: pandas DataFrame containing the data itself
#     :param columns_to_normalize: list of column names to normalize (e.g. [tbo_x, tbo_y, ...])
#     :param group_column_name: column name for the group, e.g. 'group' or 'group2'
#     :param patient_column_name: column name that has the patient names
#     :param_per_patient: default True, if True the min max normalization is done per patient, if False per sample

#     '''

#     normalized_df = df.copy()

#     if per_patient:
#         for col in columns_to_normalize:







def normalize_per_patient(df, features):
    for patient in df['patient'].unique():
        patient_df = df[df['patient'] == patient]
        for feature in features:
            max_value = patient_df[feature].max()
            min_value = patient_df[feature].min()
            df.loc[df['patient'] == patient, feature] = (patient_df[feature] - min_value) / (max_value - min_value)
    return df

def normalize_across_all_samples(df, features):
    for feature in features:
        df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())
    return df



def plot_random_samples(df, features, n_samples=5):
    # Randomly select unique bundles
    unique_bundles = df.groupby(['group', 'patient', 'bundle']).ngroup().unique()
    selected_bundles = np.random.choice(unique_bundles, size=n_samples, replace=False)

    # Plot each selected bundle
    for bundle_id in selected_bundles:
        bundle_df = df[df.groupby(['group', 'patient', 'bundle']).ngroup() == bundle_id]
        plt.figure(figsize=(10, 6))
        group, patient, bundle = bundle_df[['group', 'patient', 'bundle']].iloc[0]
        title = f"Group: {group}, Patient: {patient}, Bundle: {bundle}"
        plt.title(title)
        for feature in features:
            plt.plot(bundle_df.index, bundle_df[feature], label=feature)
        plt.xlabel('Time (rows)')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.show()



def calculate_class_weights(dataframe, group_column, patient_column, bundle_column):
    """
    Calculate class weights based on the total number of rows (samples) in each class.

    :param dataframe: Pandas DataFrame containing the dataset.
    :param group_column: Column name representing the classes (e.g., 'group2').
    :param patient_column: Column name representing the patients.
    :param bundle_column: Column name representing the individual recordings or bundles.
    :return: A tensor of class weights.
    """

    # Count the number of rows for each bundle in each patient in each class
    # bundle_counts = dataframe.groupby([group_column, patient_column, bundle_column]).size()
    unique_bundle_counts = dataframe.groupby([group_column, patient_column])[bundle_column].nunique()
    print('bundle counts')
    print(unique_bundle_counts)

    # Sum the counts for each group (class)
    class_counts = unique_bundle_counts.groupby(group_column).sum()

    print('class counts')
    print(class_counts)

    # Total number of rows (samples)
    total_samples = class_counts.sum()
    print('total samples')
    print(total_samples)


    # Calculate class weights
    class_weights = total_samples / class_counts

    # weight_for_class_0 = 1000 / (900 * 2) = 0.5556
    # weight_for_class_1 = 1000 / (100 * 2) = 5.0000

    print('class weights')
    print(class_weights)
    # Convert to torch tensor
    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32)

    return class_weights_tensor, class_counts





################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

def resampling_100(df, keywords):
    '''
    This function resamples the data to 100 rows per bundle
    parameters:
    df: pandas DataFrame containing the data
    keywords: list of keywords to be resampled
    '''
    keywords = keywords
    target_size = 100
    column_order = ['group', 'patient', 'bundle', 'name', 'med_condition', 'condition'] + keywords
    resampled_df = pd.DataFrame(columns=column_order)
    
    grouped = df.groupby(['group', 'patient', 'bundle']).size().reset_index(name='row_count')
      

    for index, row in grouped.iterrows():
        group, patient, bundle, row_count = row['group'], row['patient'], row['bundle'], row['row_count']
        # print(index, row)
        # print(group, patient, bundle, row_count)
        print(index)
        subset_df = df[(df['group'] == group) & (df['patient'] == patient) & (df['bundle'] == bundle)]  ### only filtering for those rows that belong to the current group, patient, and bundle
        subset_df = subset_df[column_order]
        
        if row_count == target_size:
            resampled_df = pd.concat([resampled_df, subset_df])
        
        else:
            interpolated_data = {}

            for keyword in keywords:
                # print(keyword)
                x_original = np.linspace(0, 1, row_count)# number of indices (row_count)
                x_new = np.linspace(0, 1, target_size) # new indices
                # print(x_new)
                y_original = subset_df[keyword].values
                # print(y)      # actual values

                interp_function = interp1d(x_original, y_original, kind='linear', fill_value='extrapolate') ### cubic interpolation, also try linear interpolation
                interpolated_data[keyword] = interp_function(x_new)

            

            interpolated_df = pd.DataFrame(interpolated_data)
            # print(interpolated_df)
            interpolated_df['group'] = group
            interpolated_df['patient'] = patient
            interpolated_df['bundle'] = bundle
            interpolated_df['name'] = subset_df['name'].values[0]
            interpolated_df['med_condition'] = subset_df['med_condition'].values[0]
            interpolated_df['condition'] = subset_df['condition'].values[0]




            resampled_df = pd.concat([resampled_df, interpolated_df])
            # print(resampled_df)

            

    return resampled_df
    # return resampled_df

################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################


################################################################################################################################################################
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def resampling_60(df, keywords):
    '''
    This function resamples the data to 60 rows per bundle
    parameters:
    df: pandas DataFrame containing the data
    keywords: list of keywords to be resampled
    '''
    target_size = 60
    column_order = ['group2', 'patient', 'bundle'] + keywords
    resampled_df = pd.DataFrame(columns=column_order)
    
    grouped = df.groupby(['group2', 'patient', 'bundle']).size().reset_index(name='row_count')
      
    for index, row in grouped.iterrows():
        group2, patient, bundle, row_count = row['group2'], row['patient'], row['bundle'], row['row_count']
        subset_df = df[(df['group2'] == group2) & (df['patient'] == patient) & (df['bundle'] == bundle)]
        subset_df = subset_df[column_order]
        
        if row_count == target_size:
            resampled_df = pd.concat([resampled_df, subset_df], ignore_index=True)
        else:
            interpolated_data = {}

            for keyword in keywords:
                x_original = np.linspace(0, 1, row_count)
                x_new = np.linspace(0, 1, target_size)
                y_original = subset_df[keyword].values

                interp_function = interp1d(x_original, y_original, kind='linear', fill_value='extrapolate')
                interpolated_data[keyword] = interp_function(x_new)

            interpolated_df = pd.DataFrame(interpolated_data)
            interpolated_df['group2'] = group2
            interpolated_df['patient'] = patient
            interpolated_df['bundle'] = bundle
            # interpolated_df['name'] = subset_df['name'].values[0]
            # interpolated_df['med_condition'] = subset_df['med_condition'].values[0]
            # interpolated_df['condition'] = subset_df['condition'].values[0]

            resampled_df = pd.concat([resampled_df, interpolated_df], ignore_index=True)

    return resampled_df
################################################################################################################################################################


################################################################################################################################################################
def patient_out_names_dataset(df, split_percent_train, random_seed):
    '''
    Explanation:
    only for control, rbd, postop and preop groups
    NOT FOR:
    CON, MILD PD, SEVERE PD
    for this, use other function
    df: dataframe, only for control, rbd, postop and preop groups
    split_percent_train: float, percentage of the patients to be taken out, e.g. 0.7 for 70% fo
    random_seed: int, random seed for reproducibility, default 1988
    
    returns: df_train, df_test
    '''
    df = df.copy()
    random.seed(random_seed)
    #create list of all groups
    all_groups = list(df['group'].unique())
    # print('all groups: ', all_groups)
    
    postop_index = next((i for i, group in enumerate(all_groups) if 'postop' in group.lower()), None) # get the index of the postop group
    if postop_index is not None:
        postop_group = all_groups.pop(postop_index) # remove the postop group from the list
        all_groups.insert(0, postop_group) # insert the postop group at the beginning of the list
    

    postop_all_patients = list(df[df['group'] == postop_group]['patient'].unique()) # get all patients from the postop group
    random.shuffle(postop_all_patients) # shuffle the list of patients
    split_idx = int(split_percent_train * len(postop_all_patients)) # get the index for the split
    postop_train_patients = postop_all_patients[:split_idx] # get the patients for the training set
    postop_test_patients = postop_all_patients[split_idx:] # get the patients for the test set

    df_postop_train = df[df['patient'].isin(postop_train_patients)] # get the rows for the training set
    df_postop_test = df[df['patient'].isin(postop_test_patients)] # get the rows for the test set



    # NOW THE SAME PATIENTS FOR PREOP GROUP
    print(postop_train_patients)
    print(postop_test_patients)

    preop_all_patients = list(df[df['group'] == 'preoperative_focus_emuDB']['patient'].unique()) # get all patients from the preop group
    preop_train_patients = [patient for patient in preop_all_patients if any(patient[:5] == postop_patient[:5] for postop_patient in postop_train_patients)] # get the same patients as in the postop training set  
    preop_test_patients = [patient for patient in preop_all_patients if any(patient[:5] == postop_patient[:5] for postop_patient in postop_test_patients)] # get the same patients as in the postop test set

    remaining_preop_patients = [patient for patient in preop_all_patients if patient not in preop_train_patients and patient not in preop_test_patients] # get the remaining patients
    random.shuffle(remaining_preop_patients) # shuffle the remaining patients#
    split_idx_preop = int(split_percent_train * len(remaining_preop_patients)) # get the index for the split


    preop_train_patients_all_patients = preop_train_patients + remaining_preop_patients[:split_idx_preop] # combine the patients from the postop training set and the patients for the training set
    preop_test_patients_all_patients = preop_test_patients + remaining_preop_patients[split_idx_preop:] # combine the patients from the postop test set and the patients for the test set
    
    df_preop_train = df[df['patient'].isin(preop_train_patients_all_patients)] # get the rows for the training set
    df_preop_test = df[df['patient'].isin(preop_test_patients_all_patients)] # get the rows for the test set



    # NOW THE SAME FOR CONTROL AND RBD GROUP
    control_all_patients = list(df[df['group'] == 'controls_focus_emuDB']['patient'].unique()) # get all patients from the control group
    random.shuffle(control_all_patients) # shuffle the list of patients
    split_idx_control = int(split_percent_train * len(control_all_patients)) # get the index for the split
    control_train_patients = control_all_patients[:split_idx_control] # get the patients for the training set
    control_test_patients = control_all_patients[split_idx_control:] # get the patients for the test set

    df_control_train = df[df['patient'].isin(control_train_patients)] # get the rows for the training set
    df_control_test = df[df['patient'].isin(control_test_patients)] # get the rows for the test set



    # RBD group
    rbd_all_patients = list(df[df['group'] == 'RBD_focus_emuDB']['patient'].unique()) # get all patients from the rbd group
    random.shuffle(rbd_all_patients) # shuffle the list of patients
    split_idx_rbd = int(split_percent_train * len(rbd_all_patients)) # get the index for the split
    rbd_train_patients = rbd_all_patients[:split_idx_rbd] # get the patients for the training set
    rbd_test_patients = rbd_all_patients[split_idx_rbd:] # get the patients for the test set

    df_rbd_train = df[df['patient'].isin(rbd_train_patients)] # get the rows for the training set
    df_rbd_test = df[df['patient'].isin(rbd_test_patients)] # get the rows for the test set


    print("Number of samples in train, rbd, control, postop, preop:", df_rbd_train.shape[0], df_control_train.shape[0], df_postop_train.shape[0], df_preop_train.shape[0])
    print("Number of samples in train, rbd, control, postop, preop:", df_rbd_test.shape[0], df_control_test.shape[0], df_postop_test.shape[0], df_preop_test.shape[0])

    df_train = pd.concat([df_control_train, df_rbd_train, df_preop_train, df_postop_train])
    df_test = pd.concat([df_control_test, df_rbd_test, df_preop_test, df_postop_test])

    return df_train, df_test
################################################################################################################################################################


################################################################################################################################################################
def patient_out_names_dataset_no_postop(df, split_percent_train, random_seed):
    '''
    Explanation:
    For control, rbd, and preop groups only.
    NOT FOR:
    CON, MILD PD, SEVERE PD, POSTOP
    
    df: dataframe, only for control, rbd, and preop groups
    split_percent_train: float, percentage of the patients to be taken out, e.g. 0.7 for 70%
    random_seed: int, random seed for reproducibility, default 1988
    
    returns: df_train, df_test
    '''
    df = df.copy()
    random.seed(random_seed)
    all_groups = list(df['group'].unique())
    
    # PREOP GROUP
    preop_all_patients = list(df[df['group'] == 'preoperative_focus_emuDB']['patient'].unique())
    random.shuffle(preop_all_patients)
    split_idx_preop = int(split_percent_train * len(preop_all_patients))
    preop_train_patients = preop_all_patients[:split_idx_preop]
    preop_test_patients = preop_all_patients[split_idx_preop:]
    
    df_preop_train = df[df['patient'].isin(preop_train_patients)]
    df_preop_test = df[df['patient'].isin(preop_test_patients)]

    # CONTROL GROUP
    control_all_patients = list(df[df['group'] == 'controls_focus_emuDB']['patient'].unique())
    random.shuffle(control_all_patients)
    split_idx_control = int(split_percent_train * len(control_all_patients))
    control_train_patients = control_all_patients[:split_idx_control]
    control_test_patients = control_all_patients[split_idx_control:]

    df_control_train = df[df['patient'].isin(control_train_patients)]
    df_control_test = df[df['patient'].isin(control_test_patients)]

    # RBD GROUP
    rbd_all_patients = list(df[df['group'] == 'RBD_focus_emuDB']['patient'].unique())
    random.shuffle(rbd_all_patients)
    split_idx_rbd = int(split_percent_train * len(rbd_all_patients))
    rbd_train_patients = rbd_all_patients[:split_idx_rbd]
    rbd_test_patients = rbd_all_patients[split_idx_rbd:]

    df_rbd_train = df[df['patient'].isin(rbd_train_patients)]
    df_rbd_test = df[df['patient'].isin(rbd_test_patients)]

    print("Number of samples in train, rbd, control, preop:", df_rbd_train.shape[0], df_control_train.shape[0], df_preop_train.shape[0])
    print("Number of samples in test, rbd, control, preop:", df_rbd_test.shape[0], df_control_test.shape[0], df_preop_test.shape[0])

    df_train = pd.concat([df_control_train, df_rbd_train, df_preop_train])
    df_test = pd.concat([df_control_test, df_rbd_test, df_preop_test])

    return df_train, df_test
################################################################################################################################################################

################################################################################################################################################################


# def calculate_class_weights(dataframe, group_column, patient_column, bundle_column):
#     """
#     Calculate class weights based on the number of samples in each class.

#     :param dataframe: Pandas DataFrame containing the dataset.
#     :param group_column: Column name representing the classes (e.g., 'group2').
#     :param patient_column: Column name representing the patients.
#     :param bundle_column: Column name representing the individual recordings or bundles.
#     :return: A tensor of class weights.
#     """

#     # Count the number of bundles (recordings) for each patient in each class
#     class_counts = dataframe.groupby([group_column, patient_column, bundle_column]).size()
    
#     # class_counts = dataframe[group_column].value_counts()
#     print('class counts')
#     print(class_counts)

#     # Sum the counts for each class
#     total_counts = class_counts.groupby(group_column).sum()
#     print('total counts')
#     print(total_counts)
#     # Total number of samples
#     total_samples = total_counts.sum()

#     # Calculate class weights
#     class_weights = total_samples / total_counts

#     # Convert to torch tensor
#     class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32)

#     return class_weights_tensor








# def min_max_normalization(df, columns_to_normalize, group_column_name, patient_column_name='patient', bundle_column_name=None, per_patient=True):
#     '''
#     This function normalizes column in a DataFrame either per patient or per sample.
#     It's the min max normalization

#     :param df: pandas DataFrame containing the data itself
#     :param columns_to_normalize: list of column names to normalize (e.g. [tbo_x, tbo_y, ...])
#     :param group_column_name: column name for the group, e.g. 'group' or 'group2'
#     :param patient_column_name: column name that has the patient names
#     :param_per_patient: default True, if True the min max normalization is done per patient, if False per sample

#     '''
#     normalized_df = df.copy()

#     # Define the grouping columns based on the parameters
#     grouping_columns = [group_column_name, patient_column_name]
#     if bundle_column_name:
#         grouping_columns.append(bundle_column_name)

#     if per_patient:
#         for col in columns_to_normalize:
#             # Group by the specified columns
#             for _, group in normalized_df.groupby(grouping_columns):
#                 col_min = group[col].min()
#                 col_max = group[col].max()
#                 # Normalize within each group and reassign the values
#                 normalized_values = (group[col] - col_min) / (col_max - col_min)
#                 normalized_df.loc[group.index, col] = normalized_values
#     else:
#         for col in columns_to_normalize:
#             col_min = normalized_df[col].min()
#             col_max = normalized_df[col].max()
#             normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)

#     return normalized_df
