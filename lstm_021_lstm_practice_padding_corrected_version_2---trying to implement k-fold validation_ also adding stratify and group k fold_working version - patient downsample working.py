# Import required libraries
import data_preprocessing
import lstm_utils
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import KFold
import helpful_modules
import os 

from sklearn.metrics import confusion_matrix, f1_score, recall_score, classification_report
import random
from datetime import datetime
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold



'''
information:
search for important or put the criterion for loss function optimizer on top where the variables are!

another thing:
add to the text file the number of samples for each cohort
add the names of the cohort (e.g. Con, rbd, mild PD, severe PD; or con, rbd, preoperative, postoperative)
add the number of classes



This version takes currently the same amount of patients for con and severe PD
- look for: patient reduction to same number of patients per group

'''

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

##################################################################################################
# Parameters 
# path = r'D:\00000_master_thesis_new\csv files\lstm\names_fix_group3\all_groups_C1toV2_annotated_centered_min_max_per_patient_group3_fix.csv'
# path = r'D:\00000_master_thesis_new\csv files\lstm\sentences_fix_group3\all_groups_1.2_3.6_filtered_sentences_master_file_no_unnamed_col_centered_min_max_per_patient_med_condition_and_group3_fix.csv'
# path = r'C:\Users\iaktug\Desktop\Master_thesis\csv files\paper_replication\filtered_datapoints_c1_to_v1_plus_v1_duration\filtered_datapoints_paper_replication_keypoint_values_first_syllable.csv'


path = r'D:\00000_master_thesis_new\csv files\sentences_dataset\1.2 and 3.6 sec filtered raw extractions of sentences dataset_centered_and_min_max_per_patient\all_groups_1.2_3.6_filtered_sentences_master_file_no_unnamed_col_centered_min_max_per_patient_ADDED_group3.csv'

file_and_path_annotation = r'D:\00000_master_thesis_new\csv files\json annotation\json_all_information.csv'


df = pd.read_csv(path)
annotation_df = pd.read_csv(file_and_path_annotation)


print(df)
print(annotation_df)
############## ALSO FILTER MED OFF AND ON ##############
if 'med_condition' not in df.columns:
    df = pd.merge(df, annotation_df[['group', 'patient', 'bundle', 'med_condition']], on=['group', 'patient', 'bundle'], how='left')

### filtering for all med_condtion except for on -> med condition on is out
df = df[df['med_condition'] != 'on'].reset_index(drop = True)
print(df)
# sys.exit()

current_group = 'group'
# current_group = 'group3'
# sys.exit()

grouped = df.groupby(['group', 'patient', 'bundle'])

print(df[current_group].unique())
print(df.columns)

#remove Unnamed: 0 column
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns = ['Unnamed: 0'])
print(df.columns)

# sys.exit()

# current_group = 'group3' ############

print(df[current_group].unique())
# sys.exit()
##################################################################################################



### change group and group2 when needed
print(df.columns)
# sys.exit()
df_keypoints = df[[current_group, 'name', 'patient', 'bundle', 'llip_x', 'llip_y', 'ulip_x', 'ulip_y', 'tbo_x', 'tbo_y', 'ttip_x', 'ttip_y']]
# df_keypoints = df[[current_group, 'name', 'patient', 'bundle', 'llip_y', 'ulip_y', 'tbo_y', 'ttip_y', 
                #    'llip_yvel', 'llip_yacc',
                #     'ulip_yvel', 'ulip_yacc',
                #     'tbo_yvel', 'tbo_yacc',
                #     'ttip_yvel', 'ttip_yacc']]
# print(df_keypoints['group2'].unique())
# print(df_keypoints['group'].unique())
# sys.exit()

# classes = ['CON','Mild PD', 'Severe PD']
# classes = ['CON', 'Severe PD']
classes = ['controls_focus_emuDB', 
           'postoperative_focus_emuDB',
           'preoperative_focus_emuDB', 
           'RBD_focus_emuDB']

print(df_keypoints)
# sys.exit()


##################################################################################################
''' 
patient reduction to same number of patients per group 
keywords:
- fixed patients
- same patients
- patient out
'''
# # print all patients in severe PD in current_group
# severe_pd_patients_number = len(df_keypoints[df_keypoints[current_group] == 'Severe PD']['patient'].unique())
# print(severe_pd_patients_number)
# # make a list with all patients in con
# con_patients = df_keypoints[df_keypoints[current_group] == 'CON']['patient'].unique()
# con_patients_train = con_patients[:severe_pd_patients_number]



# # take all patients in severe PD but same number for con
# # Filter the DataFrame to include only the desired patients
# filtered_df = df_keypoints[
#     (df_keypoints[current_group] == 'Severe PD') |
#     (df_keypoints['patient'].isin(con_patients_train) & (df_keypoints[current_group] == 'CON'))
# ]

# # Print the filtered DataFrame
# print(df_keypoints.shape)
# print(filtered_df.shape)
# # print counts of each group
# print(df_keypoints[current_group].value_counts())
# print(filtered_df[current_group].value_counts())
# # print(con_patients_train)

# df_keypoints = filtered_df
# # sys.exit()


##################################################################################################

# group_patient_bundle_list = ['group', 'patient', 'bundle']
group_patient_bundle_list = [current_group, 'patient', 'bundle']
print(df_keypoints[current_group].unique())


### chose your classes of interest
df_con_severePD = df_keypoints[df_keypoints[current_group].isin(classes)]
df_keypoints = df_con_severePD.reset_index(drop = True)
print(df_con_severePD)
print(df_keypoints)
print('unique groups')
print(df_keypoints[current_group].unique())
# patients per group3
print(df_keypoints.groupby(current_group)['patient'].nunique())
# sys.exit()
#this path is where the model is saved 
placeholder = ''
model_dir = r'D:\00000_master_thesis_new\lstm_models\{timestamp}'.format(timestamp = timestamp)


model_file_name = r'padded_correct{placeholder}.pth' ######### need to change this -> get it under the random id seed or make the random id see first and replace placeholder with the random id so that it has an unique id


path_for_model_saving = os.path.join(model_dir, model_file_name)
# path_for_model_saving = r'D:\master_thesis\2024\models\lstm_padded_correct_pad\padded_correct.pth'### .pth
path_for_saving_results = r'D:\00000_master_thesis_new\results\001 MT Results\LSTM results\con vs rbd vs preop vs postop\02 names dataset'

# Variables for Matrix
features = ['llip_x', 'llip_y', 'ulip_x', 'ulip_y', 'tbo_x', 'tbo_y', 'ttip_x', 'ttip_y']
# features = ['llip_y','ulip_y', 'tbo_y', 'ttip_y',
#             'llip_yvel', 'llip_yacc',
#             'ulip_yvel', 'ulip_yacc',
#             'tbo_yvel', 'tbo_yacc',
#             'ttip_yvel', 'ttip_yacc']

# calculate max sequence length
max_sequence_length = df_keypoints.groupby(group_patient_bundle_list).size().max()
print(max_sequence_length)
print(df_keypoints.groupby(group_patient_bundle_list).size().sort_values(ascending=False).head(10))
longest_sequences = df_keypoints.groupby(group_patient_bundle_list).size().sort_values(ascending=False).head(10)
for group, length in longest_sequences.items():
    print(f"Group: {group}, Length: {length}")


# sys.exit()

# print value counts for each group
# print(df_keypoints[current_group].value_counts())
# sys.exit()
# # print all patients in severe PD in current_group
# print(len(df_keypoints[df_keypoints[current_group] == 'Severe PD']['patient'].unique()))
# print(len(df_keypoints[df_keypoints[current_group] == 'CON']['patient'].unique()))



# sys.exit()
number_of_features = len(features)
max_sequence_length = max_sequence_length # adjust that to the maximum sample lengths in rows
test_sequence = df[features].values
print(test_sequence)
print(max_sequence_length)
# sys.exit()



print(df_keypoints)
# ################################
normalization_per_patient = False
if normalization_per_patient == True:
    normalized_df = pd.DataFrame()
    normalized_df = helpful_modules.normalize_per_patient(df_keypoints, features)
    df_keypoints = normalized_df
# normalized_df_sample = helpful_modules.normalize_across_all_samples(df, features)
# print(df_keypoints)
# print(normalized_df)

# # helpful_modules.plot_random_samples(normalized_df, features)
# helpful_modules.plot_random_samples(normalized_df_sample, features)

# df_keypoints = normalized_df
print(df_keypoints)
# sys.exit()

#################################
'''class weights'''

class_weights, class_sample_counts = helpful_modules.calculate_class_weights(df_keypoints, current_group, 'patient', 'bundle')
print(class_weights)
# print(class_sample_counts)
print(class_sample_counts) 

# 
# print(df_keypoints['group'] == 'preoperative_focus_emuDB')
# sys.exit()


# class_weights = torch.tensor([1.0, 1.0])
# print(class_weights)

# print(type(class_weights))
# print(class_sample_counts)
# print(type(class_sample_counts))
# sys.exit()
#################################

####################################
####################################
####################################
####################################
''' CHANGE THIS TO CONTROL FOCUS EMUDB, etc'''

def custom_group_split(df, current_group, n_splits=5):
    # Get unique patients for each group
    cohorts = [
        'controls_focus_emuDB',
        'RBD_focus_emuDB',
        'postoperative_focus_emuDB',
        'preoperative_focus_emuDB'
    ]
    
    patients_by_cohort = {cohort: df[df[current_group] == cohort]['patient'].unique() for cohort in cohorts}
    
    # Shuffle patients in each cohort
    for cohort in cohorts:
        np.random.shuffle(patients_by_cohort[cohort])
    
    # Calculate number of patients per fold for each cohort
    n_patients_per_fold = {cohort: max(1, len(patients) // n_splits) for cohort, patients in patients_by_cohort.items()}
    
    for i in range(n_splits):
        val_patients = []
        for cohort in cohorts:
            start = i * n_patients_per_fold[cohort]
            end = min((i + 1) * n_patients_per_fold[cohort], len(patients_by_cohort[cohort]))
            val_patients.extend(patients_by_cohort[cohort][start:end])
        
        # Create mask for validation set
        val_mask = df['patient'].isin(val_patients)
        
        train_idx = df[~val_mask].index
        val_idx = df[val_mask].index
        
        # Get training patients
        train_patients = df.loc[~val_mask, 'patient'].unique()
        
        yield train_idx, val_idx, train_patients, val_patients



# def custom_group_split(df, n_splits=5):
#     # Get unique patients for each group
#     control_patients = df[df[current_group] == 'CON']['patient'].unique()
#     pd_patients = df[df[current_group] == 'Severe PD']['patient'].unique()
   
#     # Shuffle patients
#     np.random.shuffle(control_patients)
#     np.random.shuffle(pd_patients)
   
#     # Calculate number of patients per fold
#     n_control_per_fold = max(1, len(control_patients) // n_splits)
#     n_pd_per_fold = max(1, len(pd_patients) // n_splits)
   
#     for i in range(n_splits):
#         # Select validation patients
#         start_control = i * n_control_per_fold
#         end_control = min((i + 1) * n_control_per_fold, len(control_patients))
#         start_pd = i * n_pd_per_fold
#         end_pd = min((i + 1) * n_pd_per_fold, len(pd_patients))
        
#         val_control = control_patients[start_control:end_control]
#         val_pd = pd_patients[start_pd:end_pd]
#         val_patients = np.concatenate([val_control, val_pd])
       
#         # Create mask for validation set
#         val_mask = df['patient'].isin(val_patients)
       
#         train_idx = df[~val_mask].index
#         val_idx = df[val_mask].index
       
#         # Get training patients
#         train_patients = df.loc[~val_mask, 'patient'].unique()
       
#         yield train_idx, val_idx, train_patients, val_patients


# def custom_group_split(df, n_splits=5):
#     # Get unique patients for each group
#     control_patients = df[df[current_group] == 'CON']['patient'].unique()
#     pd_patients = df[df[current_group] == 'Severe PD']['patient'].unique()
    
#     # Shuffle patients
#     np.random.shuffle(control_patients)
#     np.random.shuffle(pd_patients)
    
#     # Calculate number of patients per fold
#     n_control_per_fold = len(control_patients) // n_splits
#     n_pd_per_fold = len(pd_patients) // n_splits
    
#     for i in range(n_splits):
#         # Select validation patients
#         val_control = control_patients[i*n_control_per_fold:(i+1)*n_control_per_fold]
#         val_pd = pd_patients[i*n_pd_per_fold:(i+1)*n_pd_per_fold]
#         val_patients = np.concatenate([val_control, val_pd])
        
#         # Create mask for validation set
#         val_mask = df['patient'].isin(val_patients)
        
#         train_idx = df[~val_mask].index
#         val_idx = df[val_mask].index
        
#         # Get training patients
#         train_patients = df.loc[~val_mask, 'patient'].unique()
        
#         yield train_idx, val_idx, train_patients, val_patients




# Input variables for LSTM
batch_size = 64 # 16, 32, 64, 128,
num_hidden_layers = 64 # 16, 32, 64, 128,
num_layers = 2  # 1, 2, 3, 4
num_classes = len(classes) # change this according to classes 2 for con vs severe pd and 4 for the rest 
num_epochs =  40  # 100
learning_rate = 0.0001  # 0.0001
optimizer_ = torch.optim.Adam       # chose between SGD or Adam



random_seed = 42 # put None for randomness and number for reproduceability
random_id = f"{timestamp}_{current_group}_{'-'.join(classes)}"
# classes_in_this_model = df_keypoints[current_group].unique()
classes_in_this_model = classes

placeholder = random_id
identifier = f"{timestamp}_{current_group}_{'-'.join(classes)}"


##################################################################################################

''' write this function to the helpful_modules'''
if not os.path.exists(path_for_saving_results):
    os.makedirs(path_for_saving_results)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

##################################################################################################s


# print(df_keypoints['patient'].unique())
# print(len(df_keypoints))
### splitting into train and test set - works!
df_train, df_test = data_preprocessing.patient_based_train_test_split(df= df_keypoints, 
                                                                      group_column= current_group, 
                                                                      patient_column= 'patient', 
                                                                      split_ratio= 0.8, 
                                                                      random_seed= random_seed)

print(df_train)
#print the number of patients in each group
print(df_train[current_group].value_counts())
print(df_test[current_group].value_counts())


# For training set
train_samples_per_group = df_train.groupby('group')['patient'].nunique()
print("Number of unique samples in each group (Training set):")
print(train_samples_per_group)

# For test set
test_samples_per_group = df_test.groupby('group')['patient'].nunique()
print("\nNumber of unique samples in each group (Test set):")
print(test_samples_per_group)


# For training set
train_samples_per_group = df_train.groupby(['group', 'patient', 'bundle']).size().groupby(level=0).size()
print("Number of unique samples in each group (Training set):")
print(train_samples_per_group)

# For test set
test_samples_per_group = df_test.groupby(['group', 'patient', 'bundle']).size().groupby(level=0).size()
print("\nNumber of unique samples in each group (Test set):")
print(test_samples_per_group)

sys.exit()
sys.exit()

# # print if any patient is duplicated in the test set
# print(df_test['patient'].duplicated().sum())
# # print(df_test['patient'].unique())

# print("Unique patients in train set:", df_train['patient'].nunique())
# print("Unique patients in test set:", df_test['patient'].nunique())

# train_patients = set(df_train['patient'].unique())
# test_patients = set(df_test['patient'].unique())
# overlap = train_patients.intersection(test_patients)
# print("Overlapping patients:", overlap)
# print("Number of overlapping patients:", len(overlap))

# if 'preoperative' in df_train[current_group].unique():
#     preop_train = set(df_train[df_train[current_group].str.contains('preoperative')]['patient'].unique())
#     postop_train = set(df_train[df_train[current_group].str.contains('postoperative')]['patient'].unique())
#     print("Preop patients not in postop (train):", preop_train - postop_train)
#     print("Postop patients not in preop (train):", postop_train - preop_train)


# # Check for preoperative patients
# preop_patients = df_train[df_train['group'] == 'preoperative_focus_emuDB']['patient'].value_counts()
# print(preop_patients[preop_patients > 1])

# # Do the same for the test set
# preop_patients_test = df_test[df_test['group'] == 'preoperative_focus_emuDB']['patient'].value_counts()
# print(preop_patients_test[preop_patients_test > 1])
# # sys.exit()
# print(df_train['patient'].unique())
# print(df_test['patient'].unique())
# sys.exit()
# Define the number of folds
n_splits = 5 #5
custom_splits = custom_group_split(df_train, current_group, n_splits=n_splits)

print(df_test)
print(df_keypoints)
print(df_train)

# print patient name for each group in train and test set
print(df_train['patient'].unique())
print(df_test['patient'].unique())

# only unique patients in preoperative and postoperative
print('preop patients train')
print(df_train[df_train['group'] == 'preoperative_focus_emuDB']['patient'].unique())
print('postop patients train')
print(df_train[df_train['group'] == 'postoperative_focus_emuDB']['patient'].unique())
# only unique patients in preoperative and postoperative test
print('preop patients test')
print(df_test[df_test['group'] == 'preoperative_focus_emuDB']['patient'].unique())
print('postop patients test')
print(df_test[df_test['group'] == 'postoperative_focus_emuDB']['patient'].unique())

print()
# sys.exit()


patient_counts_nunique = df_train.groupby(current_group)['patient'].nunique()
patient_counts_sample_counts = df_train.groupby([current_group, 'patient']).size()
print(patient_counts_nunique)
print(patient_counts_sample_counts)


##################################################################################################




##################################################################################################
''' 
CREATE AN EMPTY MATRIX WITH 0's for X 8(TRAIN) AND Y(TEST)
the number of sequences are needed for the empty matrix
'''

number_of_sequences_train = df_train.groupby(group_patient_bundle_list).ngroups
number_of_sequences_test = df_test.groupby(group_patient_bundle_list).ngroups
# number_of_sequences_validation = df_validation.groupby(group_patient_bundle_list).ngroups
# print(number_of_sequences_validation)


# sys.exit()
X = np.zeros((number_of_sequences_train, max_sequence_length, number_of_features))
Y = np.zeros((number_of_sequences_test, max_sequence_length, number_of_features))
# V = np.zeros((number_of_sequences_validation, max_sequence_length, number_of_features))
# print(X.shape)
# print(Y.shape)
# print(V.shape)
# sys.exit()
### iterate over the dataframe and fill in the matrix with the time series values per sample
### keep track of the actual length of one sample
##################################################################################################


##################################################################################################
'''
in the next steps, after creating the empty matrix prior to this, I am filling the matrix with values from my dataframe
the current_actual_len_train keeps track of the actual length of each sample and stores it in a list
also the label for each group needs to be stored so it can later be assigned for the classification task and kept track of

The for loops can be written as function, so that I can call the function and it does the same labeling for all dataframes
'''

actual_len_train = []
actual_len_test = []
actual_len_validation = []

grouped_train = df_train.groupby(group_patient_bundle_list)
grouped_test = df_test.groupby(group_patient_bundle_list)

print(df_keypoints.groupby(group_patient_bundle_list).size().max())
print(df_train.groupby(group_patient_bundle_list).size().max())


# For the original df_keypoints
print("Top 10 longest sequences in original df_keypoints:")
longest_original = df_keypoints.groupby(group_patient_bundle_list).size().sort_values(ascending=False).head(50)
for group, length in longest_original.items():
    print(f"Group: {group}, Length: {length}")

print("\n")

# For df_train
print("Top 10 longest sequences in df_train:")
longest_train = df_train.groupby(group_patient_bundle_list).size().sort_values(ascending=False).head(50)
for group, length in longest_train.items():
    print(f"Group: {group}, Length: {length}")

# print all patients in preoperative in current_group which are not in postoperative
print(df_train[df_train[current_group] == 'preoperative_focus_emuDB']['patient'].unique())
print(df_train[df_train[current_group] == 'postoperative_focus_emuDB']['patient'].unique())

# sys.exit()


# grouped_validation = df_validation.groupby(group_patient_bundle_list)
print(grouped_train)
for i, (key, group) in enumerate(grouped_train):
    sequence = group[features].values
    # print(sequence)
    # print(group[features])
    # print(len(sequence))
# sys.exit()

labels_for_train = [] 
labels_for_test = []
labels_for_validation = []

how_many_exceed_train = []
how_many_exceed_test = []
### only for train data
for i, (key, group) in enumerate(grouped_train):
    # print(f'i:',{i})
    # print(f'key:', {key})
    # print('group')
    # print(group) 
    # Takes
    sequence = group[features].values
    # print(type(sequence))
    current_actual_len_train = len(sequence)


    if current_actual_len_train > max_sequence_length:
        # print(f"Warning: Sequence length {current_actual_len_train} exceeds max_sequence_length {max_sequence_length}")
        # print(f"Group: {key}")
        how_many_exceed_train.append(key)
    actual_len_train.append(current_actual_len_train)


    X[i, :current_actual_len_train, :] = sequence[:max_sequence_length]
    labels_for_train.append(key[0])

# print(len(how_many_exceed_train))

# print(len(actual_len_train))
# print(X.shape)



### only for test data
for i, (key, group) in enumerate(grouped_test):
    sequence = group[features].values
    # print(len(sequence))
    current_actual_len_test = len(sequence)
    if current_actual_len_train > max_sequence_length:
        print(f"Warning: Sequence length {current_actual_len_train} exceeds max_sequence_length {max_sequence_length}")
        print(f"Group: {key}")
        how_many_exceed_test.append(key)
    actual_len_train.append(current_actual_len_train)

    actual_len_test.append(current_actual_len_test)

    Y[i, :current_actual_len_test, :] = sequence[:max_sequence_length]

    labels_for_test.append(key[0])
# print(how_


##################################################################################################

##################################################################################################
'''
this step consits of using the label encoder so that the labels which are characters right now are encoded into
numerical 
example:
con, rbd, pre, post classes
[0 , 1 ,  2 ,    3] classes but numerical

Use the labelencoder.fit() only for the training set
the .transform then needs to be used on both, training and testing set
'''

### reshaping data individually for each dataframe (df_train and df_test)
label_encoder = LabelEncoder()

label_encoder.fit(labels_for_train)

target_labels_list_training = label_encoder.transform(labels_for_train)
target_labels_list_testing = label_encoder.transform(labels_for_test)
target_labels_list_validation = label_encoder.transform(labels_for_validation)
# print(target_labels_list_training)
# print(target_labels_list_testing)
# print(target_labels_list_validation)
# sys.exit()
##################################################################################################


##################################################################################################
'''
now I need to convert the matrices X and Y into tensors
and also the actual lengths?! #####  <----- recheck this part
'''

X = torch.tensor(X, dtype = torch.float32)
Y = torch.tensor(Y, dtype = torch.float32)
# V = torch.tensor(V, dtype = torch.float32)

train_labels_tensor = torch.tensor(target_labels_list_training, dtype=torch.long)
test_labels_tensor = torch.tensor(target_labels_list_testing, dtype=torch.long)
# validation_labels_tensor = torch.tensor(target_labels_list_validation, dtype=torch.long)

actual_len_train_tensor = torch.tensor(actual_len_train, dtype=torch.long)
actual_len_test_tensor = torch.tensor(actual_len_test, dtype=torch.long)
# actual_len_validation_tensor = torch.tensor(actual_len_validation, dtype=torch.long)


# print(actual_len_validation)
# sys.exit()
##################################################################################################



##################################################################################################
'''
DataLoader preparation
The DataLoader now takes as input the different values from before so it can handle the mapping / right indexing
automatically, meaning that nothing should get mixed up when everything is loaded correctly in the DataLoader
'''
train_dataset = lstm_utils.Data(X, train_labels_tensor, actual_len_train_tensor)
test_dataset = lstm_utils.Data(Y, test_labels_tensor, actual_len_test_tensor)
# validation_dataset = lstm_utils.Data(V, validation_labels_tensor, actual_len_validation_tensor)

'''Now the actual DataLoader is created so that the NN can use the given information correctly'''
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,  #'''False because no shuffling of the data while testing is needed'''
                         num_workers=0)



##################################################################################################



# print(train_labels_tensor)
# sys.exit()

##################################################################################################
'''
This is for the LSTM plots

'''

def plot_losses_straight(all_train_losses, all_val_losses, path_for_saving_results, identifier):
    epochs = range(1, all_train_losses.shape[0] + 1)
    
    plt.figure(figsize=(10, 6))
    
    # Plot each run's training loss
    for i in range(all_train_losses.shape[1]):
        plt.plot(epochs, all_train_losses[:, i], color='#9999FF', alpha=0.5, linewidth=1)
    
    # Plot each run's validation loss
    for i in range(all_val_losses.shape[1]):
        plt.plot(epochs, all_val_losses[:, i], color='#FFB347', alpha=0.5, linewidth=1)
    
    # Plot mean losses
    plt.plot(epochs, np.mean(all_train_losses, axis=1), color='#0000FF', label='Mean Train Loss', linewidth=2)
    plt.plot(epochs, np.mean(all_val_losses, axis=1), color='#FFA500', label='Mean Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss (Multiple Runs)', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(path_for_saving_results, f'training_validation_loss_multiple_runs_{identifier}.png'), dpi=300)
    plt.close()




def plot_losses_with_std(mean_train_losses, std_train_losses, mean_val_losses, std_val_losses, path_for_saving_results, identifier):
    epochs = range(1, len(mean_train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_train_losses, label='Train', color='#9999FF', linewidth=2)
    plt.plot(epochs, mean_val_losses, label='Validation', color='#FFA500', linewidth=2)

    plt.fill_between(epochs, mean_train_losses - std_train_losses, mean_train_losses + std_train_losses, 
                     color='#9999FF', alpha=0.3)
    plt.fill_between(epochs, mean_val_losses - std_val_losses, mean_val_losses + std_val_losses, 
                     color='#FFB347', alpha=0.3)

    plt.title('Training and Validation Loss (with Standard Deviation)', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.legend(loc = 'upper left')
    # plt.grid(True, linestyle='--', alpha=0.7)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(path_for_saving_results, f'training_validation_loss_with_std_{identifier}.png'), dpi=300)
    plt.close()




##################################################################################################




##################################################################################################
'''
modified with model.train()
and model.eval()
This needs to be reworked on:
putt all parameters and variables somewhere on top!
defining parameters 
defining epoch sizes
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}') 


model = lstm_utils.LSTM_Classifier(input_size = number_of_features, 
                        hidden_size = num_hidden_layers, 
                        num_layers = num_layers,
                        num_classes = num_classes,
                        max_sequence_length = max_sequence_length)
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ### important: can use Adam as optimizer as well
# # optimizer - Can also try Adam as optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
optimizer = optimizer_(model.parameters(), lr = learning_rate)


'''
This is my actual training loop.
here, I finaly need to use the train_loader, 
now every epoch is taking the correct data from the train_loader variable
'''
##################### mapping classes for better plots ############################

# class_mapping = {'CON': 'Control', 'Severe PD': 'Severe PD'}
# mapped_classes = [class_mapping[c] for c in classes]
class_mapping = {'controls_focus_emuDB': 'Control',
                 'RBD_focus_emuDB': 'RBD',
                 'postoperative_focus_emuDB': 'Post-operative',
                 'preoperative_focus_emuDB': 'Pre-operative'}
                
desired_order = ['Control', 'RBD', 'Pre-operative', 'Post-operative']

mapped_classes = [class_mapping[c] for c in classes]
##################### mapping classes for better plots ############################

# Lists to store results
all_train_losses = []
all_val_losses = []
all_val_accuracies = []

# running_loss = 0.0
# train_loss_list = []
# validation_loss_list = []


for fold, (train_idx, val_idx, train_patients, val_patients) in enumerate(custom_splits):
    print(f"\nFold {fold + 1}/{n_splits}")
    
    # Split data for this fold
    df_train_fold = df_train.iloc[train_idx]
    df_val_fold = df_train.iloc[val_idx]
    
    # Print information about the split
    print(f"Training set: {len(train_patients)} patients")
    print(f"Validation set: {len(val_patients)} patients")
    
    print("\nValidation set patients:")
    for group in ['controls_focus_emuDB', 'preoperative_focus_emuDB', 'postoperative_focus_emuDB', 'RBD_focus_emuDB']:
        group_patients = [patient for patient in val_patients if df_train[df_train['patient'] == patient][current_group].iloc[0] == group]
        print(f"{group}: {', '.join(group_patients)}")
    
    print("\nTraining set patients:")
    for group in ['controls_focus_emuDB', 'preoperative_focus_emuDB', 'postoperative_focus_emuDB', 'RBD_focus_emuDB']:
        group_patients = [patient for patient in train_patients if df_train[df_train['patient'] == patient][current_group].iloc[0] == group]
        print(f"{group}: {', '.join(group_patients)}")
    
    print("\nValidation set composition:")
    print(df_val_fold.groupby(current_group)['patient'].nunique())
    
    # Process train fold -> padding
    grouped_train_fold = df_train_fold.groupby(group_patient_bundle_list)
    # print("group_patient_bundle_list:", group_patient_bundle_list)
    X_train = np.zeros((len(grouped_train_fold), max_sequence_length, number_of_features))
    actual_len_train = []
    labels_for_train = []

    for i, (key, group) in enumerate(grouped_train_fold):
        sequence = group[features].values
        current_actual_len_train = len(sequence)
        actual_len_train.append(current_actual_len_train)
        # print(f"Sample {i}: Actual length = {current_actual_len_train}")
        X_train[i, :current_actual_len_train, :] = sequence[:max_sequence_length]
        labels_for_train.append(key[0])

    # Process validation fold
    grouped_val_fold = df_val_fold.groupby(group_patient_bundle_list)
    X_val = np.zeros((len(grouped_val_fold), max_sequence_length, number_of_features))
    actual_len_val = []
    labels_for_val = []

    for i, (key, group) in enumerate(grouped_val_fold):
        sequence = group[features].values
        current_actual_len_val = len(sequence)
        actual_len_val.append(current_actual_len_val)
        X_val[i, :current_actual_len_val, :] = sequence[:max_sequence_length]
        labels_for_val.append(key[0])

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    
    train_labels_tensor = torch.tensor(label_encoder.transform(labels_for_train), dtype=torch.long)
    val_labels_tensor = torch.tensor(label_encoder.transform(labels_for_val), dtype=torch.long)
    
    actual_len_train_tensor = torch.tensor(actual_len_train, dtype=torch.long)
    actual_len_val_tensor = torch.tensor(actual_len_val, dtype=torch.long)

    # Create DataLoaders
    train_dataset = lstm_utils.Data(X_train, train_labels_tensor, actual_len_train_tensor)
    val_dataset = lstm_utils.Data(X_val, val_labels_tensor, actual_len_val_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    # Create DataLoaders for this fold

    # Initialize model, optimizer, criterion
    model = lstm_utils.LSTM_Classifier(input_size=number_of_features, 
                                       hidden_size=num_hidden_layers, 
                                       num_layers=num_layers,
                                       num_classes=num_classes,
                                       max_sequence_length=max_sequence_length).to(device)
    optimizer = optimizer_(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Training loop for this fold
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0 #### change this to 0.0
        for batch_idx, (data, actual_lengths, labels) in enumerate(train_loader):
            data, actual_lengths, labels = data.to(device), actual_lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data, actual_lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        average_loss = running_loss / len(train_loader)
        train_losses.append(average_loss)
        
        # Validation
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_data, val_actual_lengths, val_labels in val_loader:
                val_data, val_actual_lengths, val_labels = val_data.to(device), val_actual_lengths.to(device), val_labels.to(device)
                val_outputs = model(val_data, val_actual_lengths)
                val_loss = criterion(val_outputs, val_labels)
                running_val_loss += val_loss.item()
                _, predicted = torch.max(val_outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum().item()
        
        average_val_loss = running_val_loss / len(val_loader)
        val_losses.append(average_val_loss)
        val_accuracy = 100 * correct / total
        
        print(f'Epoch: {epoch}, Training Loss: {average_loss:.4f}, Validation Loss: {average_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
    
    # Store results for this fold
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_val_accuracies.append(val_accuracy)

# Calculate average performance across folds
# Reshape loss lists
all_train_losses = np.array(all_train_losses).T  # Shape: (num_epochs, num_folds)
all_val_losses = np.array(all_val_losses).T  # Shape: (num_epochs, num_folds)

avg_val_accuracy = np.mean(all_val_accuracies)
print(f"Average Validation Accuracy across folds: {avg_val_accuracy:.2f}%")



# Call both plotting functions
plot_losses_straight(all_train_losses, all_val_losses, path_for_saving_results, identifier)

# Calculate mean and std for the std version
mean_train_losses = np.mean(all_train_losses, axis=1)
std_train_losses = np.std(all_train_losses, axis=1)
mean_val_losses = np.mean(all_val_losses, axis=1)
std_val_losses = np.std(all_val_losses, axis=1)

plot_losses_with_std(mean_train_losses, std_train_losses, mean_val_losses, std_val_losses, path_for_saving_results, identifier)







# Create epochs array
epochs = range(1, len(mean_train_losses) + 1)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(epochs, mean_train_losses, label='Train', color='#9999FF', linewidth=2)
plt.plot(epochs, mean_val_losses, label='Validation', color='#FF9999', linewidth=2) # #FF9999

# Add shaded regions for standard deviation
plt.fill_between(epochs, mean_train_losses - std_train_losses, mean_train_losses + std_train_losses, 
                 color='#9999FF', alpha=0.3)
plt.fill_between(epochs, mean_val_losses - std_val_losses, mean_val_losses + std_val_losses, 
                 color='#FF9999', alpha=0.3)

# Customize the plot
plt.title('Training and Validation Loss', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.legend(loc = 'upper left')
# plt.grid(True, linestyle='--', alpha=0.7)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Customize tick labels
plt.tick_params(axis='both', which='major', labelsize=12)

# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(path_for_saving_results, f'training_validation_loss_{identifier}.png'), dpi=300)
plt.close()












all_preditions = []
all_labels = []

# Initialize the final model
final_model = lstm_utils.LSTM_Classifier(input_size=number_of_features, 
                                         hidden_size=num_hidden_layers, 
                                         num_layers=num_layers,
                                         num_classes=num_classes,
                                         max_sequence_length=max_sequence_length).to(device)

# Initialize optimizer and criterion for final training
final_optimizer = optimizer_(final_model.parameters(), lr=learning_rate)
final_criterion = nn.CrossEntropyLoss(weight=class_weights)

# Final training loop
print("\nFinal training on all data:")
for epoch in range(num_epochs):
    final_model.train()
    running_loss = 0.0
    for batch_idx, (data, actual_lengths, labels) in enumerate(train_loader):
        data, actual_lengths, labels = data.to(device), actual_lengths.to(device), labels.to(device)
        final_optimizer.zero_grad()
        outputs = final_model(data, actual_lengths)
        loss = final_criterion(outputs, labels)
        loss.backward()
        final_optimizer.step()
        running_loss += loss.item()
    
    average_loss = running_loss / len(train_loader)
    print(f'Epoch: {epoch}, Training Loss: {average_loss:.4f}')

# Save the final model if needed
torch.save(final_model.state_dict(), 'final_lstm_model.pth')







# Evaluate the final model on the test set
final_model.eval()
correct = 0
total = 0
all_preditions = []
all_labels = []

with torch.no_grad():
    for test_data, test_actual_lengths, test_labels in test_loader:
        test_data, test_actual_lengths, test_labels = test_data.to(device), test_actual_lengths.to(device), test_labels.to(device)
        test_outputs = final_model(test_data, test_actual_lengths)
        _, predicted = torch.max(test_outputs.data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum().item()
        
        # Collect predictions and true labels
        all_preditions.extend(predicted.cpu().numpy())
        all_labels.extend(test_labels.cpu().numpy())

test_accuracy = 100 * correct / total
accuracy = test_accuracy
print(f'Final Model Test Accuracy: {test_accuracy:.2f}%')








# identifier = f"{timestamp}_{current_group}_{'-'.join(classes)}"

# Calculate metrics
cm = confusion_matrix(all_labels, all_preditions)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_percentage = cm_normalized * 100

class_recall = cm_normalized.diagonal()
balanced_accuracy = np.mean(class_recall)

f1 = f1_score(all_labels, all_preditions, average='weighted')
recall = recall_score(all_labels, all_preditions, average='weighted')

report = classification_report(all_labels, all_preditions, target_names=classes)

# Define filenames
plot_filename = os.path.join(path_for_saving_results, f'loss_plot_{identifier}.png')
info_filename = os.path.join(path_for_saving_results, f'training_info_{identifier}.txt')
cm_filename = os.path.join(path_for_saving_results, f'confusion_matrix_{identifier}.png')







cm_percentage_formatted = np.array([[f'{val:.1f}%' for val in row] for row in cm_percentage])

# Create a new confusion matrix with the desired order
cm_ordered = np.zeros((4,4))
for i, true_class in enumerate(desired_order):
    for j, pred_class in enumerate(desired_order):
        true_index = mapped_classes.index(true_class)
        pred_index = mapped_classes.index(pred_class)
        cm_ordered[i, j] = cm_percentage[true_index, pred_index]

# Format the confusion matrix values as percentages
cm_percentage_formatted = np.array([[f'{val:.1f}%' for val in row] for row in cm_ordered])

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_ordered, annot=cm_percentage_formatted, fmt='', cmap='Blues', 
            xticklabels=desired_order, yticklabels=desired_order)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.savefig(cm_filename)
plt.close()



# Save the information to file
with open(info_filename, 'w') as file:
    file.write(f'Timestamp: {timestamp}\n')
    file.write(f'Dataset path: {path}\n')
    file.write(f'Current group: {current_group}\n')
    file.write(f'Classes: {", ".join(classes)}\n')
    file.write(f'Accuracy of the network on the {len(test_dataset)} test data: {accuracy:.2f}%\n')
    file.write(f'Balanced Accuracy: {balanced_accuracy * 100:.2f}%\n')
    file.write(f'F1 Score: {f1:.2f}\n')
    file.write(f'Recall: {recall:.2f}\n')
    file.write(f'Learning Rate: {learning_rate}\n')
    file.write(f'Batch size: {batch_size}\n')
    file.write(f'Number of Hidden Units per Layer: {num_hidden_layers}\n')
    file.write(f'Number of Layers: {num_layers}\n')
    file.write(f'Number of Classes: {num_classes}\n')
    file.write(f'Epochs: {num_epochs}\n')
    file.write(f'Optimizer: {str(optimizer_)}\n')
    file.write(f'Loss Function: {criterion}\n')
    file.write(f'Random seed: {random_seed}\n')
    file.write(f'Normalization: {normalization_per_patient}\n')
    file.write(f'Class weights: {class_weights}\n')
    file.write(f'All validation accuracies: {all_val_accuracies}\n\n')
    file.write('Class sample counts:\n')
    file.write(class_sample_counts.to_string() + '\n\n')
    file.write('Classification Report:\n')
    file.write(report + '\n')
    file.write('\nConfusion Matrix:\n')
    file.write(str(cm) + '\n')


print(f"Results saved with identifier: {identifier}")



