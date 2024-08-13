'''
what this script contains:
- centering of the data
- normalization step for normalizing inside a cohort
- normalization step for normalizing per patient

'''

import pandas as pd
import numpy as np



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
    grouped = df_centered.groupby(['group', 'patient', 'bundle'])
    
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
    grouped = df_scaled.groupby(['group', 'patient'])

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






####################################################################################################
# Normalization step for normalizing across cohort
def minmax_scale_features_by_group(df, features):
    '''
    Scale the specified features within each group using Min-Max scaling.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the features to be scaled.
        features (List[str]): A list of feature names that need to be scaled.
    
    Returns:
        pd.DataFrame: A new DataFrame with the specified features scaled within each group.
    '''
    # Create a copy of the input DataFrame
    df_scaled = df.copy()
    
    # Group by 'group'
    grouped = df_scaled.groupby(['group'])
    
    # Iterate over each feature
    for feature in features:
        # Calculate the minimum and maximum values of the feature within each group
        feature_min = grouped[feature].transform('min')
        feature_max = grouped[feature].transform('max')
        
        # Scale the feature within each group using Min-Max scaling
        df_scaled[feature] = (df_scaled[feature] - feature_min) / (feature_max - feature_min)
    
    return df_scaled



# Normalization step for normalizing across cohort - end
####################################################################################################


####################################################################################################
# min-max scale features for each sample (bundle), regardless of the group
def minmax_scale_features_per_sample(df, features): ### change this to minmax_scale_features_by_patient
    '''
    Scale the specified features within each group and patient for each sample, using Min-Max scaling.

    Args:
        df (pd.DataFrame): The input DataFrame containing the features to be scaled.
        features (List[str]): A list of feature names that need to be scaled.

    Returns:
        pd.DataFrame: A new DataFrame with the specified features scaled within each group and patient.
    '''
    # Create a copy of the input DataFrame
    df_scaled = df.copy()

    # Group by 'group' and 'patient'
    grouped = df_scaled.groupby(['group', 'patient', 'bundle'])

    # Iterate over each feature
    for feature in features:
        # Calculate the minimum and maximum values of the feature within each group and patient
        feature_min = grouped[feature].transform('min')
        feature_max = grouped[feature].transform('max')

        # Scale the feature within each group and patient for each sample using Min-Max scaling
        df_scaled[feature] = (df_scaled[feature] - feature_min) / (feature_max - feature_min)

    return df_scaled

# min-max scale features for each sample, regardless of the group and patient - end
####################################################################################################

# ####################################################################################################
# # Normalization step for normalizing inside a cohort
############### this makes no sense because my data is not normally distributed ###############



# def normalize_features(df, features):
#     '''
#     Normalize the specified features within each group and patient.

#     Args:
#         df (pd.DataFrame): The input DataFrame containing the features to be normalized.
#         features (List[str]): A list of feature names that need to be normalized.

#     Returns:
#         pd.DataFrame: A new DataFrame with the specified features normalized within each group and patient.
#     '''
#     # Create a copy of the input DataFrame
#     df_normalized = df.copy()

#     # Group by 'group' and 'patient'
#     grouped = df_normalized.groupby(['group', 'patient'])

#     # Iterate over each feature
#     for feature in features:
#         # Calculate the mean and standard deviation of the feature within each group and patient
#         feature_mean = grouped[feature].transform('mean')
#         feature_std = grouped[feature].transform('std')

#         # Normalize the feature within each group and patient
#         df_normalized[feature] = (df_normalized[feature] - feature_mean) / feature_std

#     return df_normalized
# # Normalization step for normalizing inside a cohort - end
# ####################################################################################################
