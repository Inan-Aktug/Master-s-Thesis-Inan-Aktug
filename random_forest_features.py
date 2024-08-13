'''
this script is for the random forest feature selection

'''


import pandas as pd 
import numpy as np



def peak_velocity_positive(df, velocity_columns):
    '''
    Explanation:
    '''

    # these are the relevant columns
    vel_df = df.loc[:,['group', 'patient', 'bundle'] + velocity_columns]

    # grouped by group, patient, and bundle
    grouped_vel_data = vel_df.groupby(['group','patient','bundle'])

    peak_velocities_positive = grouped_vel_data[velocity_columns].max() ### max positive values for each group, patient, and bundle

    pos_peak_columns = ['pos_peak_' + col for col in velocity_columns] ### list comprehension instead of the following

    peak_velocities_positive.columns = pos_peak_columns

    peak_velocities_positive = peak_velocities_positive.reset_index()

    return peak_velocities_positive


def peak_velocity_negative(df, velocity_columns):
    '''
    Explanation:
    '''

    # these are the relevant columns
    vel_df = df.loc[:,['group', 'patient', 'bundle'] + velocity_columns]

    # grouped by group, patient, and bundle
    grouped_vel_data = vel_df.groupby(['group','patient','bundle'])

    peak_velocities_negative = grouped_vel_data[velocity_columns].min() ### max negative values for each group, patient, and bundle

    neg_peak_columns = ['neg_peak_' + col for col in velocity_columns] ### list comprehension instead of the following

    peak_velocities_negative.columns = neg_peak_columns

    peak_velocities_negative = peak_velocities_negative.reset_index()

    return peak_velocities_negative


def average_velocity(df, velocity_columns):
    '''
    Explanation:
    '''

    # these are the relevant columns
    vel_df = df.loc[:,['group', 'patient', 'bundle'] + velocity_columns]

    # grouped by group, patient, and bundle
    grouped_vel_data = vel_df.groupby(['group','patient','bundle'])

    average_velocities = grouped_vel_data[velocity_columns].mean() ### mean values for each group, patient, and bundle

    average_columns = ['average_' + col for col in velocity_columns] ### list comprehension for column names

    average_velocities.columns = average_columns

    average_velocities = average_velocities.reset_index()

    return average_velocities


def mean_speed(df, velocity_columns):
    '''
    Explanation:
    '''
    # these are the relevant columns
    vel_df = df.loc[:,['group', 'patient', 'bundle'] + velocity_columns]
    
    # grouped by group, patient, and bundle
    grouped_vel_data = vel_df.groupby(['group','patient','bundle'])
    
    speed_data = []
    
    for name, group in grouped_vel_data:
        speed_row = list(name)
        for column in velocity_columns:
            speed_values = np.abs(group[column])
            # print(speed_values)
            mean_speed = speed_values.mean()
            speed_row.append(mean_speed)
        speed_data.append(speed_row)
    
    columns = ['group', 'patient', 'bundle'] + ['mean_speed_' + col for col in velocity_columns]
    speed_df = pd.DataFrame(speed_data, columns=columns)
    
    return speed_df


def max_speed(df, velocity_columns):
    '''
    Explanation: This function calculates the maximum overall speed for each group, patient, and bundle
    from the given velocity columns in the DataFrame.
    '''
    # these are the relevant columns
    vel_df = df.loc[:,['group', 'patient', 'bundle'] + velocity_columns]
    
    # grouped by group, patient, and bundle
    grouped_vel_data = vel_df.groupby(['group','patient','bundle'])
    
    speed_data = []
    for name, group in grouped_vel_data:
        speed_row = list(name)
        speed_values = np.sqrt(np.sum(group[velocity_columns]**2, axis=1))
        max_speed = speed_values.max()
        speed_row.append(max_speed)
        speed_data.append(speed_row)
    
    columns = ['group', 'patient', 'bundle', 'max_speed']
    speed_df = pd.DataFrame(speed_data, columns=columns)
    
    return speed_df


def peak_acceleration_positive(df, acceleration_columns):
    '''
    Explanation:
    '''
    # these are the relevant columns
    acc_df = df.loc[:,['group', 'patient', 'bundle'] + acceleration_columns]
    
    # grouped by group, patient, and bundle
    grouped_acc_data = acc_df.groupby(['group','patient','bundle'])
    
    peak_accelerations_positive = grouped_acc_data[acceleration_columns].max() ### max positive values for each group, patient, and bundle
    
    pos_peak_columns = ['pos_peak_' + col for col in acceleration_columns] ### list comprehension instead of the following
    
    peak_accelerations_positive.columns = pos_peak_columns
    
    peak_accelerations_positive = peak_accelerations_positive.reset_index()
    
    return peak_accelerations_positive


def peak_acceleration_negative(df, acceleration_columns):
    '''
    Explanation:
    '''
    # these are the relevant columns
    acc_df = df.loc[:,['group', 'patient', 'bundle'] + acceleration_columns]
    
    # grouped by group, patient, and bundle
    grouped_acc_data = acc_df.groupby(['group','patient','bundle'])
    
    peak_accelerations_negative = grouped_acc_data[acceleration_columns].min() ### max negative values for each group, patient, and bundle
    
    neg_peak_columns = ['neg_peak_' + col for col in acceleration_columns] ### list comprehension instead of the following
    
    peak_accelerations_negative.columns = neg_peak_columns
    
    peak_accelerations_negative = peak_accelerations_negative.reset_index()
    
    return peak_accelerations_negative


def average_acceleration(df, acceleration_columns):
    '''
    Explanation:
    '''
    # these are the relevant columns
    acc_df = df.loc[:,['group', 'patient', 'bundle'] + acceleration_columns]
    
    # grouped by group, patient, and bundle
    grouped_acc_data = acc_df.groupby(['group','patient','bundle'])
    
    average_accelerations = grouped_acc_data[acceleration_columns].mean() ### mean values for each group, patient, and bundle
    
    average_columns = ['average_' + col for col in acceleration_columns] ### list comprehension for column names
    
    average_accelerations.columns = average_columns
    
    average_accelerations = average_accelerations.reset_index()
    
    return average_accelerations


def range_feature(df, keywords):
    '''
    Explanation:
    range columns should be keywords? -> test this
    '''
    range_df = df.loc[:, ['group', 'patient', 'bundle'] + keywords]
    grouped_range_data = range_df.groupby(['group', 'patient', 'bundle'])
    max_values = grouped_range_data[keywords].max()
    min_values = grouped_range_data[keywords].min()
    range_values = max_values - min_values
    range_columns = ['range_' + col for col in keywords]
    
    range_values.columns = range_columns
    range_df = range_values.reset_index()
    return range_df


def total_length(df, keyword):
    """
    Calculates the total length for a specified keyword across grouped combinations of 'group', 'patient', and 'bundle'.
    
    Parameters:
    - df: Pandas DataFrame containing the data.
    - keyword: Enter keywords[0] to take the first item of the list or just any keyword as string like 'tbo_y'. The keyword for which the total length is to be calculated.
    
    Returns:
    - A Pandas DataFrame with the groups, patients, bundles, and the total length for the specified keyword.
    """
    # Select only the relevant columns including the keyword
    len_df = df[['group', 'patient', 'bundle', keyword]]
    
    # Group by 'group', 'patient', and 'bundle', then count the non-null entries for the specified keyword
    grouped_len_data = len_df.groupby(['group', 'patient', 'bundle'])[keyword].count().reset_index(name='total_length')
    
    return grouped_len_data


def time_till_dip(df, keywords):
    '''
    Explanation:
    '''
    # these are the relevant columns
    time_till_dip_df = df.loc[:,['group', 'patient', 'bundle'] + keywords]
    
    # grouped by group, patient, and bundle
    grouped_time_till_dip_data = time_till_dip_df.groupby(['group','patient','bundle'])
    
    def min_index(group):
        group = group.reset_index(drop=True)  # Reset index within each group
        min_indices = group[keywords].idxmin()
        return pd.Series(min_indices.values, index=['time_dip_' + col for col in keywords])
    
    time_dip_indices = grouped_time_till_dip_data.apply(min_index)
    time_dip_indices = time_dip_indices.reset_index()
    
    return time_dip_indices


def time_till_peak(df, keywords):
    '''
    Explanation:
    '''
    # these are the relevant columns
    time_till_peak_df = df.loc[:,['group', 'patient', 'bundle'] + keywords]
    
    # grouped by group, patient, and bundle
    grouped_time_till_peak_data = time_till_peak_df.groupby(['group','patient','bundle'])
    
    def max_index(group):
        group = group.reset_index(drop=True)  # Reset index within each group
        min_indices = group[keywords].idxmax()
        return pd.Series(min_indices.values, index=['time_peak_' + col for col in keywords])
    
    time_peak_indices = grouped_time_till_peak_data.apply(max_index)
    time_peak_indices = time_peak_indices.reset_index()
    
    return time_peak_indices


def calculate_turning_points(df, keywords):
    '''
    Explanation:
    This function calculates the turning points in a list of values.
    It takes the current value and compares it to the next three values to determine if it is a turning point and to reduce noise.
    '''
    # DataFrame to store the turning points count for each combination
    turning_points_summary = []

    # Group by 'group', 'patient', 'bundle'
    grouped = df.groupby(['group', 'patient', 'bundle'])
    
    for name, group in grouped:
        # print(name) #
        # print(group) #
        result = {'group': name[0], 'patient': name[1], 'bundle': name[2]}
        for keyword in keywords:
            values = group[keyword].values
            
            increasing_count = 0
            decreasing_count = 0
            for i in range(1, len(values) - 3):
                if values[i-1] > values[i] < values[i+1] < values[i+2] < values[i+3]:
                    increasing_count += 1
                elif values[i-1] < values[i] > values[i+1] > values[i+2] > values[i+3]:
                    decreasing_count += 1
            result[f'{keyword}_increasing'] = increasing_count
            result[f'{keyword}_decreasing'] = decreasing_count
        turning_points_summary.append(result)

    # Convert the summary list to a DataFrame
    turning_points_df = pd.DataFrame(turning_points_summary)
    return turning_points_df