'''
compared to the other version, this version is clean and deletes unnecessary code
- adjusts the paths for the new csv files

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random as rd
import sys
import scipy.fft

import random_forest_features as rff
import matplotlib.ticker as MaxNLocator

start_time = time.time()


# names dataset
# centered values
df_names_centered = pd.read_csv(r'D:\00000_master_thesis_new\csv files\names_dataset\centered\all_groups_C1toV2_annotated_centered.csv')
# min max per group
df_names_min_max_per_group = pd.read_csv(r'D:\00000_master_thesis_new\csv files\names_dataset\centered_and_min_max_per_group\all_groups_C1toV2_annotated_centered_min_max_per_group.csv')
# min max per patient
df_names_min_max_per_patient = pd.read_csv(r'D:\00000_master_thesis_new\csv files\names_dataset\centered_and_min_max_per_patient\all_groups_C1toV2_annotated_centered_min_max_per_patient.csv')







# Define the order of names and the keywords to calculate means for
name_order = ['lani', 'lena', 'lina', 'loni', 'luna', 
              'mali', 'mela', 'mila', 'moli', 'mula']
keywords = ['llip_x', 'llip_y',
            'tbo_x', 'tbo_y', 
            'ttip_x', 'ttip_y',
            'ulip_x', 'ulip_y']



df = df_names_centered

df_master = pd.DataFrame()


# Index(['patient', 'group', 'name', 'bundle', 'llip_x', 'llip_y', 'tbo_x',
#        'tbo_y', 'ttip_x', 'ttip_y', 'ulip_x', 'ulip_y', 'llip_xvel',
#        'llip_yvel', 'llip_tvel', 'llip_xacc', 'llip_yacc', 'llip_tacc',
#        'tbo_xvel', 'tbo_yvel', 'tbo_tvel', 'tbo_xacc', 'tbo_yacc', 'tbo_tacc',
#        'ulip_xvel', 'ulip_yvel', 'ulip_tvel', 'ulip_xacc', 'ulip_yacc',
#        'ulip_tacc', 'ttip_xvel', 'ttip_yvel', 'ttip_tvel', 'ttip_xacc',
#        'ttip_yacc', 'ttip_tacc']





########################################################################################################################
##### Counts the distribution of name or sentence in either dataset -> change sentence with name in the code below
# # print(df)
# result = (
#     df.groupby(['group', 'patient', 'bundle'])
#     .apply(lambda x: x['name'].iloc[0])
#     .reset_index(name='name')
# )

# print(result)
# name_count = result['name'].value_counts().reset_index()
# print(name_count)
# sys.exit()
##### part end
########################################################################################################################




### done
## PEAK VELOCITY calculation
# all velocities in current df
vel_df = df.loc[:,['group', 'patient', 'bundle', 
                    'tbo_yvel', 'tbo_xvel', 'tbo_tvel',
                    'ttip_yvel', 'ttip_xvel', 'ttip_tvel',
                    'ulip_yvel', 'ulip_xvel', 'ulip_tvel',
                    'llip_yvel', 'llip_xvel', 'llip_tvel'
]] 

all_velocity = ['tbo_yvel', 'tbo_xvel', 'tbo_tvel',
                    'ttip_yvel', 'ttip_xvel', 'ttip_tvel',
                    'ulip_yvel', 'ulip_xvel', 'ulip_tvel',
                    'llip_yvel', 'llip_xvel', 'llip_tvel']


grouped_vel_data = vel_df.groupby(['group','patient','bundle'])
# print(grouped_vel_data)

peak_velocities_positive = grouped_vel_data[all_velocity].max()
peak_velocities_negative = grouped_vel_data[all_velocity].min()



# print('vel_df')
# print(vel_df)
# print('peak velocities')
# print(peak_velocities)

df_master = peak_velocities_positive
new_column_names = ['pos_peak_' + col for col in all_velocity] ### list comprehension instead of the following
# new_column_names = ['peak_tbo_yvel', 'peak_tbo_xvel', 'peak_tbo_tvel',
#                     'peak_ttip_yvel', 'peak_ttip_xvel', 'peak_ttip_tvel',
#                     'peak_ulip_yvel', 'peak_ulip_xvel', 'peak_ulip_tvel',
#                     'peak_llip_yvel', 'peak_llip_xvel', 'peak_llip_tvel']


df_master.columns = new_column_names
# print('df master')
# print(df_master)












# # #### TOTAL LENGTH
# total_len_df = df.loc[:,['group', 'patient', 'bundle', 'tbo_x']]

# # Group by 'group', 'patient', and 'bundle' only, as you want the count of 'tbo_x' rows for each bundle
# grouped = total_len_df.groupby(['group', 'patient', 'bundle']).size().reset_index(name='total_length')

# # print(grouped)
# df_master = pd.merge(df_master, grouped, on=['group', 'patient', 'bundle'], how='left') ############################################# merges the range_df to the master df
# # print(list(df_master.columns))
# # print(df_master)




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


time_till_peak_test = time_till_peak(df, keywords)
# print(time_till_peak_test)

print(df)







grouped_df = df.groupby(['group', 'patient', 'bundle'])


def calculate_turning_points(values):
    increasing_indices = []
    decreasing_indices = []

    for i in range(1, len(values) - 3):
        if values[i-1] > values[i] < values[i+1] < values[i+2] < values[i+3]:
            # print('increasing')
            # rounded_values = [round(values[j], 3) for j in range(i-1, i+4)]
            # differences = [round(rounded_values[k+1] - rounded_values[k], 3) for k in range(4)]
            # print(rounded_values)
            # print('Differences:', differences)
            increasing_indices.append(i)
        elif values[i-1] < values[i] > values[i+1] > values[i+2] > values[i+3]:
            decreasing_indices.append(i)

    return increasing_indices, decreasing_indices

turning_points_data = []

for name, group in grouped_df:
    group_data = {'group': name[0], 'patient': name[1], 'bundle': name[2]}

    for keyword in keywords:
        values = group[keyword].values
        increasing_indices, decreasing_indices = calculate_turning_points(values)
        group_data[f'tp_increasing_count_{keyword}'] = len(increasing_indices)
        group_data[f'tp_decreasing_count_{keyword}'] = len(decreasing_indices)

    turning_points_data.append(group_data)

turning_points_df = pd.DataFrame(turning_points_data)
print(turning_points_df)








#############################################################################################################################
# this is for a single recording / bundle to visualize the turning points
# Get the specific bundle DataFrame
bundle_df = grouped_df.get_group(('preoperative_focus_emuDB', 'WOKL65_ses', '0133_bndl'))

# Create subplots for each keyword
num_keywords = len(keywords)
fig, axes = plt.subplots(num_keywords, 1, figsize=(10, 4*num_keywords), sharex=True)

# Iterate over each keyword
for i, keyword in enumerate(keywords):
    # Get the values for the current keyword in the current bundle
    values = bundle_df[keyword].values
    
    # Calculate the increasing and decreasing turning points for the current keyword
    increasing_indices, decreasing_indices = calculate_turning_points(values)
    # print(increasing_indices)
    # sys.exit()
    # Plot the keyword values
    axes[i].plot(bundle_df.index, values, label=keyword)
    
    # Plot the increasing turning points as vertical lines
    for idx in increasing_indices:
        axes[i].axvline(x=bundle_df.index[idx], color='green', linestyle='--', alpha=0.7)
    
    # Plot the decreasing turning points as vertical lines
    for idx in decreasing_indices:
        axes[i].axvline(x=bundle_df.index[idx], color='red', linestyle='--', alpha=0.7)
    
    # Set the title and labels for the current subplot
    axes[i].set_title(f'Turning Points for {keyword}')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Value')
    axes[i].legend()

# Adjust the spacing between subplots
plt.tight_layout()

# # Display the plot
# plt.show()
# sys.exit()
#############################################################################################################################




# import matplotlib.pyplot as plt

# y_keywords = ['llip_y', 'tbo_y', 'ttip_y', 'ulip_y']

# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# axes = axes.flatten()

# for i, group in enumerate(turning_points_df['group'].unique()):
#     group_data = turning_points_df[turning_points_df['group'] == group]
    
#     boxplot_data = [group_data[f'{keyword}_decreasing_count'] for keyword in y_keywords]
    
#     axes[i].boxplot(boxplot_data)
#     axes[i].set_title(f'Group: {group}')
#     axes[i].set_xticklabels(y_keywords, rotation=45)
#     axes[i].set_ylabel('Decreasing Turning Points')

# plt.tight_layout()
# plt.show()








# print(df.columns)
# sys.exit()





##########################################################################################################################################################
import matplotlib.pyplot as plt

# Create a mapping dictionary from the unique combination of 'group', 'patient', 'bundle' to 'name'
name_mapping = df.groupby(['group', 'patient', 'bundle'])['name'].first().to_dict()

# Map the 'name' values to the 'turning_points_df' based on the unique combination
turning_points_df['name'] = turning_points_df.apply(lambda row: name_mapping[(row['group'], row['patient'], row['bundle'])], axis=1)

y_keywords = ['llip_y', 'tbo_y', 'ttip_y', 'ulip_y']
names = ['mela', 'lena']



fig, axes = plt.subplots(len(names), len(turning_points_df['group'].unique()), figsize=(16, 6 * len(names)), sharey=True)

for i, name in enumerate(names):
    name_data = turning_points_df[turning_points_df['name'] == name]
    
    for j, group in enumerate(turning_points_df['group'].unique()):
        group_data = name_data[name_data['group'] == group]
        
        if not group_data.empty:
            boxplot_data = [group_data[f'tp_decreasing_count_{keyword}'] for keyword in y_keywords]
            
            ax = axes[i, j]
            ax.boxplot(boxplot_data)
            ax.set_title(f'Name: {name}, Group: {group}')
            ax.set_xticklabels(y_keywords, rotation=45)
            ax.set_ylabel('Decreasing Turning Points')
        else:
            ax = axes[i, j]
            ax.set_visible(False)

# plt.tight_layout()
# plt.show()
##########################################################################################################################################################









##########################################################################################################################################################
#best version so far
# Unique groups in your dataset for comparison
unique_groups = turning_points_df['group'].unique()

# Loop through each name to create a separate plot
for name in name_order:
    # Set up a figure for each name with subplots for each keyword
    fig, axes = plt.subplots(2, 4, figsize=(20, 10)) # Adjust figsize as needed
    fig.suptitle(f'Analysis for {name}', fontsize=16)
    
    for i, keyword in enumerate(keywords):
        # Locate subplot position based on keyword index
        ax = axes[i // 4, i % 4]
        
        # Data preparation for boxplot
        boxplot_data = []
        for group in unique_groups:
            # Filter data for current name, keyword, and group
            data = turning_points_df[(turning_points_df['name'] == name) &
                                     (turning_points_df['group'] == group)][f'tp_decreasing_count_{keyword}']
            boxplot_data.append(data)
        
        # Plotting the boxplot for the current keyword
        ax.boxplot(boxplot_data, labels=unique_groups)
        ax.set_title(f'{keyword}', fontsize=12)
        ax.set_ylabel('Counts', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        #q: how can i set the  y-axis limit to the max value of the actual boxplot data?


        ax.set_ylim(bottom = -0.5)
        # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Adjust layout to not overlap subplots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Show/save figure
    # plt.show()
    # To save the figure, uncomment the following line and provide the desired filename
    # plt.savefig(f'{name}_analysis.png')
##########################################################################################################################################################












##########################################################################################################################################################
# Define the desired order of your groups
group_order = ['controls_focus_emuDB', 'RBD_focus_emuDB', 'preoperative_focus_emuDB', 'postoperative_focus_emuDB']

# Make sure unique_groups is sorted according to group_order
unique_groups_sorted = sorted(unique_groups, key=lambda x: group_order.index(x))

# Loop through each name to create a separate plot
for name in name_order:
    # Set up a figure for each name with subplots for each keyword
    fig, axes = plt.subplots(2, 4, figsize=(20, 10)) # Adjust figsize as needed
    fig.suptitle(f'Analysis for {name}', fontsize=16)
    
    # Initialize a variable to store the maximum value across all boxplots for the current name
    max_value = 0
    
    for i, keyword in enumerate(keywords):
        # Locate subplot position based on keyword index
        ax = axes[i // 4, i % 4]
        
        # Data preparation for boxplot
        boxplot_data = []
        for group in unique_groups_sorted: # Use the sorted group list
            # Filter data for current name, keyword, and group
            data = turning_points_df[(turning_points_df['name'] == name) &
                                     (turning_points_df['group'] == group)][f'tp_decreasing_count_{keyword}']
            boxplot_data.append(data)
            
            # Update max_value if necessary
            if not data.empty:
                max_value = max(max_value, data.max())
        
        # Plotting the boxplot for the current keyword
        ax.boxplot(boxplot_data, labels=unique_groups_sorted) # Use the sorted group labels
        ax.set_title(f'{keyword}', fontsize=12)
        ax.set_ylabel('Counts', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
    
    # Set the y-axis limit to the max value of the actual boxplot data
    # Adding some margin for better visualization
    ax.set_ylim(bottom=-0.5, top=max_value + (max_value * 0.1))
    
    # Adjust layout to not overlap subplots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Show/save figure
    # plt.show()
    # To save the figure, uncomment the following line and provide the desired filename
    # plt.savefig(f'{name}_analysis.png')
##########################################################################################################################################################





#### scaling issue
# make boxplots start from 0
# use y axis stepsize of 1





##########################################################################################################################################################
#### scaling issue
# make boxplots start from 0
# use y axis stepsize of 1

# Create a mapping dictionary from the unique combination of 'group', 'patient', 'bundle' to 'name'
name_mapping = df.groupby(['group', 'patient', 'bundle'])['name'].first().to_dict()

# Map the 'name' values to the 'turning_points_df' based on the unique combination
turning_points_df['name'] = turning_points_df.apply(lambda row: name_mapping[(row['group'], row['patient'], row['bundle'])], axis=1)

# Unique groups in your dataset for comparison
unique_groups = turning_points_df['group'].unique()

# Loop through each name to create a separate plot
for name in name_order:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))  # Adjust figsize as needed
    fig.suptitle(f'Analysis for {name}', fontsize=16)
    
    for i, keyword in enumerate(keywords):
        ax = axes[i // 4, i % 4]  # Locate subplot position
        
        boxplot_data = []
        max_value = 0  # Initialize max value for this keyword
        for group in unique_groups:
            data = turning_points_df[(turning_points_df['name'] == name) &
                                     (turning_points_df['group'] == group)][f'{keyword}_decreasing_count']
            boxplot_data.append(data)
            
            # Update max_value if current group's max is higher
            current_max = data.max()
            if current_max > max_value:
                max_value = current_max
        
        # Plot the boxplot
        ax.boxplot(boxplot_data, labels=unique_groups)
        ax.set_title(f'{keyword}', fontsize=12)
        ax.set_ylabel('Counts', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        
        # Set y-axis limit with some buffer, e.g., 10% more than max_value
        ax.set_ylim(bottom=-0.5, top=max_value + 0.1 * max_value)

        # Ensure y-axis ticks increment by 1
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))


    # Adjust layout to not overlap subplots
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Show/save figure
    plt.show()
    # To save the figure, uncomment the following line and provide the desired filename
    # plt.savefig(f'{name}_analysis.png')
##########################################################################################################################################################








































































# sys.exit()



















#### PEAK ACCELERATION
peak_acc_df = df.loc[:,['group', 'patient', 'bundle',
                    'tbo_xacc', 'tbo_yacc', 'tbo_tacc', 'ttip_xacc', 'ttip_yacc', 'ttip_tacc', 'ulip_xacc', 'ulip_yacc', 'ulip_tacc', 'llip_xacc', 'llip_yacc', 'llip_tacc']]
# peak_+list[i]+'acc'
all_peak_acc=[ 'tbo_xacc', 'tbo_yacc', 'tbo_tacc', 'ttip_xacc', 'ttip_yacc', 'ttip_tacc', 'ulip_xacc', 'ulip_yacc', 'ulip_tacc', 'llip_xacc', 'llip_yacc', 'llip_tacc']
peak_acc_colnames = ['peak_' + col for col in all_peak_acc] ### list comprehension instead of the following

grouped_peak_acc = peak_acc_df.groupby(['group', 'patient', 'bundle'])

peak_acc_values = grouped_peak_acc[all_peak_acc].max() # all the peak acc values in a df

peak_acc_values.columns = peak_acc_colnames
# print(peak_acc_values)

df_master = pd.merge(df_master, peak_acc_values, on=['group', 'patient', 'bundle'], how='left') ############################################# merges the range_df to the master df
# print(df_master)




# #### NAME lina, luna ...
name_df = df.loc[:,['group', 'patient', 'bundle', 'name']]
name_cols = ['name']
grouped_name = name_df.groupby(['group', 'patient', 'bundle']).agg({'name': lambda x: ','.join(x.unique())}).reset_index()


def get_name_no(name):
# ['lani' 'mali' 'loni' 'moli' 'mula' 'lina' 'mela' 'luna' 'mila' 'lena']
    name_mapping = {'lani': 0, 'lena': 1, 'lina': 2, 'loni': 3, 'luna': 4,
                    'mali': 5, 'mela': 6, 'mila': 7, 'moli': 8, 'mula': 9}
    return name_mapping[name]

# # def get_vowel(name):
# #     return name[1]

# # def get_vowel_no(vowel):
# #     vowel_mapping = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
# #     return vowel_mapping[vowel]

grouped_name['name_no'] = grouped_name['name'].apply(get_name_no)
# # grouped_name['vowel'] = grouped_name['name'].apply(get_vowel)
# # grouped_name['vowel_no'] = grouped_name['vowel'].apply(get_vowel_no)

# print(grouped_name)
grouped_name = grouped_name.drop(columns=['name'])

# print(grouped_name)
# sys.exit()
# all_names = grouped_name[name_cols]
# print(grouped_name)
# print(all_names)

df_master = pd.merge(df_master, grouped_name, on=['group', 'patient', 'bundle'], how='left') ############################################# merges the range_df to the master df
# print(df_master)
# print(df_master.columns)






############ Segmenting features into n segments dynamically
### functions
def calculating_range(segment):
    return segment.max() - segment.min()

def calculating_peak_velocity(segment):
    return segment.max()

def calculating_average_velocity(segment):
    return segment.mean()

def calculating_peak_acceleration(segment):
    return segment.max()

def calculating_average_speed(segment):
    return segment.mean()

def calculating_maximum_speed(segment):
    return segment.max()

    



segmentation_df = df.loc[:,['group', 'patient', 'bundle',
                   'llip_x', 'llip_y', 'tbo_x', 'tbo_y', 'ttip_x', 'ttip_y', 'ulip_x', 'ulip_y', 
                   'llip_xvel', 'llip_yvel', 'llip_tvel', 'llip_xacc', 'llip_yacc', 'llip_tacc',
                   'tbo_xvel', 'tbo_yvel', 'tbo_tvel', 'tbo_xacc', 'tbo_yacc', 'tbo_tacc',
                   'ulip_xvel', 'ulip_yvel', 'ulip_tvel', 'ulip_xacc', 'ulip_yacc', 'ulip_tacc', 
                   'ttip_xvel', 'ttip_yvel', 'ttip_tvel', 'ttip_xacc', 'ttip_yacc', 'ttip_tacc']
                   ]


# calculating speed values from acceleration
speed_df = pd.DataFrame()
# speed_df['group'] = segmentation_df['group']
# speed_df['patient'] = segmentation_df['patient']
# speed_df['bundle'] = segmentation_df['bundle']
speed_df['speed_llip'] = np.sqrt(segmentation_df['llip_xvel']**2 + segmentation_df['llip_yvel']**2)
speed_df['speed_ulip'] = np.sqrt(segmentation_df['ulip_xvel']**2 + segmentation_df['ulip_yvel']**2)
speed_df['speed_tbo'] = np.sqrt(segmentation_df['tbo_xvel']**2 + segmentation_df['tbo_yvel']**2)
speed_df['speed_ttip'] = np.sqrt(segmentation_df['ttip_xvel']**2 + segmentation_df['ttip_yvel']**2)

segmentation_df = pd.concat([segmentation_df, speed_df], axis=1)



### dictionaries:
keywords_xy_dict = {'keyword': ['llip_x', 'llip_y', 'tbo_x', 'tbo_y', 'ttip_x', 'ttip_y', 'ulip_x', 'ulip_y']}

acceleration_dict = {'acc': ['llip_xacc', 'llip_yacc', 'llip_tacc',
                        'tbo_xacc', 'tbo_yacc', 'tbo_tacc',
                        'ulip_xacc', 'ulip_yacc', 'ulip_tacc',
                        'ttip_xacc', 'ttip_yacc', 'ttip_tacc']
                }

velocity_dict = {'vel': ['llip_xvel', 'llip_yvel', 'llip_tvel',
                    'tbo_xvel', 'tbo_yvel', 'tbo_tvel',
                    'ulip_xvel', 'ulip_yvel', 'ulip_tvel',
                    'ttip_xvel', 'ttip_yvel', 'ttip_tvel',]
            }   

speed_dict = {'speed' : ['speed_llip', 'speed_ulip', 'speed_tbo', 'speed_ttip']}




print(segmentation_df)
segmented_features_df = pd.DataFrame()
segmented_features_df = df_master.loc[:,['group', 'patient', 'bundle']]

print(segmented_features_df)
test_df = pd.DataFrame()


segments = 4
segmented_data = []
test_counter = 0

grouped_segmentation_df = segmentation_df.groupby(['group', 'patient', 'bundle'])
counts = grouped_segmentation_df.size().reset_index(name='row_count')


index_of_segment_list = 0

for index, row in counts.iterrows():
    current_number_of_rows = row['row_count']
    # print(type(total_rows))
    # print(row)
    current_group = row['group']
    current_patient = row['patient']
    current_bundle = row['bundle']
    # print(current_number_of_rows, current_group, current_patient, current_bundle)
    rows_per_segment, remainder = divmod(current_number_of_rows, segments)
    # segment_sizes is a list of the length of segments I have and the segment_size as element
    segment_sizes = [rows_per_segment + 1 if i < remainder else rows_per_segment for i in range(segments)]


    ### current unique group patient bundle 
    current_unique_gpb = segmentation_df[
    (segmentation_df['group'] == current_group) &
    (segmentation_df['patient'] == current_patient) &
    (segmentation_df['bundle'] == current_bundle)]



    if index_of_segment_list >= segments:
        index_of_segment_list = 0
    
    start_idx = 0

    segments_list = []
    for i, size in enumerate(segment_sizes):
        current_start = start_idx
        current_end = start_idx + segment_sizes[index_of_segment_list]
        index_of_segment_list += 1

        current_segment = current_unique_gpb[current_start:current_end]
        # print(current_segment)
        # row_dict = {
        #     'group' : current_group,
        #     'patient': current_patient,
        #     'bundle': current_bundle
        # }


        # print(current_segment)
        for key, columns in keywords_xy_dict.items():
            for column in columns:
                keyword_column_name = f'seg_{i+1}_range_{column}'
                
                current_calculation = calculating_range(current_segment[column])
                # print(current_calculation)
                # row_dict[keyword_column_name] = current_calculation
                # print(row_dict)
                target_row = (segmented_features_df['group'] == current_group) & (segmented_features_df['patient'] == current_patient) & (segmented_features_df['bundle'] == current_bundle)
                segmented_features_df.loc[target_row, keyword_column_name] = current_calculation


        for acc, columns in acceleration_dict.items():
            for column in columns:
                acceleration_column_name = f'seg_{i+1}_peak_acc_{column}'

                current_calculation = calculating_peak_acceleration(current_segment[column])
                target_row = (segmented_features_df['group'] == current_group) & (segmented_features_df['patient'] == current_patient) & (segmented_features_df['bundle'] == current_bundle)
                segmented_features_df.loc[target_row, acceleration_column_name] = current_calculation
        


        for vel, columns in velocity_dict.items():
            for column in columns:
                velocity_column_name = f'seg_{i+1}_peak_vel_{column}'

                current_calculation = calculating_peak_velocity(current_segment[column])
                target_row = (segmented_features_df['group'] == current_group) & (segmented_features_df['patient'] == current_patient) & (segmented_features_df['bundle'] == current_bundle)
                segmented_features_df.loc[target_row, velocity_column_name] = current_calculation
        
        for vel, columns in velocity_dict.items():
            for column in columns:
                velocity_column_name = f'seg_{i+1}_average_vel_{column}'

                current_calculation = calculating_average_velocity(current_segment[column])
                target_row = (segmented_features_df['group'] == current_group) & (segmented_features_df['patient'] == current_patient) & (segmented_features_df['bundle'] == current_bundle)
                segmented_features_df.loc[target_row, velocity_column_name] = current_calculation
        
        for speed, columns in speed_dict.items():
            for column in columns:
                speed_column_name = f'seg_{i+1}_average_speed_{column}'

                current_calculation = calculating_average_speed(current_segment[column])
                target_row = (segmented_features_df['group'] == current_group) & (segmented_features_df['patient'] == current_patient) & (segmented_features_df['bundle'] == current_bundle)
                segmented_features_df.loc[target_row, speed_column_name] = current_calculation

        
        for speed, columns in speed_dict.items():
            for column in columns:
                speed_column_name = f'seg_{i+1}_max_speed_{column}'

                current_calculation = calculating_maximum_speed(current_segment[column])
                target_row = (segmented_features_df['group'] == current_group) & (segmented_features_df['patient'] == current_patient) & (segmented_features_df['bundle'] == current_bundle)
                segmented_features_df.loc[target_row, speed_column_name] = current_calculation



        # test_df=pd.DataFrame([row_dict])
        # counts = pd.concat([counts, test_df])

            
                # test_df['group'] = current_group
                # test_df['patient'] = current_patient
                # test_df['bundle'] = current_bundle
                # test_df[keyword_column_name] = current_calculation
        # print(counts)


segmented_features_df = pd.merge(segmented_features_df, grouped_name, on=['group', 'patient', 'bundle'], how='left')
print(segmented_features_df)

# segmented_features_df.to_csv(r'C:\Users\iaktug\Desktop\Master_thesis\csv files\segmented_features\sentences_dataset\with_names_number\sentences_dataset_4_segmented_features_names_number.csv', index=False)
# segmented_features_df.to_excel(r'C:\Users\iaktug\Desktop\Master_thesis\csv files\segmented_features\sentences_dataset\with_names_number\sentences_dataset_4_segmented_features_names_number.xlsx', index=False)





print("--- %s seconds ---" % (time.time() - start_time))
sys.exit()
############ End - Segmenting features into n segments dynamically











##### USEFUL INFORMATION: ###########################################################################################################
'''segments = 4 # You can change this to 10 or any other number

for index, row in counts.iterrows():
    total_rows = row['row_count']
    rows_per_segment, remainder = divmod(total_rows, segments)
    segment_sizes = [rows_per_segment + 1 if i < remainder else rows_per_segment for i in range(segments)]

    start_idx = 0
    for i, size in enumerate(segment_sizes):
        current_segment = segmentation_df.iloc[start_idx:start_idx + size]
        start_idx += size

        # Calculate features using the dictionaries and add to the DataFrame
        for key, columns in keywords_xy_dict.items():
            for column in columns:
                new_column_name = f'seg_{i+1}_range_{column}'
                segmentation_df.loc[start_idx:start_idx + size, new_column_name] = calculate_range(current_segment[column])

        # Similarly, you can add calculations for acceleration and velocity using the other dictionaries
'''
############################################################################################################################################################################################################













############ (Fast) Fourier transformation (FFT)
#        'patient', 'group', 'name', 'bundle', 
#        'llip_x', 'llip_y', 'tbo_x', 'tbo_y', 'ttip_x', 'ttip_y', 'ulip_x', 'ulip_y', 
#        'llip_xvel', 'llip_yvel', 'llip_tvel', 'llip_xacc', 'llip_yacc', 'llip_tacc',
#        'tbo_xvel', 'tbo_yvel', 'tbo_tvel', 'tbo_xacc', 'tbo_yacc', 'tbo_tacc',
#        'ulip_xvel', 'ulip_yvel', 'ulip_tvel', 'ulip_xacc', 'ulip_yacc', 'ulip_tacc', 
#        'ttip_xvel', 'ttip_yvel', 'ttip_tvel', 'ttip_xacc', 'ttip_yacc', 'ttip_tacc']

fft_df = df.loc[:,['group', 'patient', 'bundle',
                   'llip_x', 'llip_y', 'tbo_x', 'tbo_y', 'ttip_x', 'ttip_y', 'ulip_x', 'ulip_y', 
                   'llip_xvel', 'llip_yvel', 'llip_tvel', 'llip_xacc', 'llip_yacc', 'llip_tacc',
                   'tbo_xvel', 'tbo_yvel', 'tbo_tvel', 'tbo_xacc', 'tbo_yacc', 'tbo_tacc',
                   'ulip_xvel', 'ulip_yvel', 'ulip_tvel', 'ulip_xacc', 'ulip_yacc', 'ulip_tacc', 
                   'ttip_xvel', 'ttip_yvel', 'ttip_tvel', 'ttip_xacc', 'ttip_yacc', 'ttip_tacc']
                   ]


### dictionaries:
keywords_xy_dict = {'keyword': ['llip_x', 'llip_y', 'tbo_x', 'tbo_y', 'ttip_x', 'ttip_y', 'ulip_x', 'ulip_y']}

acceleration_dict = {'acc': ['llip_xacc', 'llip_yacc', 'llip_tacc',
                        'tbo_xacc', 'tbo_yacc', 'tbo_tacc',
                        'ulip_xacc', 'ulip_yacc', 'ulip_tacc',
                        'ttip_xacc', 'ttip_yacc', 'ttip_tacc']
                }
velocity_dict = {'vel': ['llip_xvel', 'llip_yvel', 'llip_tvel',
                    'tbo_xvel', 'tbo_yvel', 'tbo_tvel',
                    'ulip_xvel', 'ulip_yvel', 'ulip_tvel',
                    'ttip_xvel', 'ttip_yvel', 'ttip_tvel',]
            }   

# print(keywords_xy_dict['keyword'][0])


## implementing segments -> segment size / len(patient bundle)
segments = 4
segmented_data = []

grouped_fft = fft_df.groupby(['group', 'patient', 'bundle'])
counts = grouped_fft.size().reset_index(name='row_count')

# print(counts)


# Iterate through the DataFrame to calculate segment sizes and create new columns
for index, row in counts.iterrows():
    total_rows = row['row_count']
    rows_per_segment, remainder = divmod(total_rows, segments)
    segment_sizes = [rows_per_segment + 1 if i < remainder else rows_per_segment for i in range(segments)]
    # print(segment_sizes)
    # print(index)
    # print(row)


    start_idx = 0
    # Create the new columns with segment sizes
    for i, size in enumerate(segment_sizes):
        current_segment = fft_df.iloc[start_idx:start_idx + size] ### takes the current segment for calculations
        start_idx += size # adds the size to the start index

############### CONTINUE HERE

        print(current_segment)
        
        fft_llip_x = np.fft.fft(current_segment['llip_x'])
        fft_llip_y = np.fft.fft(current_segment['llip_y'])

        fft_llip_x = scipy.fft.fft(current_segment['llip_x'].values)
        magnitude_test = np.abs(fft_llip_x)
        # print(magnitude_test)
        # sys.exit()
        print(fft_llip_x)


##################################################################
# do this calculation not for segments but for the whole name (80-130 rows)
        # T = 1 / 250 # frequency
        # N = 80 #
                
        # # print(xf)
        # xf = scipy.fft.fftfreq(N, T)
        # fft_llip_x = scipy.fft.fft(current_segment['llip_x'].values, n = N)
        # fft_llip_x2 = scipy.fft.fft(current_segment['llip_x'].values, n = 2*N)
        # xf2 = scipy.fft.fftfreq(2*N, T)


        # print(len(xf))
        # print(len(xf2))         ### this
        # print(len(fft_llip_x))
        # print(len(fft_llip_x2)) ### this
        # print(xf)

        # plt.plot(xf, abs(fft_llip_x))
        # plt.plot(xf2, abs(fft_llip_x2))
        # plt.show()
        # sys.exit()
##################################################################
##############


        column_name_prefix = f'seg_{i + 1}_'

        counts.at[index, column_name_prefix + 'tbo_x'] = size
        # print(i)
        print(size, 'size')



# sys.exit()
total_rows = 133

# rows_per_segment, remainder = divmod(total_rows, segments)

# print(rows_per_segment, remainder)
# segment_sizes = [rows_per_segment + 1 if i < remainder else rows_per_segment for i in range(segments)]
# print(segment_sizes)
############ End of (Fast) Fourier transformation (FFT)




# print(df_master.columns)
print(df_master)
### export
# df_master.to_csv(r'C:\Users\iaktug\Desktop\Master_thesis\csv files\rerun_of_second_with_new_graphs\centered_values_sentences_dataset\features_for_sentences_dataset_only_centered.csv') ###here

print("--- %s seconds ---" % (time.time() - start_time))