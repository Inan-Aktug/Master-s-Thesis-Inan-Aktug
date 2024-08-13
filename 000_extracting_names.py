import pandas as pd
import os
from pathlib import Path
from matplotlib import pyplot as plt
import time
import json
import numpy as np
import math
from tqdm import tqdm


import sys


freq48000 = 48000
freq250 = 250
output_dir = r'D:\00000_master_thesis_new\csv files'

# INPUT DIRECTORY
paths = [
         r'D:\00000_master_thesis_new\000_original_data\RBD_focus_emuDB',
         r'D:\00000_master_thesis_new\000_original_data\preoperative_focus_emuDB',
         r'D:\00000_master_thesis_new\000_original_data\postoperative_focus_emuDB',
         r'D:\00000_master_thesis_new\000_original_data\controls_focus_emuDB'
]

# n_v2 = {'RBD_focus_emuDB': 0, 'preoperative_focus_emuDB': 0, 'postoperative_focus_emuDB': 0, 'controls_focus_emuDB': 0}

for path in paths:
    print('--', path)
    start_time = time.time()
    current_group = os.path.basename(path)

    total_rows = 0
    n_words = 0
    big_df = []
    pathlist = Path(path).glob('**/*')

    for path in tqdm(pathlist, desc=f'Going through the recording folders of {current_group}'):
        path_in_str = str(path)

        med_condition = 'off'
        condition = None
        girl_name = None

        if path_in_str.endswith("annot.json"):
            current_path = os.path.dirname(path_in_str)# takes only the file path to be in the current folder
            path_list = current_path.split('\\')

            # print(current_bundle, current_session, current_path)
            # sys.exit()
            current_bundle = path_list[-1]
            current_session = path_list[-2]

            # sys.exit()
            list_files_in_folder = os.listdir(current_path) # checks for all files in the folder and puts them into a list
            check_all_files = [False] * 5
            for file in list_files_in_folder: # iterates over the list of files
                full_path_file = os.path.join(current_path, file)
                if file.endswith(".ulip.csv"):
                    df_ulip = pd.read_csv(full_path_file, index_col=0)
                    check_all_files[0] = True
                elif file.endswith(".ttip.csv"):
                    df_ttip = pd.read_csv(full_path_file, index_col=0)
                    check_all_files[1] = True
                elif file.endswith(".llip.csv"):
                    df_llip = pd.read_csv(full_path_file, index_col=0)
                    check_all_files[2] = True
                elif file.endswith(".tbo.csv"):
                    df_tbo = pd.read_csv(full_path_file, index_col=0)
                    check_all_files[3] = True

                elif file.endswith(".json"):
                    check_all_files[4] = True
                    annot = json.load(open(full_path_file, 'r')) #annot is the opened json file, glob function gives a list that's why [0]
                    annot_list = {}
                    for an in annot['levels']:
                        if an['type'] == 'EVENT':
                            for it in an['items']: #inside items (it)
                                t = it['samplePoint']
                                label = it['labels'][0]['value']
                                name = it['labels'][0]['name']
                                if name == 'med_condition':
                                    med_condition = label
                                elif name == 'condition':
                                    condition = label
                                else:
                                    annot_list[label] = [t, 0]
                        elif an['type'] == 'SEGMENT':
                            for it in an['items']:
                                t = it['sampleStart']
                                dt = it['sampleDur']
                                label = it['labels'][0]['value']
                                name = it['labels'][0]['name']
                                if name == 'word' and label != '':
                                    girl_name = label
                                annot_list[label] = [t, dt]
                        else:
                            print(an['type'])

            # print(annot_list.keys(), med_condition, condition, girl_name)

            # if med_condition is None or condition is None or girl_name is None:
            #     print(f'Problem with med_conditions {med_condition}, condition = {condition}, girlname =  {girl_name}')
            #     print(annot)
            #     continue

            if girl_name == '':
                print(f'Problem with med_conditions {med_condition}, condition = {condition}, girlname =  {girl_name}')
                continue

            if not np.all(check_all_files):
                print(f'Not all files found in {current_path}')
                continue

            # not using C0 and V0 from the annotations because can be missing in many files
            # start_ = min(annot_list.get(f'C1',    [np.inf])[0],  # min function takes the minimum of these 2 values for the starting point (calculated to seconds)
            #              annot_list.get(f'onsC1', [np.inf])[0])
            #
            # # end_ takes V2 starting point + its duration
            # end_ = max(annot_list.get(f'V2', [0])[0] + annot_list.get(f'V2', [0, 0])[1],  # same as above for end point
            #            annot_list.get(f'targV2', [0])[0])

            start_ = annot_list.get(f'onsC1', [np.inf])[0]

            # end_ takes V2 starting point + its duration
            end_ = annot_list.get(f'targV2', [0])[0]

            # if 'V2' in annot_list:
            #     n_v2[current_group] += 1

            if start_ == np.inf or end_ == 0:  # check if there is annotation -> if not it continues to the next iteration
                continue

            n_words += 1
            starting_row = start_ / freq48000 * freq250
            ending_row = end_ / freq48000 * freq250

            starting_row = math.floor(starting_row)  # starting row of current ttip, llip, ulip, tbo df
            ending_row = math.ceil(ending_row)  # ending row of current ttip, llip, ulip, tbo df

            # taking corresponding rows from the df's:
            needed_rows_ulip = df_ulip.iloc[starting_row:ending_row]
            needed_rows_llip = df_llip.iloc[starting_row:ending_row]
            needed_rows_ttip = df_ttip.iloc[starting_row:ending_row]
            needed_rows_tbo  = df_tbo.iloc[starting_row:ending_row]

            current_concat_df = pd.concat([needed_rows_llip, needed_rows_tbo, needed_rows_ulip, needed_rows_ttip],
                                          ignore_index=False, axis=1)

            current_concat_df['name'] = girl_name
            current_concat_df['med_condition'] = med_condition
            current_concat_df['condition'] = condition
            current_concat_df['group'] = current_group
            current_concat_df['session'] = current_session  # session is patient
            current_concat_df['bundle'] = current_bundle  # bundle is recorded sentence

            big_df.append(current_concat_df)

    big_df = pd.concat(big_df, ignore_index=True)
    print(big_df)
    print(n_words)
    
    csv_path = os.path.join(output_dir, f'{current_group}_C1toV2_annotated.csv')
    print(csv_path)

    ''' exporting as csv file '''
    big_df.to_csv(os.path.join(output_dir, f'{current_group}_C1toV2_annotated.csv'), index=False)

    print("--- %s seconds ---" % (time.time() - start_time))
print('stop')