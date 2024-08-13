### Information
# this script iterates over all folders and json files and extracts the information of C1 to V2 of sample start and sample duration
# Control and RBD groups don't have C1 sample start, therefore name_sampleStart is the same as C1_sampleStart
###


import os
import json
import pandas as pd
import sys

def targeted_json_data_extraction(json_data, group, patient, bundle):
    """Extract only the specified fields from the JSON data."""
    row_data = {'group': group, 'patient': patient, 'bundle': bundle}
    for level in json_data.get('levels', []):
        items = level.get('items', [])
        for item in items:
            for label in item.get('labels', []):
                feature_name = label.get('name', '')
                feature_value = label.get('value', '')
                if feature_name in ['med_condition', 'condition']:
                    row_data[feature_name] = feature_value
                if feature_name in ['segment'] and feature_value in ['C1', 'V1', 'C2', 'V2']:
                    row_data[f'{feature_value}_sampleStart'] = item.get('sampleStart', '')
                    row_data[f'{feature_value}_sampleDur'] = item.get('sampleDur', '')
                if feature_name == 'word':
                    row_data['name'] = feature_value
                    row_data['name_sampleStart'] = item.get('sampleStart', '')
                    row_data['name_sampleDur'] = item.get('sampleDur', '')
    return row_data

def main():
    root_dir = 'D:\\00000_master_thesis_new\\all_data'  
    output_data = []
    
    for cohort in os.listdir(root_dir):
        cohort_path = os.path.join(root_dir, cohort)
        for patient in os.listdir(cohort_path):
            patient_path = os.path.join(cohort_path, patient)
            if not os.path.isdir(patient_path):
                continue
            for bundle in os.listdir(patient_path):
                bundle_path = os.path.join(patient_path, bundle)
                if not os.path.isdir(bundle_path):
                    continue
                for file in os.listdir(bundle_path):
                    if file.endswith('.json'):
                        json_path = os.path.join(bundle_path, file)
                        with open(json_path, 'r') as f:
                            json_data = json.load(f)
                        if all(len(level['items']) == 0 for level in json_data.get('levels', [])):
                            continue
                        extracted_data = targeted_json_data_extraction(json_data, cohort, patient, bundle)
                        output_data.append(extracted_data)
    
    df = pd.DataFrame(output_data)
    df.to_csv(r'D:\00000_master_thesis_new\annotations from json files\names_dataset_all_annotations.csv', index=False) # output
    df.to_excel(r'D:\00000_master_thesis_new\annotations from json files\names_dataset_all_annotations.xlsx', index=False)

if __name__ == '__main__':
    main()
