import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import seaborn as sns
import random
import sys
from collections import Counter
import os
import helpful_modules as hm
from datetime import datetime



from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# from imblearn.ensemble import BalancedRandomForestClassifier
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import StratifiedGroupKFold




'''
NOTE:

These are the datasets which are filtered for group3 according to paper replication dataset
The code for filtereing is 


SCJUE59_ses patient in control group and not mild PD

'''
### med condition annotation
path_med_condition_filter = r'D:\00000_master_thesis_new\csv files\json annotation'
file_med_condition_filter = r'json_all_information.csv'
file_and_path_med_condition_filter = os.path.join(path_med_condition_filter, file_med_condition_filter)
med_condition_filter_df = pd.read_csv(file_and_path_med_condition_filter)

# print(med_condition_filter_df)



### paper replication dataset - 2068 datapoints (rows)
path_paper_rep = r'D:\00000_master_thesis_new\csv files\rf_features\paper_replication_dataset\features_like_other_datasets'
file_paper_rep = r'c1_v1_plus_duration_filtered_datapoints_paper_replication_keypoint_values_first_syllable_centered_min_max_per_patient_rf_features.csv'
file_and_path_paper_rep = os.path.join(path_paper_rep, file_paper_rep)
paper_rep_df = pd.read_csv(file_and_path_paper_rep)


### names dataset - 3489 datapoints (rows) 
path_names_dataset = r'D:\00000_master_thesis_new\csv files\rf_features\names_dataset\csv'
file_names_dataset = r'all_groups_C1toV2_annotated_centered_min_max_per_group_rf_features.csv'
file_and_path_names_dataset = os.path.join(path_names_dataset, file_names_dataset)
names_dataset_df = pd.read_csv(file_and_path_names_dataset)


### sentences dataset - 3697 datapoints (rows) 
path_sentences_dataset = r'D:\00000_master_thesis_new\csv files\rf_features\sentences_datasets\csv'
file_sentences_dataset = r'all_groups_1.2_3.6_filtered_sentences_master_file_no_unnamed_col_centered_min_max_per_patient_rf_features.csv'
file_and_path_sentences_dataset = os.path.join(path_sentences_dataset, file_sentences_dataset)
sentences_dataset_df = pd.read_csv(file_and_path_sentences_dataset)


# all_dataframes = [paper_rep_df, names_dataset_df, sentences_dataset_df]

# print(names_dataset_df.columns)
# print(sentences_dataset_df)

# print(sentences_dataset_df['group3'].value_counts())



######### HERE WHICH DATASET IS USED FOR TRAINING AND TESTING ####################
current_df = sentences_dataset_df
file_and_path = file_and_path_sentences_dataset


#adding med condition to dataframe if 
print(current_df.columns)

if 'med_condition' not in current_df.columns:
    current_df = pd.merge(current_df, med_condition_filter_df[['group', 'patient', 'bundle', 'med_condition']], 
                          on=['group', 'patient', 'bundle'], 
                          how='left')
    
print(current_df.columns)
print(current_df['med_condition'].value_counts())
result = current_df[current_df['med_condition'].notna()].groupby(['group', 'med_condition']).size().unstack(fill_value=0)
print(result)

# To get the total counts for each med_condition across all groups
print(result.sum())

# sys.exit()

######### filter for med_off condition ################
# take everything from med_off condition except for 'on'
current_df = current_df[current_df['med_condition'] != 'on']
print(current_df['med_condition'].value_counts())


result = current_df[current_df['med_condition'].notna()].groupby(['group', 'med_condition']).size().unstack(fill_value=0)
print(result)


# sys.exit()





######### HERE WHICH DATASET IS USED FOR TRAINING AND TESTING ####################




'''
removing features from the random forest original csv file which is loaded at the beginning
lose some features:

'''
features_to_top = [
    'ttip_x_increasing', 'ttip_y_increasing', 
    'ttip_y_decreasing', 'ttip_x_decreasing',
    'tbo_x_increasing', 'tbo_y_increasing',
    'tbo_x_decreasing', 'tbo_y_decreasing',
    'ulip_x_increasing', 'ulip_y_increasing', 
    'ulip_x_decreasing', 'ulip_y_decreasing',
    'llip_x_increasing', 'llip_y_increasing',
    'llip_x_decreasing', 'llip_y_decreasing'
]



current_df = current_df.drop(columns=features_to_top, axis=1)


train_percentage = 0.8
# random_seed = random.randint(0, 1000)
random_seed = 1988
# df_train, df_test = hm.patient_out_names_dataset(current_df, train_percentage, random_seed)
# df_train, df_test = hm.patient_out_names_dataset_no_postop(current_df, train_percentage, random_seed) ### USE FOR PAPER REPLICATION DATASET

all_patients = current_df['patient'].unique()
print(all_patients)
# sys.exit(9)
train_patients = ['ARGE50_2_ses' 'BAKI67_2_ses' 'BRTH56_2_ses' 'HEGU51_2_ses'
 'HOOL63_2_ses' 'LEOL66_2_ses' 'LOEAN52_2_ses' 'SCHO77_2_ses'
 'STVO63_2_ses' 'WEMA63_2_ses' 'ARGE50_ses' 'BAKI67_ses' 'BRTH56_ses'
 'HEGU51_ses' 'HOOL63_ses' 'LEOL66_ses' 'LOEAN52_ses' 'SCHO77_ses'
 'STVO63_ses' 'WEMA63_ses' 'DEMA56_ses' 'DESU63_ses' 'KAFR57_ses'
 'KRTH61_ses' 'MOEFR40_ses' 'MUEBJ62_ses' 'PIWI53_ses' 'WITH53_ses'
 'BAEMI56_ses' 'BAER69_ses' 'BOEGA54_ses' 'DEKA63_ses' 'GIMI73_ses'
 'KEPA65_ses' 'KESV68_ses' 'KOEHU54_ses' 'LAAN64_ses' 'MOEAR53_ses'
 'NOBE63_ses' 'OKFE70_ses' 'PEST41_ses' 'POWO59_ses' 'SCJUE59_ses'
 'SWGI50_ses' 'TEHE48_ses' 'VIKE62_ses' 'WEIN51_ses' 'ZAHE51_ses'
 'BUGA65_ses' 'DUER50_ses' 'GEHE55_ses' 'GRMA61_ses' 'HIHU64_ses'
 'IBTH59_ses' 'LUNO52_ses' 'MIPA47_ses' 'MIPE57_ses' 'PEUL63_ses'
 'ROEFR54_ses' 'SARE54_ses' 'SCFR54_ses' 'SWHE53_ses' 'WEBE48_ses'
 'WODE57_ses' 'ZIBE59_ses']
test_patients = ['CRJO65_2_ses' 'JUEDO49_2_ses' 'SCJUE60_2_ses' 'CRJO65_ses' 'JUEDO49_ses'
 'SCJUE60_ses' 'QUTH59_ses' 'WOKL65_ses' 'ANHE49_ses' 'BAKA70_ses'
 'HABE57_ses' 'KATH76_ses' 'TEHA63_ses' 'BERA61_ses' 'GUOL65_ses'
 'KOED57_ses' 'REFR45_ses' 'WALU54_ses']


df_train = current_df[current_df['patient'].isin(all_patients)]
df_test = current_df[current_df['patient'].isin(all_patients)]

# print(df_train)
# print(df_test)
# sys.exit()

print(df_train['group'].value_counts())
print(df_test['group'].value_counts())
# sys.exit()

##########################################################################################################################
##########################################################################################################################
# uncomment later #
# controls_focus_emuDB patients
# print(df_train[df_train['group'] == 'controls_focus_emuDB']['patient'].unique())
# print(df_test[df_test['group'] == 'controls_focus_emuDB']['patient'].unique())
# # RBD_focus_emuDB patients
# print(df_train[df_train['group'] == 'RBD_focus_emuDB']['patient'].unique())
# print(df_test[df_test['group'] == 'RBD_focus_emuDB']['patient'].unique())
# # preoperative_focus_emuDB
# print(df_train[df_train['group'] == 'preoperative_focus_emuDB']['patient'].unique())
# print(df_test[df_test['group'] == 'preoperative_focus_emuDB']['patient'].unique())
# # postoperative_focus_emuDB
# print(df_train[df_train['group'] == 'postoperative_focus_emuDB']['patient'].unique())
# print(df_test[df_test['group'] == 'postoperative_focus_emuDB']['patient'].unique())



# # list all the patients in the dataset for each group
# print(current_df[current_df['group'] == 'controls_focus_emuDB']['patient'].unique())
# print(current_df[current_df['group'] == 'RBD_focus_emuDB']['patient'].unique())
# print(current_df[current_df['group'] == 'preoperative_focus_emuDB']['patient'].unique())
# print(current_df[current_df['group'] == 'postoperative_focus_emuDB']['patient'].unique())



# #only unique patients in preoperative and postoperative
# print('preop patients train')
# print(df_train[df_train['group'] == 'preoperative_focus_emuDB']['patient'].unique())
# print('postop patients train')
# print(df_train[df_train['group'] == 'postoperative_focus_emuDB']['patient'].unique())
# # only unique patients in preoperative and postoperative test
# print('preop patients test')
# print(df_test[df_test['group'] == 'preoperative_focus_emuDB']['patient'].unique())
# print('postop patients test')
# print(df_test[df_test['group'] == 'postoperative_focus_emuDB']['patient'].unique())

print('train')
print(df_train['group'].value_counts())
print('test')
print(df_test['group'].value_counts())

sys.exit()
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

# sys.exit()




# #### choosing 80% of the patients from Severe PD group in group3
# current_df_severe_df = current_df[current_df['group3'] == 'Severe PD']
# current_df_con_df = current_df[current_df['group3'] == 'CON']

# severe_pd_patiens = list(current_df_severe_df['patient'].unique())
# con_patients = list(paper_rep_con_df['patient'].unique())


# # Shuffle the patient lists
# np.random.seed(35)  # use 1988 and 35 for reproducibility
# np.random.shuffle(severe_pd_patiens)
# np.random.shuffle(con_patients)

# print(severe_pd_patiens)
# print(con_patients)
# # sys.exit()


# train_percentage = 0.8
# severe_pd_train = int(len(severe_pd_patiens) * train_percentage)
# con_train = int(len(con_patients) * train_percentage)
# severe_pd_test = int(len(severe_pd_patiens) - severe_pd_train)
# con_test = int(len(con_patients) - con_train)


# severe_train_df = current_df[current_df['patient'].isin(severe_pd_patiens[:severe_pd_train])]
# con_train_df = current_df[current_df['patient'].isin(con_patients[:con_train])]

# severe_test_df = current_df[current_df['patient'].isin(severe_pd_patiens[severe_pd_train:])]
# con_test_df = current_df[current_df['patient'].isin(con_patients[con_train:])]

# df_train = pd.concat([severe_train_df, con_train_df])
# df_test = pd.concat([severe_test_df, con_test_df])


# print(df_train['group3'].value_counts())
# print(df_test['group3'].value_counts())

# sys.exit()

########################### changing labels of group3 ##########################

if 'postoperative_focus_emuDB' not in df_train['group']:
    print('postoperative_focus_emuDB is not in the dataset')

    label_map = {
    'controls_focus_emuDB': 'Control',
    'RBD_focus_emuDB': 'RBD',
    'preoperative_focus_emuDB': 'Preoperative',
}

else:
    label_map = {
        'controls_focus_emuDB': 'Control',
        'RBD_focus_emuDB': 'RBD',
        'preoperative_focus_emuDB': 'Preoperative',
        'postoperative_focus_emuDB': 'Postoperative'
    }


df_train['group'] = df_train['group'].map(label_map)
df_test['group'] = df_test['group'].map(label_map)

print(df_train['group'].value_counts())

print('df train con patients:', df_train[df_train['group'] == 'Control']['patient'].unique())
print('df train rbd patients:', df_train[df_train['group'] == 'RBD']['patient'].unique())
print('df train preop patients:', df_train[df_train['group'] == 'Preoperative']['patient'].unique())
print('df train postop patients:', df_train[df_train['group'] == 'Postoperative']['patient'].unique())
sys.exit()


# print('df train con patients:', df_train[df_train['group3'] == 'Control']['patient'].unique())
# print('df train severe pd patients:', df_train[df_train['group3'] == 'Severe PD']['patient'].unique())
# print('df test con patients:', df_test[df_test['group3'] == 'Control']['patient'].unique())
# print('df test severe pd patients:', df_test[df_test['group3'] == 'Severe PD']['patient'].unique())
# sys.exit()
########################### changing labels of group3 ##########################


if 'name' in df_train.columns:
    df_train = df_train.drop('name', axis=1)
    df_train = df_train.drop('med_condition', axis=1)
    df_train = df_train.drop('condition', axis=1)
    
    df_test = df_test.drop('name', axis=1)
    df_test = df_test.drop('med_condition', axis=1)
    df_test = df_test.drop('condition', axis=1)


if 'sentence' in df_train.columns:
    df_train = df_train.drop('sentence', axis=1)
    df_test = df_test.drop('sentence', axis=1)

if 'group2' in df_train.columns:
    df_train = df_train.drop('group2', axis=1)
    df_test = df_test.drop('group2', axis=1)

if 'group3' in df_train.columns:
    df_train = df_train.drop('group3', axis=1)
    df_test = df_test.drop('group3', axis=1)

if 'med_condition' in df_train.columns:
    df_train = df_train.drop('med_condition', axis=1)
    df_test = df_test.drop('med_condition', axis=1)

### predicting group
X_train = df_train.drop(['group', 'patient', 'bundle'], axis = 1)
y_train = df_train['group']
X_test = df_test.drop(['group', 'patient', 'bundle'], axis = 1)
y_test = df_test['group']

print(df_train.columns)
# sys.exit()

#### stratify parameters ####
groups = df_train['patient'].values
cv = StratifiedGroupKFold(n_splits=5)

pipeline = Pipeline([
    ('smote', SMOTE()),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Define parameter grid for GridSearch
param_grid = {
    'rf__n_estimators': [0, 10, 50, 100, 200],
    'rf__max_depth': [None, 5, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4, 8]
}

# GridSearch with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=cv.split(X_train, y_train, groups), n_jobs=-1, verbose=2)

# Fit the model (this performs the cross-validation)
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_



# Perform cross-validation on the best model
cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv.split(X_train, y_train, groups))
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
print("Standard deviation of CV scores:", cv_scores.std())

# Now evaluate on the test set
y_pred = best_model.predict(X_test)
balanced_acc = balanced_accuracy_score(y_test, y_pred)




# Calculate and print metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_acc)
print("Classification Report:\n", classification_report(y_test, y_pred))



# Additional model information
best_model = grid_search.best_estimator_.named_steps['rf']
print("The total number of features used in the model is:", X_train.shape[1])
average_tree_depth = sum([estimator.tree_.max_depth for estimator in best_model.estimators_]) / len(best_model.estimators_)
print("Average Tree Depth:", average_tree_depth)
# print("Best Parameters:", grid_search.best_params_)

best_params = grid_search.best_params_



def save_output_to_file(folder, filename, content, df_train, df_test, train_patients, test_patients):
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)

    # Fetch unique groups and medical conditions from both datasets
    unique_groups_train = df_train['group'].unique()
    unique_groups_test = df_test['group'].unique()
    # unique_med_train = df_train['med_condition'].unique()
    # unique_med_test = df_test['med_condition'].unique()

    # Prepare additional content
    additional_content = "\nFile Used:\n" + file_and_path + "\n"
    additional_content += f"\nUnique Groups in Training Set: {', '.join(unique_groups_train)}\n"
    additional_content += f"Unique Groups in Testing Set: {', '.join(unique_groups_test)}\n"
    # additional_content += f"Unique Medical Conditions in Training Set: {', '.join(unique_med_train)}\n"
    # additional_content += f"Unique Medical Conditions in Testing Set: {', '.join(unique_med_test)}\n"

    additional_content += "\nPatient Names in Training Set:\n" + ", ".join(train_patients) + "\n"
    additional_content += "Patient Names in Testing Set:\n" + ", ".join(test_patients) + "\n"

    additional_content += "\nTrain Set Group Counts:\n" + str(df_train['group'].value_counts()) + "\n"
    additional_content += "Test Set Group Counts:\n" + str(df_test['group'].value_counts()) + "\n"
    
    additional_content += "\nTrain Percentage:\n" + str(train_percentage) + "\n"
    

    additional_content += "\nCross-validation scores:\n"
    additional_content += f"Scores: {cv_scores}\n"
    additional_content += f"Mean CV score: {cv_scores.mean():.4f}\n"
    additional_content += f"Standard deviation of CV scores: {cv_scores.std():.4f}\n"



    # Write everything to file
    with open(file_path, 'w') as file:
        file.write(content)
        file.write(additional_content)



def save_feature_importance(folder, model, feature_names):
    importances = model.feature_importances_

    
    
    indices = np.argsort(importances)
    
    # Increase figure size and adjust for horizontal orientation
    plt.figure(figsize=(10, 15))
    
    plt.title("Feature Importance")
    plt.barh(range(len(importances)), importances[indices], color='skyblue')
    
    # Adjust y-axis labels
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices], fontsize=10)
    
    # Add padding to the left
    plt.gcf().subplots_adjust(left=0.3)
    
    # Invert y-axis to have most important features at the top
    # plt.gca().invert_yaxis()
    
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'feature_importance.png'))
    plt.savefig(os.path.join(folder, 'feature_importance.svg'))
    plt.close()


def save_confusion_matrix(folder, y_true, y_pred, labels):
    desired_order = ['Control', 'RBD', 'Preoperative', 'Postoperative']
    # Filter and sort labels based on the desired order
    ordered_labels = [label for label in desired_order if label in labels]
    # Create the confusion matrix with the ordered labels
    cm = confusion_matrix(y_true, y_pred, labels=ordered_labels, normalize='true')
    # cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues', xticklabels=ordered_labels, yticklabels=ordered_labels)
    # sns.heatmap(cm, annot=True, fmt='.1%', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'confusion_matrix.png'))
    plt.savefig(os.path.join(folder, 'confusion_matrix.svg'))
    plt.close()




def save_roc_curve(folder, fpr, tpr, roc_auc, labels=None):
    plt.figure(figsize=(8, 6))
    if labels is None:
        # Binary classification
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    else:
        # Multi-class classification
        for i in range(len(labels)):
            plt.plot(fpr[i], tpr[i], label=f"{labels[i]} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, 'roc_curve.png'))
    plt.savefig(os.path.join(folder, 'roc_curve.svg'))
    plt.close()




accuracy_for_filename = accuracy_score(y_test, y_pred)
accuracy_for_filename = str(accuracy_for_filename).replace(".", "_")

balanced_acc_for_filename = str(balanced_acc).replace(".", "_")

timestamp = datetime.now().strftime("%Y-%m-%d---%H-%M-%S")
output_folder = r"D:\00000_master_thesis_new\results\001 MT Results\Random Forest results\con vs rbd vs preop vs postop - med off\sentences data\results_" + timestamp + ' balanced_acc_' + balanced_acc_for_filename[0:4] + '_train_percentage_' + str(train_percentage)

output = f"Best Parameters: {best_params}\n"
output += f"Accuracy: {accuracy_score(y_test, y_pred)}\n"
output += f"Balanced Accuracy: {balanced_acc:.4f}\n"  
output += f"Classification Report:\n{classification_report(y_test, y_pred)}\n"
output += f"Total number of features used in the model: {X_train.shape[1]}\n"
output += f"Average Tree Depth: {average_tree_depth}\n"
output += f"Cross-validation scores: {cv_scores}\n"
output += f"Mean CV score: {cv_scores.mean():.4f}\n"
output += f"Standard deviation of CV scores: {cv_scores.std():.4f}\n"




train_patients = df_train['patient'].unique()
test_patients = df_test['patient'].unique()

output_filename = f"model_results_{timestamp}.txt"

save_output_to_file(output_folder, output_filename, output, df_train, df_test, train_patients, test_patients)

save_feature_importance(output_folder, best_model, X_train.columns)
save_confusion_matrix(output_folder, y_test, y_pred, labels=best_model.classes_)


