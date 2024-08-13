import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
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


path = r'D:\00000_master_thesis_new\csv files\rf_features\names_dataset\csv'
file = r'all_groups_C1toV2_annotated_centered_min_max_per_patient_rf_features.csv'
file_and_path = os.path.join(path, file)



df = pd.read_csv(file_and_path)

###     information about the test / train set 
#       9/13 patients taken from postop in total (70%)
#       16/23 needed from preop in total (70%)
#       same 9 from the 16 needed + 7 random ones 
#       
###


df_train, df_test = hm.patient_out_names_dataset(df, 0.7)

print(df_train['group'].value_counts())
print(df_test['group'].value_counts())




# # Define class weights -> not working
# class_weights = {
#     'RBD_focus_emuDB': 1.0,
#     'controls_focus_emuDB': 1.0,
#     'postoperative_focus_emuDB': 1.0,
#     'preoperative_focus_emuDB': 0.0000000000001  # Lower weight for the overrepresented class
# }



### predicting group
X_train = df_train.drop(['group', 'patient', 'bundle', 'name', 'med_condition', 'condition'], axis = 1)
y_train = df_train['group']
X_test = df_test.drop(['group', 'patient', 'bundle', 'name', 'med_condition', 'condition'], axis = 1)
y_test = df_test['group']


### predicting name 
# X_train = df_train.drop(['group', 'patient', 'bundle', 'name', 'med_condition', 'condition'], axis = 1)
# y_train = df_train['name']
# X_test = df_test.drop(['group', 'patient', 'bundle', 'name', 'med_condition', 'condition'], axis = 1)
# y_test = df_test['name']



# Create a Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
# model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=5, random_state=42) # tried different values, no improvement of the accuracy


# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)







####
# # Extract unique class names from the training data
# unique_classes = y_train.unique()

# # Extract keys from class_weights dictionary
# class_weights_keys = list(class_weights.keys())

# # Check if all unique classes are in the class_weights dictionary
# missing_in_weights = set(unique_classes) - set(class_weights_keys)
# extra_in_weights = set(class_weights_keys) - set(unique_classes)

# if not missing_in_weights and not extra_in_weights:
#     print("All class names match between the data and class_weights dictionary.")
# else:
#     if missing_in_weights:
#         print("These classes are in the data but not in the class_weights dictionary:", missing_in_weights)
#     if extra_in_weights:
#         print("These classes are in the class_weights dictionary but not in the data:", extra_in_weights)

####




# Evaluate the model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Get feature importances
importances = model.feature_importances_

# Create a DataFrame to display the feature importances
feature_importances = pd.DataFrame({'Feature': X_test.columns, 'Importance': importances})

# Sort the DataFrame by importance in descending order
feature_importances = feature_importances.sort_values('Importance', ascending=False)

# # Print the feature importances
# print("Feature Importances:")
# print(feature_importances)


num_features = X_train.shape[1]
print(f"The total number of features used in the model is: {num_features}")


# Tree depth for Random Forest with max_depth = None (default)
tree_depths = [tree.get_depth() for tree in model.estimators_]
average_tree_depth = sum(tree_depths) / len(tree_depths)
print("Average Tree Depth:", average_tree_depth)




# Define the shortened class names
# shortened_class_names = ['RBD', 'Control', 'Post-op off', 'Post-op on', 'Pre-op']
# Get the unique class labels from the model
class_labels = model.classes_

# Create a dictionary to map the original class labels to their shortened versions
# label_mapping = {
#     'controls_focus_emuDB': 'Control',
#     'RBD_focus_emuDB': 'RBD',
    
#     'postoperative_off': 'Postoperative off',
#     'postoperative_on': 'Postoperative on',
#     'preoperative_focus_emuDB': 'Preoperative'
# }


### for regular random forest when predicting group
label_mapping = {
    'controls_focus_emuDB': 'Control',
    'RBD_focus_emuDB': 'RBD',
    'postoperative_focus_emuDB': 'Postoperative',
    'preoperative_focus_emuDB': 'Preoperative'
}


### for RF when predicting name number 
# label_mapping = {0: 'lani', 1: 'lena', 2: 'lina', 3: 'loni', 4: 'luna',
#                  5: 'mali', 6: 'mela', 7: 'mila', 8: 'moli', 9: 'mula'}



# Create the shortened class names list in the correct order
shortened_class_names = [label_mapping[label] for label in class_labels]



# Compute the confusion matrix and normalize it
matrix = confusion_matrix(y_test, y_pred)
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16, 7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size': 20},
            cmap=plt.cm.Greens, linewidths=0.2)

# Add labels to the plot
tick_marks = np.arange(len(shortened_class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, shortened_class_names, rotation=25)
plt.yticks(tick_marks2, shortened_class_names, rotation=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix for Random Forest Model')
# plt.show()


# Set the figure size and plot style
plt.figure(figsize=(12, 6))
plt.style.use('ggplot')

# Create a bar chart of feature importances
plt.bar(feature_importances['Feature'], feature_importances['Importance'])

# Set the chart title and labels
target_name = y_train.name  # Get the name of the target variable
plt.title(f'Feature Importances for Predicting {target_name}')
plt.xlabel('Features')
plt.ylabel('Importance')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=90)

# Adjust the plot margins for better readability
plt.subplots_adjust(top=0.95, bottom=0.5)

# Show the plot
plt.show()