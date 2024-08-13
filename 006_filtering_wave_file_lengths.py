

############################################
# here, I will create another annotation file where only the files are included which are
# 1.1 - 3.6 sec long
#############################################


import pandas as pd
import matplotlib.pyplot as plt
import heapq
from scipy.stats import norm
import numpy as np
import math


###
# this file is adding a column named total length into the annotation and creating a new annotation file where
# only files which are 1.2 - 3.6 seconds long are taken 
#
# check which file is being used !!!
df = pd.read_excel(r'C:\Users\iaktug\Desktop\Master_thesis\0_sentence_annotation\03_only_good_updated_annotations.xlsx')

sr = 48000

# add a new column names total_length in seconds
df['total_length'] = (df['end_point'] / sr) - (df['start_point'] / sr)


# Filter for all values between 1.2 and 3.6 seconds
filtered_df = df[(df['total_length'] >= 1.2) & (df['total_length'] <= 3.6)]


# print(df.head())
# print(filtered_df.head())
print(df)
print(filtered_df)


### exporting files
# filtered_df.to_csv(r'C:\Users\iaktug\Desktop\Master_thesis\0_sentence_annotation\04_only_good_updated_annotations.csv', index=False)
# filtered_df.to_excel(r'C:\Users\iaktug\Desktop\Master_thesis\0_sentence_annotation\04_only_good_updated_annotations.xlsx', index=False)