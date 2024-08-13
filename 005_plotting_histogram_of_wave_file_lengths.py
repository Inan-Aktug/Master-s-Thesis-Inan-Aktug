import pandas as pd
import matplotlib.pyplot as plt
import heapq
from scipy.stats import norm
import numpy as np
import math



df = pd.read_excel(r'C:\Users\iaktug\Desktop\Master_thesis\0_sentence_annotation\03_only_good_updated_annotations.xlsx')


# print(df.columns)
all_start = []
all_end = []
length_in_sec = []
for i, row in df.iterrows():
    start = round(row['start_point'] / 48000, 2)
    end = round(row['end_point'] / 48000, 2)
    all_start.append(start)
    all_end.append(end)
    current_length = round(end - start, 2)
    length_in_sec.append(current_length)



ten_smallest = heapq.nsmallest(10, length_in_sec)
ten_largest = heapq.nlargest(10, length_in_sec)

print(ten_smallest)
print(ten_largest)


# counter = 0
# nan_counter = 0

length_in_sec_no_nan = []

for i in length_in_sec:
    
    if not math.isnan(i):
        length_in_sec_no_nan.append(i)
        
        # print(f'index is NaN', {counter})
        # print(f"{i} is NaN")
        # # nan_counter += 1
    # counter += 1    


print(len(length_in_sec))
print(len(length_in_sec_no_nan))
# print(nan_counter)
# print(length_in_sec[3483])




#################################
# plotting histogram 
plt.hist(length_in_sec_no_nan, bins=100, edgecolor = 'black')
plt.title('Histogram of wav file length distribution')
plt.xlabel('Time [s]')
plt.ylabel('Frequency')
plt.show()
##############





