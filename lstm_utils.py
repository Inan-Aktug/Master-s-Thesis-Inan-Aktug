# Import required libraries
import data_preprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
import matplotlib.pyplot as plt
# import seaborn as sns
import math
# from sklearn.model_selection import KFold

##################################################################################################
'''
creating Data class 
this class later takes inputs and is currently not taking our actualy values
It takes inputs from the DataLoader later
it would look something like this:
train_dataset = Data(X_train_tensor, train_labels_tensor, actual_len_train_tensor) 
test_dataset = Data(Y_test_tensor, test_labels_tensor, actual_len_test_tensor)

additional information:
Data & DataLoader
Data creates data for the Dataset object. It does not matter if it's train or test. 
Then the Dataset is passed to the DataLoader which handles batches and shuffling if shuffle = True
'''
class Data(Dataset):
    def __init__(self, X, y, actual_lengths):
        self.X = X
        self.y = y
        self.actual_lengths = actual_lengths
        self.len = self.X.shape[0]

    def __getitem__(self, index): 
        '''
        HERE IS GET THE Actual index and the items of that index so that it is connected, even when shuffeled 
       '''
        # print(f"Index: {index}, X shape: {self.X[index].shape}, Actual length: {self.actual_lengths[index]}")
        return self.X[index], self.actual_lengths[index], self.y[index]
        # Debug print statements
        # if index < 5:  # Print for the first 5 indices
        #     print(f"Index: {index}")
        #     print(f"Data shape: {self.X[index].shape}")
        #     print(f"Actual length: {self.actual_lengths[index]}")
        #     print(f"Label: {self.y[index]}")
        return self.X[index], self.actual_lengths[index], self.y[index]
        
    def __len__(self):
        return self.len

##################################################################################################





##################################################################################################
'''
Here, I create the actual LSTM Network where I combine all the values needed

- h0 = torch.zeroes or torch.rand ?
- c0 same as above


'''

class LSTM_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, max_sequence_length, dropout_rate=0.3):
        super(LSTM_Classifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, num_classes)
        self.max_sequence_length = max_sequence_length

    def forward(self, x, actual_lengths):
        h0 = torch.rand(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.rand(self.num_layers, x.size(0), self.hidden_size)

        packed_input = nn.utils.rnn.pack_padded_sequence(x, actual_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        batch_size = output.size(0)
        rows = torch.arange(batch_size)
        cols = (actual_lengths - 1).clamp(min=0, max=output.size(1)-1)
        relevant_output = output[rows, cols]

        # Apply dropout before the linear layer
        relevant_output = self.dropout(relevant_output)

        # Pass through the linear layer
        out = self.linear(relevant_output)
        return out

# class LSTM_Classifier(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes, num_layers, max_sequence_length):
#         super(LSTM_Classifier, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_size, num_classes)
#         self.max_sequence_length = max_sequence_length


#         # # batch norm per layer
#         # self.bn = nn.BatchNorm1d(hidden_size)

#         # self.fc = nn.Linear(hidden_size, num_classes)


#     def forward(self, x, actual_lengths):
#         # print('Input x shape:', x.shape)

#         h0 = torch.rand(self.num_layers, x.size(0), self.hidden_size) 
#         c0 = torch.rand(self.num_layers, x.size(0), self.hidden_size)


#         '''
#         missing in the forward is the use of the linear layer!!! -- fixed
#         '''
#         # print(f"Input x shape: {x.shape}")
#         # print(f"Actual lengths before clamp: {actual_lengths}")
#         # actual_lengths = torch.clamp(actual_lengths, max=x.size(1))
#         # print(f"Actual lengths after clamp: {actual_lengths}")
#         # print("Actual lengths:", actual_lengths[:5])  # Print the lengths of the first few sequences in the batch
#         packed_input = nn.utils.rnn.pack_padded_sequence(x, actual_lengths, batch_first=True, enforce_sorted=False)


#         # print(x) 
#         # print(actual_lengths) # is a tensor which has as many values as the batch size 
#         # sys.exit()
#         packed_output, (hidden, cell) = self.lstm(packed_input)
#         # print(packed_output)
#         # print(hidden.shape)
#         # sys.exit()
#         output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
#         # print("Unpacked LSTM output shape:", output.shape)

#         # print('output shape here')
#         # print("Output shape:", output.shape)
#         # # print("Index tensor shape:", idx.shape)

#         ##############
#         batch_size = output.size(0)
#         # print(batch_size)
#         rows = torch.arange(batch_size)
#         # print('rows')
#         # print(rows)

#         # cols = actual_lengths - 1
#         cols = (actual_lengths - 1).clamp(min = 0, max=output.size(1)-1)

#         relevant_output = output[rows,cols]
#         # Pass through the linear layer
#         out = self.linear(relevant_output)



#         return out

'''
recheck the code above,
make sure everything is correctly setup
'''


##################################################################################################