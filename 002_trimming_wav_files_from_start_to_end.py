import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import soundfile as sf
import shutil
import sys


# Directory path
# dir_path = r'C:\Users\iaktug\Desktop\Master_thesis\001_good_data\controls_focus_emuDB\BAKA70_ses'
# dir_path = r'C:\Users\iaktug\Desktop\Master_thesis\all_data'

dir_path = r'D:\00000_master_thesis_new\all_data_trimmed'
# dir_path = r'C:\Users\iaktug\Desktop\Master_thesis\001_good_data\controls_focus_emuDB' # 21
# dir_path = r'C:\Users\iaktug\Desktop\Master_thesis\001_good_data\postoperative_focus_emuDB' # 1
# dir_path = r'C:\Users\iaktug\Desktop\Master_thesis\001_good_data\preoperative_focus_emuDB' # 12
# dir_path = r'C:\Users\iaktug\Desktop\Master_thesis\001_good_data\RBD_focus_emuDB' #11




df = pd.read_excel(r'D:\00000_master_thesis_new\0_sentence_annotation\0_only_good_updated_annotations-3.xlsx')

df['bundle'] = df['bundle'].astype(str)
# print(df)
# sys.exit()




# def write_trimmed_audio(file_path, start_point, end_point, y):
#     trimmed_audio = y[start_point:end_point]
#     sf.write(file_path, trimmed_audio, 48000)





# Iterate over each file in the directory
for root, dirs, files in os.walk(dir_path):
    for file in files:
        if file.endswith('.wav') and not file.startswith('trimmed'): ### not trimmed
            file_path = os.path.join(root, file)

            # Load the audio file
            y, sr = librosa.load(file_path, sr=48000)

            # Apply smoothing
            window_size = 480
            window = np.ones(int(window_size))/float(window_size)
            y_smooth = np.convolve(y, window, 'same')



            # Calculate the differential of the smoothed audio signal
            differential = np.diff(y)
            differential_smooth = np.diff(y_smooth)
            

            
            # Find the start point of the audio signal of interest
            n_std_dev=2.5
            small_value=0.00001 ##original value 0.00001
            quiet_length=1200 ## origninal value 2400 -> 1200 seem to work very well

            threshold = np.mean(np.abs(differential_smooth)) + n_std_dev * np.std(differential_smooth)
            index_array = np.where(np.abs(differential_smooth) > threshold)[0]




            # If index_array is empty, set start_point to 0, else set it to the first element of index_array.
            # If threshold is not triggered
            if index_array.size == 0:
                start_point = 0
            else:
                start_point = index_array[0]

            # Initialize quiet_counter to 0
            quiet_counter = 0

            # While start_point is greater than 0 and quiet_counter is less than quiet_length
            while start_point > 0 and quiet_counter < quiet_length:

                # Check the absolute value of the element at index start_point in differential_smooth
                differential_at_start_point = np.abs(differential_smooth[start_point])

                # If the absolute value is greater than small_value, reset quiet_counter to 0
                if differential_at_start_point > small_value:
                    quiet_counter = 0

                # Else increment quiet_counter
                else:
                    quiet_counter += 1

                # Decrement start_point by 1 for the next iteration
                start_point -= 1




            # Calculate the number of samples that make up 0.1 seconds
            segment_size = int(0.1 * sr)

            # Calculate the hop size for 50% overlap
            hop_size = segment_size // 2  

            # Calculate the RMS values with the new segment and hop sizes
            frames = librosa.util.frame(y_smooth, frame_length=segment_size, hop_length=hop_size)
            rms = np.sqrt(np.mean(frames**2, axis=0))





################

            # Reverse the differential_smooth array first
            # reversed_audio = np.flip(abs(differential_smooth)) ### maybe try with regular differential
            reversed_audio = np.flip(abs(y))

            # Number of samples per chunk and the sample rate.
            chunk_size = 4800 # try 2400, 1200
            sample_rate = 48000 

            # How many future chunks to compare with.
            num_future_chunks = 3 # try 4, 5

            # Threshold for difference in mean value to consider as sound.
            diff_threshold = 0.2 # 
            
            # Initialize end_point to the start of the reversed audio (which is the end of the original signal)
            end_point = len(reversed_audio)
            # print('reversed_audio:' , reversed_audio)
            # print('end point:' , end_point)
            # Calculate the number of chunks in the array.
            num_chunks = len(reversed_audio) // chunk_size
            
            # for making a list of all mean values
            mean_of_chunks = []
                
            
            for i in range(num_chunks):
                # print(num_chunks)
                chunk_start = i * chunk_size
                chunk_end = chunk_start + chunk_size


                current_mean = np.mean(reversed_audio[chunk_start:chunk_end])
                # print(current_mean)
                #### make a list of all mean values
                mean_of_chunks.append(current_mean)


            threshold
            chunk_counter = 0
            chunk_counter_abs = 0
            for i in range(len(mean_of_chunks)-num_future_chunks):
                current_chunk = mean_of_chunks[i]
                future_chunk1 = mean_of_chunks[i+1]
                future_chunk2 = mean_of_chunks[i+2]
                future_chunk3 = mean_of_chunks[i+3]

                #difference in %
                diff1 = current_chunk / future_chunk1
                diff2 = current_chunk / future_chunk2
                diff3 = current_chunk / future_chunk3
                
                
                
                print('current chunk:', current_chunk)
                print('future chunk1:', future_chunk1)
                print('future chunk2:', future_chunk2)
                print('future chunk3:', future_chunk3)
                print('diff1:', diff1)
                print('diff2:', diff2)
                print('diff3:', diff3)
                print('-----------------')
                
                # Check if all differences are below the threshold
                if diff1 < diff_threshold and diff2 < diff_threshold or diff1 < diff_threshold and diff3 < diff_threshold or diff2 < diff_threshold and diff3 < diff_threshold :
                    # If they are, break the loop
                    break

                chunk_counter += 1

                ################# recheck this part

                # # Correctly calculate percentage differences
                # diff1 = abs(current_chunk - future_chunk1) / current_chunk
                # diff2 = abs(current_chunk - future_chunk2) / current_chunk
                # diff3 = abs(current_chunk - future_chunk3) / current_chunk

                # # Correct condition to find when two out of three differences exceed the threshold
                # if (diff1 >= diff_threshold and diff2 >= diff_threshold) or (diff1 >= diff_threshold and diff3 >= diff_threshold) or (diff2 >= diff_threshold and diff3 >= diff_threshold):
                #     # If the condition is met, break the loop
                #     break

                # chunk_counter_abs += 1
                
                # print('--------- abs ---------')
                # print('chunk counter:', chunk_counter)
                # print('diff1_abs:', diff1)
                # print('diff2_abs:', diff2)
                # print('diff3_abs:', diff3)
                # print('--------- abs end---------')
                ################# recheck this part end


            # Compute the end point in the original signal
            end_point = len(differential_smooth) - chunk_counter * chunk_size

            # If the end point is before the start point, skip this file
            if end_point < start_point:
                print(f"Skipping {file_path} because end point is before start point.")
                continue



            # print('start:',start_point)
            # print('end:', end_point)


            # Create an x-axis for the RMS plot in seconds
            rms_time = np.linspace(0, len(rms)*0.05, len(rms))




            # Extract group name, patient name, and bndl name
            parts = root.split(os.sep)
            group_name = parts[-3]  # Adjust index as needed based on your directory structure
            patient_name = parts[-2]
            bndl_name = parts[-1]  
            bndl_number = file.replace('.wav', '')
            # print(bndl_number)






            ### calling the trimming function
            # After start and end point is calculated

            # trimmed_file_path = os.path.join(root, f"trimmed_{file}")
            # write_trimmed_audio(trimmed_file_path, start_point, end_point, y) ### calling the trim function with its parameters
            







            # print(bndl_name)
            # print(bndl_number)
            # print(patient_name)
            

            

            matching_rows = df[(df['group'] == group_name) & (df['patient'] == patient_name) & (df['bundle'] == bndl_name)]
            # print(f"Matching rows for group={group_name}, patient={patient_name}, bundle={bndl_name}:\n", matching_rows)
            # print(matching_rows)
            if matching_rows.empty:
                print(f"No matching rows found for group={group_name}, patient={patient_name}, bundle={bndl_name}. Skipping...")
                continue

            row_index = matching_rows.index[0]
            print(f"Index: {row_index}")


            start_point = int(start_point)
            end_point = int(end_point)



            # # Add start_point and end_point to this row
            df.at[row_index, 'start_point'] = start_point
            df.at[row_index, 'end_point'] = end_point

            # print(df)
print(df)
df['start_point'] = df['start_point'].astype('Int64')
df['end_point'] = df['end_point'].astype('Int64')
print(df)

