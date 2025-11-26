import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import datatime

# From Wearable_Dataset.ipynb file to create a vector from the dataframe
def create_df_array(dataframe):
    matrix_df = dataframe.values
    matrix = np.array(matrix_df)
    array_df = matrix.flatten()
    return array_df

# From Wearable_Dataset.ipynb file to convert UTC arrays to array in seconds relative to 0 (record beginning)
def time_abs(UTC_array):
    new_array = []
    for utc in UTC_array:
        time=(datetime.datetime.strptime(utc,'%Y-%m-%d %H:%M:%S')-datetime.datetime.strptime(UTC_array[0], '%Y-%m-%d %H:%M:%S')).total_seconds()
        new_array.append(int(time))
    return new_array

# From Wearable_Dataset.ipynb file to get all features (signals) from directory and place them in dictionaries
def read_signals(main_folder):
    signal_dict = {}
    time_dict = {}
    fs_dict = {}

    subfolders = next(os.walk(main_folder))[1]
    utc_start_dict = {}

    for folder_name in subfolders
        folder_path = os.path.join(main_folder, folder_name)
        files = os.listdir(folder_path)
        
        signals = {}
        time_line = {}
        fs_signal = {}

        desired_files = {'HR.csv', 'tags.csv'}

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith('.csv') and file_name in desired_files:
                if file_name == 'tags.csv':
                    try:
                        df = pd.read_csv(file_path, header=None)
                        tags_vector = create_df_array(df)
                        tags_UTC_vector = np.insert(tags_vector, 0, utc_start_dict[folder_name])
                        signal_array = time_abs(tags_UTC_vector)
                    except pd.errors.EmptyDataError:
                        signal_array=[]
                
                else:
                    df = pd.read_csv(file_path)
                    fs = df.loc[0]
                    fs = int(fs[0])
                    df.drop([0], axis = 0, inplace=True)
                    signal_array = df.values
                    time_array = np.linspace(0, len(signal_array)/fs, len(signal_array))
                
                signal_name = file_name.split('.')[0]
                signals[signal_name] = signal_array
                time_line[signal_name] = time_array
                fs_signal[signal_name] = fs
        
        signal_dict[folder_name] = signals
        time_dict[folder_name] = time_line
        fs_dict[folder_name] = fs_signal

        return signal_dict, time_dict, fs_dict


class StressDetectionDataset(Dataset):
    def __init__(self, signal_data)