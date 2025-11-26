import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import datetime

class StressDetectionDataset(Dataset):
    # Just used Heart Rate for now, and uses a sequence length of 64
    def __init__(self, path, ts_length = 64):
        self.samples = []
        self.labels = []
        self.class_map = {'STRESS': 0, 'AEROBIC': 1, 'ANAEROBIC': 2}

        for label_name, label_idx in self.class_map.items():
            # Each label has it's own directory in Wearable_Dataset
            class_dir = os.path.join(path, label_name)

            # Each subject has it's own directory inside Wearable_Dataset/LABEL (STRESS, AEROBIC, ANAEROBIC)
            for subject_name in os.listdir(class_dir):
                subject_dir = os.path.join(class_dir, subject_name)
                hr_file = os.path.join(subject_dir, 'HR.csv')

                # Have to skip first two rows, data starts at row 3
                df = pd.read_csv(hr_file, header=None, skiprows=2)
                hr_values = df[0].values.astype(float)

            total_sequences = len(hr_values) // ts_length

            # Seperate each samples by the sequence length
            for i in range(total_sequences):
                start = i * ts_length
                end = start + ts_length
                sequence = hr_values[start:end]

                self.samples.append(sequence)
                self.labels.append(label_idx)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self.samples[index]
        y = self.samples[index]

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        x_tensor = x_tensor.unsqueeze(-1) # unsqueeze so we have the B,T,N shape

        return x_tensor, y_tensor