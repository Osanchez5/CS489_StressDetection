import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import datetime

class StressDetectionDataset(Dataset):
    # Just used Heart Rate for now, and uses a sequence length of 64
    def __init__(self, path, ts_length = 64, step = 8):
        self.samples = []
        self.labels = []
        self.timestamps = []

        base_dir = os.path.dirname(path.rstrip('/'))
        stress_levels_path_v1 = os.path.join(base_dir, "Stress_Level_v1.csv")
        stress_levels_path_v2 = os.path.join(base_dir, "Stress_Level_v2.csv")

        self.stress_scores = {}

        def load_scores(file_path):
            if os.path.exists(file_path):
                print(f"Loading scores from: {file_path}")
                df = pd.read_csv(file_path).set_index('Unnamed: 0')
                scores_dict = df.apply(lambda row: row.to_dict(), axis=1).to_dict()
                return scores_dict
            else:
                print(f"Warning: Score file not found: {file_path}")
            return {}

        self.stress_scores.update(load_scores(stress_levels_path_v1))
        self.stress_scores.update(load_scores(stress_levels_path_v2))
        print(f"Loaded scores for {len(self.stress_scores)} subjects.")

        label_name = 'STRESS'
        class_dir = os.path.join(path, label_name)
        
        for subject_name in os.listdir(class_dir):
            subject_id = subject_name.upper()[:3]
            
            if subject_id in self.stress_scores:
                subject_dir = os.path.join(class_dir, subject_name)
                hr_file = os.path.join(subject_dir, 'HR.csv')

                meta_df = pd.read_csv(hr_file, header=None, nrows=2)
                start_val = meta_df.iloc[0,0]

                try:
                    start_timestamp = float(start_val)
                except ValueError:
                    start_timestamp = pd.to_datetime(start_val).timestamp()

                frequency = float(meta_df.iloc[1,0])

                df_hr = pd.read_csv(hr_file, header=None, skiprows=2)
                hr_values = df_hr[0].values.astype(float)

                num_samples = len(hr_values)
                if num_samples < ts_length:
                    continue

                subject_scores = list(self.stress_scores[subject_id].values())
                segments_per_task = num_samples // len(subject_scores)
                # print(f" [Processing] {subject_id}: {len(subject_scores)} tasks, {segments_per_task} steps/task.")

                for task_idx, score in enumerate(subject_scores):
                    task_start = task_idx * segments_per_task
                    task_end = (task_idx + 1) * segments_per_task

                    for start in range(task_start, task_end - ts_length + 1, ts_length):
                        end = start + ts_length
                        sequence = hr_values[start:end]

                        seq_start_time = start_timestamp + (start / frequency)
                        dates = pd.date_range(
                            start=pd.to_datetime(seq_start_time, unit='s'),
                            periods=ts_length,
                            freq=f'{1000/frequency}ms'
                        )

                        time_features = np.stack([
                            dates.month,
                            dates.day,
                            dates.weekday,
                            dates.hour
                        ], axis=1)

                        self.samples.append(sequence)
                        self.labels.append(score)
                        self.timestamps.append(time_features)

        print(f"Total forcasting segments loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self.samples[index]
        y = self.labels[index]
        t = self.timestamps[index]
        
        mean = np.mean(x)
        std = np.std(x)

        if std == 0:
            std = 1.0
        
        x = (x - mean) / std

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor([y], dtype=torch.float32)
        t_tensor = torch.tensor(t, dtype=torch.float32)

        return x_tensor, y_tensor, t_tensor