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
                eda_file = os.path.join(subject_dir, 'EDA.csv')
                temp_file = os.path.join(subject_dir, 'TEMP.csv')

                def load_signal(filepath, col_name):
                    try:
                        meta = pd.read_csv(filepath, header=None, nrows=2)
                        start_time = float(meta.iloc[0,0]) if isinstance(meta.iloc[0, 0], (int, float)) else pd.to_datetime(meta.iloc[0, 0]).timestamp()
                        freq = float(meta.iloc[1,0])

                        data = pd.read_csv(filepath, header=None, skiprows=2)
                        values = data[0].values.astype(float)

                        time_index = pd.to_datetime(start_time, unit='s') + pd.to_timedelta(np.arange(len(values)) / freq, unit='s')
                        series = pd.Series(values, index=time_index, name=col_name)
                        resampled = series.resample('1s').mean()
                        return resampled
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
                        return None

                hr_series = load_signal(hr_file, 'HR')
                eda_series = load_signal(eda_file, 'EDA')
                temp_series = load_signal(temp_file, 'TEMP')

                if hr_series is None or eda_series is None or temp_series is None:
                    continue

                df_merged = pd.concat([hr_series, eda_series, temp_series], axis=1, join='inner')
                
                if len(df_merged) < ts_length:
                    continue

                merged_values = df_merged.values
                timestamps = df_merged.index

                subject_scores = list(self.stress_scores[subject_id].values())
                segments_per_task = len(merged_values) // len(subject_scores)

                for task_idx, score in enumerate(subject_scores):
                    task_start = task_idx * segments_per_task
                    task_end = (task_idx + 1) * segments_per_task

                    for start in range(task_start, task_end - ts_length + 1, step):
                        end = start + ts_length
                        sequence = merged_values[start:end]

                        seq_start_time = timestamps[start]
                        dates = pd.date_range(
                            start=seq_start_time,
                            periods=ts_length,
                            freq='1s'
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

        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)

        std[std==0] = 1.0
        
        x = (x - mean) / std

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor([y], dtype=torch.float32)
        t_tensor = torch.tensor(t, dtype=torch.float32)

        return x_tensor, y_tensor, t_tensor