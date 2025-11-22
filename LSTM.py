import torch
import torch.nn as nn

# Will be potentially utilized for comparison with the timesnet model
# Might be moved into model.py
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

    def forward(x):
        return x