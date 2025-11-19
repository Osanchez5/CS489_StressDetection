import torch
import torch.nn as nn

# Will be potentially utilized for comparison with the times net model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

    def forward(x):
        return x