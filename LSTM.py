import torch
import torch.nn as nn

# Will be potentially utilized for comparison with the timesnet model
# Might be moved into model.py
class LSTM(nn.Module):
    def __init__(self, inputDim=3, hiddenDim=1, layerDim=1, outputDim=1):
        super(LSTM, self).__init__()

        self.hD = hiddenDim
        self.lD = layerDim
        self.lstm = nn.LSTM(inputDim, hiddenDim, layerDim)
        self.linear = nn.Linear(hiddenDim, outputDim)

    def forward(x):

        return x