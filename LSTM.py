import torch
import torch.nn as nn

# Will be potentially utilized for comparison with the timesnet model
# Not the main focus of the project so the implementation will be simple
# Might be moved into model.py
class LSTM(nn.Module):
    def __init__(self, inputSize=3, hiddenSize=1, numLayers=1, outputDim=1):
        super(LSTM, self).__init__()

        # Set values for nn.LSTM and nn.Linear
        self.hS = hiddenSize
        self.nL = numLayers
        self.lstm = nn.LSTM(inputSize, hiddenSize, numLayers)
        self.linear = nn.Linear(hiddenSize, outputDim)

    def forward(self, x):
        # Initial implementation, https://www.geeksforgeeks.org/deep-learning/long-short-term-memory-networks-using-pytorch/
        
        h0 = torch.zeros(self.nL, x.size(0), self.hS)
        c0 = torch.zeros(self.nL, x.size(0), self.hS)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return x