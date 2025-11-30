import torch
import torch.nn as nn


class MAE_loss():
    def __init__(self):
        super(MAE_loss, self).__init__()
    
    def forward(self, predicted, targets):
        # 1/N SUM(|targets - predicted|)
        abs_diff = torch.abs(targets - predicted)
        loss = torch.mean(abs_diff)
        return loss
    
class MSE_loss():
    def __init__(self):
        super(MSE_loss, self).__init__()
    
    def forward(self, predicted, targets):
        # 1/N SUM(targets - predicted)^2
        squared = (targets - predicted) ** 2
        loss = torch.mean(squared)
        return loss