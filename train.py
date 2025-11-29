import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Tuple

def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_train_loss = 0.0



    return




def validate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
) -> Tuple[float, float, float, float, float, float, float]:
    
    return



def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float, float, float, float]:
    
    return