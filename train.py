import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from losses import MAE_loss

def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_train_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels, x_mark_enc in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        x_mark_enc = x_mark_enc.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, x_mark_enc)
        outputs = outputs.mean(dim=1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_train_loss / len(dataloader)
    epoch_mae = MAE_loss()(torch.tensor(all_preds), torch.tensor(all_labels)).item()

    return avg_loss, epoch_mae

def validate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
) -> Tuple[float, float, float, float, float, float, float]:
    
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, x_mark_enc in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            x_mark_enc = x_mark_enc.to(device)

            outputs = model(inputs, x_mark_enc)
            outputs = outputs.mean(dim=1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        epoch_mae = MAE_loss()(torch.tensor(all_preds), torch.tensor(all_labels)).item()

    return avg_loss, epoch_mae


def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float, float, float, float]:
    
    criterion = MAE_loss()
    return validate (model, dataloader, criterion, device)