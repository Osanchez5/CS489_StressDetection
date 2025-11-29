import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from typing import Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        labels = labels.long()
        # print(labels)

        x_mark_enc = torch.ones(inputs.shape[0], inputs.shape[1]).to(device)

        optimizer.zero_grad()
        outputs = model(inputs, x_mark_enc)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        _,preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_train_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy




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
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.long()

            x_mark_enc = torch.ones(inputs.shape[0], inputs.shape[1]).to(device)
            outputs = model(inputs, x_mark_enc)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _,preds = torch.max(outputs,1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy



def test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[float, float, float, float, float, float]:
    

    loss, acc = validate(model, dataloader, criterion, device)

    return loss, acc