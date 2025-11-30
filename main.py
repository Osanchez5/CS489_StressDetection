import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from losses import MSE_loss, MAE_loss
from dataset import StressDetectionDataset
from model import TimesNet
from train import train_one_epoch, validate, test
from config import config_args
from typing import Any
from LSTM import LSTM
from pathlib import Path

def run_training(args: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Your model is running on {DEVICE}...\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



    # Load dataset
    # Load in the subject info csv file? We'd only care about the info column
    file_path = "stressDetection/dataset/Wearable_Dataset"
    dataset = StressDetectionDataset(path=file_path, ts_length=args.seq_len)
    print(f"Total segements loaded : {len(dataset)}")
    kfLength = np.arange(len(dataset))

    best_acc, best_path = 0.0, Path("../models/best.pt")
    # df = None # For now
    os.makedirs(os.path.join(args.output_dir, args.version, "logs"), exist_ok=True)
    fold_results, train_df, val_df = [], [], []

    # Kfold dataset splitting
    # Will potentially be changed depending on how the dataset is read in
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(kfLength)):
        # Get the test and training data
        print(f"================================= FOLD {fold+1} =================================")
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.2,
            random_state=args.seed
        )

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        test_ds = Subset(dataset, test_idx)

        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
        test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4)

        # The arguments will probably have to be updated to work with the model
        model = TimesNet(args).to(DEVICE)
        # LSTM model, values are hardcoded in LSTM.py
        # model = LSTM().to(DEVICE)

        # Loss here

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # criterion = nn.CrossEntropyLoss()
        # Try this loss, regression task
        criterion = MAE_loss()
        # Training loop here

        for epoch in range(args.epochs):
            start_time = time.time()

            train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, DEVICE)
            val_loss, val_acc = validate(model, val_dl, criterion, DEVICE)

            end_time = time.time()
            epoch_mins = int((end_time - start_time) / 60)
            epoch_secs = int((end_time - start_time) % 60)

            print(f"EPOCH {epoch+1}/{args.epochs} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}"
                    f"| Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
            # From sample code given in class, save the most accurate model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_path)
                print(f"    [saved] {best_path} (acc={best_acc:.4f})")

if __name__ == "__main__":
    args = config_args.parse_args()
    run_training(args)