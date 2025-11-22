import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
# from losses import DiceLoss, CrossEntropyLoss # Added CE Loss to losses.py
from dataset import CSVDataset
from model import TimesNet
from train import train_one_epoch, validate, test
from config import config_args
from typing import Any


def run_training(args: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Your model is running on {DEVICE}...\n")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



    # Load dataset
    # Load in the subject info csv file? We'd only care about the info column
    csv_file_path = None # For now
    df = None # For now
    os.makedirs(os.path.join(args.output_dir, args.version, "logs"), exist_ok=True)
    


    fold_results, train_df, val_df = [], [], []

    # Get the total length of the dataframe
    kfLength = np.arange(len(df))

    # Kfold dataset splitting
    # Will potentially be changed depending on how the dataset is read in
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(kfLength)):
        # Get the test and training data
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=args.seed)

        train_ds = CSVDataset(args, train_df)
        val_ds = CSVDataset(args, val_df)
        test_ds = CSVDataset(args, test_df)

        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=1)
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=1)
        test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=1)

        # The arguments will probably have to be updated to work with the model
        model = TimesNet(args).to(DEVICE)

        # Loss here

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Training loop here



if __name__ == "__main__":
    args = config_args.parse_args()
    run_training(args)