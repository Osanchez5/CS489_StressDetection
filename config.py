import argparse
config_args = argparse.ArgumentParser()

# Will be replaced with parameters needed by our specific model
#----------------------------------- Dataset Configs -----------------------------------#
config_args.add_argument('--dataset_dir', type = str, default = "./Dataset/", help = "The root directory of the dataset")
config_args.add_argument('--image_dir', type = str, default = "Images", help = "The root directory of the image files")
config_args.add_argument('--mask_dir', type = str, default = "Masks", help = "The root directory of the mask files")
config_args.add_argument('--csv_file', type = str, default = "dataset.csv", help = "CSV file name")

#----------------------------------- Preprocessing Configs -----------------------------------#
config_args.add_argument('--seed', type = int, default = 42, help = "seed for reproduciability")

#----------------------------------- Model Configs -----------------------------------#

#----------------------------------- Training Configs -----------------------------------#
config_args.add_argument('--epochs', type = int, default = 20, help = "# of epochs")
config_args.add_argument('--num_folds', type = int, default = 5, help = "# of folds")
config_args.add_argument('--batch', type = int, default = 32, help = "batch size")
config_args.add_argument('--patience', type = int, default = 5, help = "# of epochs before early stopping")
config_args.add_argument('--lr', type = float, default = 1e-5, help = "learning rate")
config_args.add_argument('--output_dir', type = str, default = "./outputs", help = "The root directory of the outputs")
config_args.add_argument('--device_name', type = str, default = "0", help = "The available gpu in the cluster, check with nvidia_smi")
config_args.add_argument('--version', type = str, default = "Model_1", help = "The name of the version run (creates a directory based on the name).")


# Model specific arguments
config_args.add_argument('--task_name', type=str, default='forecasting')
config_args.add_argument('--seq_len', type=int, default=64)
config_args.add_argument('--label_len', type=int, default=1)
config_args.add_argument('--c_out', type=int, default=1)

# config_args.add_argument('--task_name', type=str, default="classification")
# config_args.add_argument('--seq_len', type=int, default=64)
# config_args.add_argument('--label_len', type=int, default=1)
config_args.add_argument('--pred_len', type=int, default=0)
config_args.add_argument('--enc_in', type=int, default=1)
config_args.add_argument('--d_model', type=int, default=64)
config_args.add_argument('--embd', type=str, default='fixed')
config_args.add_argument('--freq', type=str, default='s')
config_args.add_argument('--dropout', type=float, default=0.1)
config_args.add_argument('--e_layers', type=int, default=3)
config_args.add_argument('--d_ff', type=int, default=64)
config_args.add_argument( "--num_kernels", type=int, default=6)
config_args.add_argument('--top_k', type=int, default=3)
