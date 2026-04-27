import numpy as np
import os
import sys
from torch import nn
import torch.cuda
from src.utils import data_helper
from src.utils.experiment.experiment_runner_mlp import ExperimentRunnerMLP

# ## Experiment focus
# - Number of epoch
# - Learning rates

# ==================================================================================

# Experiment parameters
# ================== Simple ==================
# epoch_nums = np.linspace(start=10, stop=10, num=1, dtype=int).tolist()
# learning_rates = np.linspace(start=0.001, stop=0.001, num=1).tolist()
# hidden_sizes = [
#     [2, 2, 2]
# ]
# ================== Actual ==================
epoch_nums = np.linspace(start=10, stop=1000, num=100, dtype=int).tolist()
learning_rates = np.linspace(start=0.001, stop=0.001, num=1).tolist()
hidden_sizes = [
    [2],
    # [2, 2],
    [2, 2, 2],
    # [2, 2, 2, 2],
    # [2, 2, 2, 2, 2],
    [2, 2, 2, 2, 2, 2],
    # [4],
    [6],
    # [8],
    # [10],
    [12]
]
# ==================================================================================

# Global configuration
housing_filename = "data/housing.csv"
housing_column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
                        "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
housing_output_column_name = housing_column_names[-1]

# Hardware configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hardware configuration
device_num = torch.cuda.device_count()  # Number of CPUs/GPUs

# For splitting dataset into training and testing data
test_size = 0.2  # Expect value-range is (0, 1)
validation_size = None  # This means that we do not need validation dataset
random_state = 40
batch_size = 32

# For base learner, i.e. the housing model
# Must be equal to number of input features
input_size = len(housing_column_names[:-1])
activation_type = nn.ReLU()
activations = None
output_size = 1
dropout_rate = None
dropout_indices = []
model_configurations = {
    "input_size": input_size,
    # Will be updated in experiment_runner_mlp.py
    # "hidden_sizes": hidden_sizes,
    "activation_type": activation_type,
    "activations": activations,
    "output_size": output_size,
    "dropout_rate": dropout_rate,
    "dropout_indices": dropout_indices
}
# To use a customised loss function, create a Python function and pass the function instance to train_models()
epoch_num_per_log = 20  # Number of epoches to log once
loss_function = nn.MSELoss()

# File system metadata
project_root_folder_path = os.path.dirname(os.path.dirname(__file__))
repository_experiment_path = os.path.join(os.path.join(
    os.path.join(project_root_folder_path, f"data"), f"experiment"), f"mlp")
repository_experiment_csv_path = os.path.join(
    repository_experiment_path, f"csv")
experiment_csv_name = f"experiment_runner_mlp"

# ==================================================================================
experiment_configurations = {
    "epoch_num_per_log": epoch_num_per_log,
    "loss_function": loss_function,
    "device": device
}
experiment_configurations.update(model_configurations)


if __name__ == "__main__":
    print(f"Running runner of MLP ...")

    print(f"Using device {device} ...")

    if torch.cuda.is_available():
        assert device.type == "cuda", f"Invalid device (please use CUDA): {device}"
    else:
        # We do not run model
        print(f"Cannot find cuda to run model ...")
        print(f"Exiting programme ...")

        sys.exit(1)

    runner = ExperimentRunnerMLP(
        configurations=experiment_configurations,
        # Experiment parameters
        epoch_num=epoch_nums,
        learning_rate=learning_rates,
        hidden_size=hidden_sizes
    )

    data_loader_train, data_loader_test = data_helper.get_data_loaders(
        filename=housing_filename,
        column_names=housing_column_names,
        output_column_name=housing_output_column_name,
        test_size=test_size,
        random_state=random_state,
        batch_size=batch_size,
        validation_size=validation_size,
        normalise_data=True,
        remove_outliers=False
    )

    # Run experiment
    runner.run(
        data_loader_train=data_loader_train,
        data_loader_validation=None,
        data_loader_test=data_loader_test
    )

    # Save experiment data
    runner.save_csv(
        absolute_folder_path=repository_experiment_csv_path,
        csv_file_name=experiment_csv_name
    )
    print(f"Experiment data (.csv) is saved ...")

    print(f"Runner of MLP finishes running ...")
