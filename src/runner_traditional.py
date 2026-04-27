import numpy as np
import os
from torch import nn
import torch.cuda
from src.utils import data_helper
from src.utils.experiment.experiment_runner_traditional import ExperimentRunnerTraditional

# ## Experiment focus
# - Ensemble size
# - Number of epoch
# - Learning rates

# ==================================================================================


# Experiment parameters
# ================== Simple ==================
# ensemble_sizes = np.linspace(start=2, stop=2, num=1, dtype=int).tolist()
# epoch_nums = np.linspace(start=400, stop=400, num=1, dtype=int).tolist()
# learning_rates = np.linspace(start=0.001, stop=0.001, num=1).tolist()
# hidden_sizes = [
#     [2, 2, 2]
# ]
# ================== Actual ==================
ensemble_sizes = np.linspace(start=2, stop=16, num=15, dtype=int).tolist()
epoch_nums = np.linspace(start=200, stop=400, num=2, dtype=int).tolist()
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
device_num = torch.cuda.device_count()  # Number of CPUs/GPUs

# For splitting dataset into training and testing data
test_size = 0.2  # Expect value-range is (0, 1)
# Expect value-range is (0, 1) or None (no validation set)
validation_size = None
random_state = 40
batch_size = 32

# For ensemble model
master_address = "localhost"
master_port = "51123"
dist_backend = "nccl"  # GPU: nccl; CPU (not supported): gloo
gloo_socket_filename = "lo0"  # For CPU (not supported)

# For base learner, i.e. the housing model
# Must be equal to number of input features
input_size = len(housing_column_names[:-1])
activation_type = nn.ReLU()
activations = None
output_size = 1
dropout_rate = 0.5
dropout_indices = [0]
model_configurations = {
    "input_size": input_size,
    # Will be updated in experiment_runner_traditional.py
    # "hidden_sizes": hidden_sizes,
    "activation_type": activation_type,
    "activations": activations,
    "output_size": output_size,
    "dropout_rate": dropout_rate,
    "dropout_indices": dropout_indices
}
voter = f"nn"
# To use a customised loss function, create a Python function and pass the function instance to train_models()
loss_function = nn.MSELoss()
# For training model
epoch_num_per_log = 20  # Number of epoches to log once


# File system metadata
project_root_folder_path = os.path.dirname(os.path.dirname(__file__))
repository_experiment_path = os.path.join(os.path.join(
    os.path.join(project_root_folder_path, f"data"), f"experiment"), f"traditional")
repository_experiment_csv_path = os.path.join(
    repository_experiment_path, f"csv")
experiment_csv_name = f"experiment_runner_traditional"
# TODO: Below code is legacy code (maybe) and should be removed -> use repository_experiment_path instead
relative_repository_experiment_path = "data/experiment/traditional"
repository_experiment_model_path = f"{relative_repository_experiment_path}/model"

# ==================================================================================
experiment_configurations = {
    "epoch_num_per_log": epoch_num_per_log,
    "loss_function": loss_function,
    "device_num": device_num,
    "dist_backend": dist_backend,
    "master_address": master_address,
    "master_port": master_port,
    "model_repository_path": repository_experiment_model_path, 
    "voter": voter
}
experiment_configurations.update(model_configurations)


if __name__ == "__main__":
    print(f"Running runner of traditional ensemble model ...")

    # Run experiment
    print(f"Number of CUDA devices detected: {device_num}")

    runner = ExperimentRunnerTraditional(
        configurations=experiment_configurations,
        ensemble_size=ensemble_sizes,
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

    print(f"Traditional ensemble model runner finishes running ...")
