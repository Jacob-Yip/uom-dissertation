import numpy as np
import os
from torch import nn
import torch.cuda
from src.utils import data_helper
from src.utils.experiment.experiment_runner_static_ncl import ExperimentRunnerStaticNCL

# ## Experiment focus
# - Values of lambda, i.e. correlation penalty coefficient
# - Ensemble size
# - Number of epoch
# - Learning rates

# ==================================================================================

# Experiment parameters
# ================== Simple ==================
# correlation_penalty_coefficients = np.linspace(
#     start=0.5, stop=0.5, num=1).tolist()  # Expect value [0, 1]
# ensemble_sizes = np.linspace(start=4, stop=4, num=1, dtype=int).tolist()
# epoch_nums = np.linspace(start=400, stop=400, num=1, dtype=int).tolist()
# learning_rates = np.linspace(start=0.001, stop=0.001, num=1).tolist()
# hidden_sizes = [
#     [2, 2, 2]
# ]
# ================== Actual ==================
correlation_penalty_coefficients = np.linspace(
    start=0, stop=1.5, num=16).tolist()  # Expect value [0, 1]
ensemble_sizes = np.linspace(start=2, stop=16, num=15, dtype=int).tolist()
epoch_nums = np.linspace(start=400, stop=400, num=1, dtype=int).tolist()
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
validation_size = 0.25
random_state = 40
batch_size = 32
# Number of epoches to do 1 validation testing to see whether we should stop training early
validation_cycle = 50
# Number of epoches beforehand to check whether our validation loss has gotten worse
validation_epoch_window = 500

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
    # Will be updated in experiment_runner_static_ncl.py
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
    os.path.join(project_root_folder_path, f"data"), f"experiment"), f"static-ncl")
repository_experiment_csv_path = os.path.join(
    repository_experiment_path, f"csv")
experiment_csv_name = f"experiment_runner_static_ncl"
# TODO: Below code is legacy code (maybe) and should be removed -> use repository_experiment_path instead
relative_repository_experiment_path = "data/experiment/static-ncl"
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
    "validation_cycle": validation_cycle,
    "validation_epoch_window": validation_epoch_window, 
    "voter": voter
}
experiment_configurations.update(model_configurations)


if __name__ == "__main__":
    print(f"Running runner of static NCL ensemble model ...")

    # Run experiment
    print(f"Number of CUDA devices detected: {device_num}")

    runner = ExperimentRunnerStaticNCL(
        configurations=experiment_configurations,
        correlation_penalty_coefficient=correlation_penalty_coefficients,
        ensemble_size=ensemble_sizes,
        epoch_num=epoch_nums,
        learning_rate=learning_rates,
        hidden_size=hidden_sizes
    )

    data_loader_train, data_loader_validation, data_loader_test = data_helper.get_data_loaders(
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
        data_loader_validation=data_loader_validation,
        data_loader_test=data_loader_test
    )

    # Save experiment data
    runner.save_csv(
        absolute_folder_path=repository_experiment_csv_path,
        csv_file_name=experiment_csv_name
    )
    print(f"Experiment data (.csv) is saved ...")

    print(f"Static NCL ensemble model runner finishes running ...")
