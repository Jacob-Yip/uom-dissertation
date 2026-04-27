import numpy as np
from torch import nn
import torch.cuda
from src.utils.experiment_helper import ExperimentStaticNCLHelper

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
# epoch_nums = np.linspace(start=50, stop=50, num=1, dtype=int).tolist()
# learning_rates = np.linspace(start=0.001, stop=0.001, num=1).tolist()
# hidden_sizes = [
#     [4, 4]
# ]
# ================== Actual ==================
correlation_penalty_coefficients = np.linspace(
    start=0.8, stop=1.5, num=8).tolist()  # Expect value [0, 1]
ensemble_sizes = np.linspace(start=2, stop=16, num=8, dtype=int).tolist()
epoch_nums = np.linspace(start=500, stop=500, num=1, dtype=int).tolist()
learning_rates = np.linspace(start=0.001, stop=0.001, num=1).tolist()
hidden_sizes = [
    [2],
    [2, 2],
    [2, 2, 2],
    [2, 2, 2, 2]
#     [2, 2, 2, 2, 2],
#     [2, 2, 2, 2, 2, 2]
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
    # Will be updated in experiment_helper.py
    # "hidden_sizes": hidden_sizes,
    "activation_type": activation_type,
    "activations": activations,
    "output_size": output_size,
    "dropout_rate": dropout_rate,
    "dropout_indices": dropout_indices
}
# To use a customised loss function, create a Python function and pass the function instance to train_models()
loss_function = nn.MSELoss()
# For training model
epoch_num_per_log = 20  # Number of epoches to log once
repository_experiment_path = "data/experiment/static-ncl"
repository_experiment_model_path = f"{repository_experiment_path}/model"
repository_experiment_csv_path = f"{repository_experiment_path}/csv"
experiment_csv_name = f"experiment_static_ncl"
repository_experiment_img_path = f"{repository_experiment_path}/img"
experiment_img_name = f"experiment_static_ncl"

# For experiment
image_resolution = 300


if __name__ == "__main__":
    print(f"Running experiment of static NCL ensemble model ...")

    # Run experiment
    print(f"Number of CUDA devices detected: {device_num}")

    helper = ExperimentStaticNCLHelper(
        test_size=test_size,
        random_state=random_state,
        batch_size=batch_size,
        experiment_model_repository_path=repository_experiment_model_path,
        master_address=master_address,
        master_port=master_port,
        device_num=device_num,
        dist_backend=dist_backend,
        correlation_penalty_coefficients=correlation_penalty_coefficients,
        ensemble_sizes=ensemble_sizes,
        epoch_nums=epoch_nums,
        learning_rates=learning_rates,
        hidden_sizes=hidden_sizes,
        validation_size=validation_size
    )
    print(f"Using device {helper.device} ...")

    data_loader_train, data_loader_validation, data_loader_test = helper.get_data_loaders(
        filename=housing_filename,
        column_names=housing_column_names,
        output_column_name=housing_output_column_name,
        normalise_data=True,
        remove_outliers=False
    )

    # Run experiment
    helper.experiment(
        model_configurations=model_configurations,
        loss_function=loss_function,
        data_loader_train=data_loader_train,
        data_loader_validation=data_loader_validation,
        data_loader_test=data_loader_test,
        epoch_num_per_log=epoch_num_per_log,
        validation_cycle=validation_cycle,
        validation_epoch_window=validation_epoch_window
    )

    # Save experiment data
    helper.save_csv(
        repository_experiment_csv_path=repository_experiment_csv_path,
        csv_name=experiment_csv_name
    )
    print(f"Experiment data (.csv) is saved ...")
    helper.plot_graph(
        repository_experiment_img_path=repository_experiment_img_path,
        img_name=experiment_img_name,
        save_img=True,
        image_resolution=image_resolution,
        # There's no need to show the image, which might actually block executions of programs
        show_img=False
    )
    print(f"Experiment images are saved ...")

    print(f"Static NCL ensemble model experiment finishes running ...")
