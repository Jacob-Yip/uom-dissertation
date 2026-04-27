import numpy as np
from torch import nn
import torch.cuda
from src.utils.experiment_helper import ExperimentTraditionalHelper

# ## Experiment focus
# - Ensemble size
# - Number of epoch
# - Learning rates

# ==================================================================================

# Experiment parameters
# ================== Simple ==================
# ensemble_sizes = np.linspace(start=2, stop=2, num=1, dtype=int).tolist()
# epoch_nums = np.linspace(start=50, stop=50, num=1, dtype=int).tolist()
# learning_rates = np.linspace(start=0.002, stop=0.002, num=1).tolist()
# ================== Actual ==================
ensemble_sizes = np.linspace(start=2, stop=16, num=8, dtype=int).tolist()
epoch_nums = np.linspace(start=50, stop=300, num=6, dtype=int).tolist()
learning_rates = np.linspace(start=0.0005, stop=0.002, num=4).tolist()
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
hidden_sizes = [6]
activation_type = nn.ReLU()
activations = None
output_size = 1
dropout_rate = 0.5
dropout_indices = [0]
model_configurations = {
    "input_size": input_size,
    "hidden_sizes": hidden_sizes,
    "activation_type": activation_type,
    "activations": activations,
    "output_size": output_size,
    "dropout_rate": dropout_rate,
    "dropout_indices": dropout_indices
}
# To use a customised loss function, create a Python function and pass the function instance to train_models()
ensemble_loss_function = nn.MSELoss()
# Give a tensor instead of a scalar as the final output
individual_loss_function = nn.MSELoss(reduction="none")
# For training model
epoch_num_per_log = 20  # Number of epoches to log once
repository_experiment_path = "data/experiment/traditional"
repository_experiment_model_path = f"{repository_experiment_path}/model"
repository_experiment_csv_path = f"{repository_experiment_path}/csv"
experiment_csv_name = f"experiment_traditional"
repository_experiment_img_path = f"{repository_experiment_path}/img"
experiment_img_name = f"experiment_traditional"

# For experiment
image_resolution = 300


if __name__ == "__main__":
    print(f"Running experiment of traditional ensemble model ...")

    # Run experiment
    print(f"Number of CUDA devices detected: {device_num}")

    helper = ExperimentTraditionalHelper(
        test_size=test_size,
        random_state=random_state,
        batch_size=batch_size,
        experiment_model_repository_path=repository_experiment_model_path,
        master_address=master_address,
        master_port=master_port,
        device_num=device_num,
        dist_backend=dist_backend,
        ensemble_sizes=ensemble_sizes,
        epoch_nums=epoch_nums,
        learning_rates=learning_rates,
        validation_size=validation_size
    )
    print(f"Using device {helper.device} ...")

    data_loader_train, data_loader_test = helper.get_data_loaders(
        filename=housing_filename,
        column_names=housing_column_names,
        output_column_name=housing_output_column_name,
        normalise_data=True,
        remove_outliers=False
    )

    # Run experiment
    helper.experiment(
        model_configurations=model_configurations,
        ensemble_loss_function=ensemble_loss_function,
        individual_loss_function=individual_loss_function,
        data_loader_train=data_loader_train,
        data_loader_test=data_loader_test,
        epoch_num_per_log=epoch_num_per_log
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

    print(f"Traditional ensemble model experiment finishes running ...")
