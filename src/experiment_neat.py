import numpy as np
import os
import sys
from torch import nn
import torch.cuda
from src.utils.experiment_helper import ExperimentNEATHelper

# ## Experiment focus
# - Number of epoch
# - Learning rates
# - Number of evolution epoch, i.e. number of generations
# - Maximum fitness value

# ==================================================================================

# Experiment parameters
"""
NOTE: Below configurations will lead to experiment failure -> insufficient RAM
The experiment was done in batches manually by running jobscripts of different hyperparameter values multiple times
"""
# ================== Simple ==================
# epoch_nums = np.linspace(start=50, stop=50, num=6, dtype=int).tolist()
# learning_rates = np.linspace(start=0.005, stop=0.005, num=1).tolist()
# evolution_epoches = np.linspace(
#     start=2, stop=2, num=1, dtype=int).tolist()
# max_fitnesses = np.linspace(start=20, stop=20, num=1, dtype=int).tolist()
# ================== Actual ==================
# NOTE: Process will always be killed due to insufficient memory
epoch_nums = np.linspace(start=50, stop=150, num=3, dtype=int).tolist()
learning_rates = np.linspace(start=0.001, stop=0.001, num=1).tolist()
evolution_epoches = np.linspace(
    start=10, stop=50, num=3, dtype=int).tolist()
max_fitnesses = np.linspace(start=30, stop=70, num=3, dtype=int).tolist()
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
# Expect value-range is (0, 1) or None (no validation set)
validation_size = None
random_state = 40
batch_size = 32

# NEAT ensemble parameters
"""
These are dummy parameters for this program
They are only useful when you have an ensemble of NEAT models
"""
rank = 0
device = torch.device(f"cuda:{rank}")
device_index = 0

# For MLP base learner, i.e. the housing model
# Must be equal to number of input features
input_size = len(housing_column_names[:-1])
hidden_sizes = [64, 64]
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
loss_function = nn.MSELoss()
# For training model
epoch_num_per_log = 20  # Number of epoches to log once
repository_experiment_path = "data/experiment/neat"
repository_experiment_model_path = f"{repository_experiment_path}/model"
repository_experiment_csv_path = f"{repository_experiment_path}/csv"
experiment_csv_name = f"experiment_neat"
repository_experiment_img_path = f"{repository_experiment_path}/img"
experiment_img_name = f"experiment_neat"

# For experiment
image_resolution = 300


if __name__ == "__main__":
    print(f"Running experiment of NEAT ...", flush=True)

    print(f"Using device {device} ...", flush=True)

    if torch.cuda.is_available():
        assert device.type == "cuda", f"Invalid device (please use CUDA): {device}"
    else:
        # We do not run model
        print(f"Cannot find cuda to run model ...", flush=True)
        print(f"Exiting programme ...", flush=True)

        sys.exit(1)

    config_path = os.path.join(os.path.dirname(
        __file__), os.path.join("../", "config-housing-neat.ini"))

    helper = ExperimentNEATHelper(
        test_size=test_size,
        random_state=random_state,
        batch_size=batch_size,
        experiment_model_repository_path=repository_experiment_model_path,
        config_path=config_path,
        epoch_nums=epoch_nums,
        learning_rates=learning_rates,
        evolution_epoches=evolution_epoches,
        max_fitnesses=max_fitnesses,
        validation_size=validation_size
    )

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
        loss_function=loss_function,
        data_loader_train=data_loader_train,
        data_loader_test=data_loader_test,
        epoch_num_per_log=epoch_num_per_log
    )

    # Save experiment data
    helper.save_csv(
        repository_experiment_csv_path=repository_experiment_csv_path,
        csv_name=experiment_csv_name
    )
    print(f"Experiment data (.csv) is saved ...", flush=True)
    helper.plot_graph(
        repository_experiment_img_path=repository_experiment_img_path,
        img_name=experiment_img_name,
        save_img=True,
        image_resolution=image_resolution,
        # There's no need to show the image, which might actually block executions of programs
        show_img=False
    )
    print(f"Experiment images are saved ...", flush=True)

    print(f"NEAT experiment finishes running ...", flush=True)
