import neat
import numpy as np
import os
from torch import nn
import torch.cuda
from src.utils.experiment_helper import ExperimentNEATNCLHelper

# ## Experiment focus
# - Values of lambda, i.e. correlation penalty coefficient
# - Ensemble size
# - Number of epoch
# - Learning rates
# - Evolution epoch
# - Max fitness

# ==================================================================================

# Experiment parameters
# ================== Simple ==================
# correlation_penalty_coefficients = np.linspace(
#     start=0.5, stop=0.5, num=1).tolist()  # Expect value [0, 1]
# ensemble_sizes = np.linspace(start=4, stop=4, num=1, dtype=int).tolist()
# epoch_nums = np.linspace(start=50, stop=50, num=1, dtype=int).tolist()
# learning_rates = np.linspace(start=0.001, stop=0.001, num=1).tolist()
# evolution_epoches = np.linspace(start=50, stop=50, num=1, dtype=int).tolist()
# max_fitnesses = np.linspace(start=10, stop=50, num=3).tolist()
# ================== Actual ==================
correlation_penalty_coefficients = np.linspace(
    start=0, stop=1, num=5).tolist()  # Expect value [0, 1]
ensemble_sizes = np.linspace(start=2, stop=4, num=2, dtype=int).tolist()
epoch_nums = np.linspace(start=100, stop=100, num=1, dtype=int).tolist()
learning_rates = np.linspace(start=0.001, stop=0.001, num=1).tolist()
evolution_epoches = np.linspace(start=110, stop=170, num=3, dtype=int).tolist()
max_fitnesses = np.linspace(start=10, stop=50, num=3).tolist()
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

# NEAT parameters
# Number of evolution generations
evolution_epoch = 50
# TODO: Update this hyperparameter to find the best value
max_fitness = 50
config_filename = f"config-housing-neat-ncl.ini"

# NCL ensemble parameters
master_address = "localhost"
master_port = "51123"
dist_backend = "nccl"  # GPU: nccl; CPU (not supported): gloo
gloo_socket_filename = "lo0"  # For CPU (not supported)

# NCL base learner, i.e. the housing model
# Must be equal to number of input features
input_size = len(housing_column_names[:-1])
hidden_sizes = [64, 64]
activation_type = nn.ReLU()
activations = None
output_size = 1
dropout_rate = 0.5
dropout_indices = [0]
# NOTE: Not for NEAT-NCL -> for static-NCL and traditional-ensemble
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
repository_experiment_path = "data/experiment/neat-ncl"
repository_experiment_model_path = f"{repository_experiment_path}/model"
repository_experiment_csv_path = f"{repository_experiment_path}/csv"
experiment_csv_name = f"experiment_neat_ncl"
repository_experiment_img_path = f"{repository_experiment_path}/img"
experiment_img_name = f"experiment_neat_ncl"

# For experiment
image_resolution = 300


if __name__ == "__main__":
    print(f"Running experiment of NEAT NCL ensemble model ...")

    # Run experiment
    print(f"Number of CUDA devices detected: {device_num}")

    config_path = os.path.join(os.path.dirname(
        __file__), os.path.join("../", config_filename))

    print(f"Using config file at {config_path}")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    helper = ExperimentNEATNCLHelper(
        test_size=test_size,
        random_state=random_state,
        batch_size=batch_size,
        experiment_model_repository_path=repository_experiment_model_path,
        master_address=master_address,
        master_port=master_port,
        device_num=device_num,
        dist_backend=dist_backend,
        config=config,
        correlation_penalty_coefficients=correlation_penalty_coefficients,
        ensemble_sizes=ensemble_sizes,
        epoch_nums=epoch_nums,
        learning_rates=learning_rates,
        evolution_epoches=evolution_epoches,
        max_fitnesses=max_fitnesses,
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

    print(f"NEAT NCL ensemble model experiment finishes running ...")
