import neat
import numpy as np
import os
from torch import nn
from src.neat.neat_ncl_genome import NEATNCLGenome
from src.utils import data_helper
from src.utils.experiment.experiment_runner_neat_ncl import ExperimentRunnerNEATNCL

# ## Experiment focus
# - Ensemble size
# - Learning rates

# ==================================================================================


# Experiment parameters
# ================== Simple ==================
# ensemble_sizes = np.linspace(start=2, stop=2, num=1, dtype=int).tolist()
# learning_rates = np.linspace(start=0.001, stop=0.001, num=1).tolist()
# ================== Actual ==================
ensemble_sizes = np.linspace(start=50, stop=250, num=5, dtype=int).tolist()
learning_rates = np.linspace(start=0.001, stop=0.001, num=1).tolist()
# ==================================================================================

# Global configuration
housing_filename = "data/housing.csv"
housing_column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
                        "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
housing_output_column_name = housing_column_names[-1]

# Hardware configuration
device_num = os.cpu_count()  # Number of CPUs/GPUs

# For splitting dataset into training and testing data
test_size = 0.2  # Expect value-range is (0, 1)
# Expect value-range is (0, 1) or None (no validation set)
validation_size = None
random_state = 40
# NOTE: Technically, 506 is more than the actual batch_size because 506 is the total size of the training + testing dataset but this works
batch_size = 506  # The full data in 1 batch

# NEAT parameters
config_filename = f"config-housing-neat-ncl.ini"

# For ensemble model
voter = f"median"
# To use a customised loss function, create a Python function and pass the function instance to train_models()
loss_function = nn.MSELoss()
min_lambda = 0.1
max_lambda = 0.9
# Number of evolutions
# NOTE: This is set to (evolution_epoch + 1) because I also want the evolution with index evolution_epoch (which is the (evolution_epoch + 1)st evolution due to NEAT Python implementation) in my output csv file
evoluatin_epoch = 401
# Number of epoches to output 1 target expriment dataset
evolution_epoch_step_size = 10


# File system metadata
project_root_folder_path = os.path.dirname(os.path.dirname(__file__))
repository_experiment_path = os.path.join(os.path.join(
    os.path.join(project_root_folder_path, f"data"), f"experiment"), f"neat-ncl")
repository_experiment_csv_path = os.path.join(
    repository_experiment_path, f"csv")
repository_experiment_model_path = os.path.join(
    repository_experiment_path, f"model")
experiment_csv_name = f"experiment_runner_neat_ncl"

# ==================================================================================
experiment_configurations = {
    "loss_function": loss_function,
    "min_correlation_penalty_coefficient": min_lambda,
    "max_correlation_penalty_coefficient": max_lambda,
    "evolution_epoch": evoluatin_epoch,
    "evolution_epoch_step_size": evolution_epoch_step_size,
    "model_repository_path": repository_experiment_model_path, 
    "voter": voter
}


if __name__ == "__main__":
    print(f"Running runner of NEAT NCL ensemble model ...")

    # Run experiment
    print(f"Number of CPU cores detected: {device_num}")

    absolute_config_path = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), config_filename)

    print(f"Using config file at {absolute_config_path}")

    config = neat.Config(NEATNCLGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, absolute_config_path)

    experiment_configurations["config"] = config

    runner = ExperimentRunnerNEATNCL(
        configurations=experiment_configurations,
        ensemble_size=ensemble_sizes,
        learning_rate=learning_rates
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

    print(f"NEAT NCL ensemble model runner finishes running ...")
