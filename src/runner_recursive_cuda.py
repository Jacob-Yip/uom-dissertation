import neat
import numpy as np
import os
from torch import nn
import torch.cuda
from src.neat.neat_ncl_genome import NEATNCLGenome
from src.utils import data_helper
from src.utils.experiment.experiment_runner_recursive import ExperimentRunnerRecursive

# ## Experiment focus
"""
The recursive data structure indicating what to create in the the below form: 
    {
        "<ensemble_net_type>": [<sub_architecture_dict>]
        "voter": <voter_type>
    }
    - If <architecture_dict> does not have the key "voter", it is for the base case. Otherwise, it is for the step case, i.e. RecursiveEnsembleNet
"""
# - Model Architecture

# ==================================================================================


# Global configuration
housing_filename = "data/housing.csv"
housing_column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
                        "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
housing_output_column_name = housing_column_names[-1]

# Hardware configuration
device_num_cuda = torch.cuda.device_count()  # Number of CPUs
device_num_cpu = os.cpu_count()  # Number of CPUs
# Mainly for voters in the recursive ensemble model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For splitting dataset into training and testing data
test_size = 0.2  # Expect value-range is (0, 1)
# Expect value-range is (0, 1) or None (no validation set)
validation_size = 0.25
random_state = 40
# TODO: Update me if applicable
# For traditional/(static NCL)
# batch_size = 32
# For NEAT NCL
# NOTE: Technically, 506 is more than the actual batch_size because 506 is the total size of the training + testing dataset but this works
batch_size = 506  # The full data in 1 batch
# Number of epoches to do 1 validation testing to see whether we should stop training early
validation_cycle = 50
# Number of epoches beforehand to check whether our validation loss has gotten worse
validation_epoch_window = 500

# For ensemble model
master_address = "localhost"
master_port = "51123"
dist_backend = "nccl"  # GPU: nccl; CPU (not supported): gloo
gloo_socket_filename = "lo0"  # For CPU (not supported)

# NEAT parameters
config_filename = f"config-housing-neat-ncl.ini"

# For base learner, i.e. the housing model
# Must be equal to number of input features
input_size = len(housing_column_names[:-1])
# Picked by me cause it is one of the best
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
voter = f"nn"
# To use a customised loss function, create a Python function and pass the function instance to train_models()
loss_function = nn.MSELoss()
# For training model
epoch_num_per_log = 20  # Number of epoches to log once
learning_rate = 0.001
train_epoch = 400
# Static NCL
# Picked by me cause it is one of the best
correlation_penalty_coefficient = 0.7
# NEAT NCL
min_correlation_penalty_coefficient = 0.1
max_correlation_penalty_coefficient = 0.9
# Picked by me
population_size = 50
# Number of evolutions
# NOTE: This is set to (evolution_epoch + 1) because I also want the evolution with index evolution_epoch (which is the (evolution_epoch + 1)st evolution due to NEAT Python implementation) in my output csv file
evoluatin_epoch = 401
# Number of epoches to output 1 target expriment dataset
evolution_epoch_step_size = 10
absolute_config_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), config_filename)

print(f"Using config file at {absolute_config_path}")

config = neat.Config(NEATNCLGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation, absolute_config_path)


# File system metadata
project_root_folder_path = os.path.dirname(os.path.dirname(__file__))
repository_experiment_path = os.path.join(os.path.join(
    os.path.join(project_root_folder_path, f"data"), f"experiment"), f"recursive")
repository_experiment_csv_path = os.path.join(
    repository_experiment_path, f"csv")
experiment_csv_name = f"experiment_runner_recursive"
# TODO: Below code is legacy code (maybe) and should be removed -> use repository_experiment_path instead
relative_repository_experiment_path = "data/experiment/recursive"
repository_experiment_model_path = f"{relative_repository_experiment_path}/model"

# ==================================================================================
experiment_configurations = {
    "epoch_num_per_log": epoch_num_per_log,
    "loss_function": loss_function,
    "device": device,
    "device_num": device_num_cuda,
    "learning_rate": learning_rate,
    "dist_backend": dist_backend,
    "master_address": master_address,
    "master_port": master_port,
    "train_epoch": train_epoch,
    # Static NCL
    "correlation_penalty_coefficient": correlation_penalty_coefficient,
    # NEAT NCL
    "min_correlation_penalty_coefficient": min_correlation_penalty_coefficient,
    "max_correlation_penalty_coefficient": max_correlation_penalty_coefficient,
    "evolution_epoch": evoluatin_epoch,
    "evolution_epoch_step_size": evolution_epoch_step_size,
    "population_size": population_size,
    "model_repository_path": repository_experiment_model_path,
    "validation_cycle": validation_cycle,
    "validation_epoch_window": validation_epoch_window,
    "voter": voter
}
experiment_configurations.update(model_configurations)

# Experiment parameters
# ================== Actual ==================
architectures = [
    # Traditional + Arithmetic mean
    {
        "traditional": [
            {
                "traditional": [
                    # Base case
                    {
                        "ensemble_size": 3,
                        "model_configurations": model_configurations,
                        "model_repository_path": repository_experiment_model_path,
                        "ensemble_voter": "arithmetic_mean"
                    },
                    # Base case
                    {
                        "ensemble_size": 3,
                        "model_configurations": model_configurations,
                        "model_repository_path": repository_experiment_model_path,
                        "ensemble_voter": "arithmetic_mean"
                    }
                ],
                "voter": "arithmetic_mean"
            },
            {
                "traditional": [
                    # Base case
                    {
                        "ensemble_size": 3,
                        "model_configurations": model_configurations,
                        "model_repository_path": repository_experiment_model_path,
                        "ensemble_voter": "arithmetic_mean"
                    },
                    # Base case
                    {
                        "ensemble_size": 3,
                        "model_configurations": model_configurations,
                        "model_repository_path": repository_experiment_model_path,
                        "ensemble_voter": "arithmetic_mean"
                    }
                ],
                "voter": "arithmetic_mean"
            }
        ],
        "voter": "arithmetic_mean"
    },
    {
        "traditional": [
            # Base case
            {
                "ensemble_size": 3,
                "model_configurations": model_configurations,
                "model_repository_path": repository_experiment_model_path,
                "ensemble_voter": "arithmetic_mean"
            },
            # Base case
            {
                "ensemble_size": 3,
                "model_configurations": model_configurations,
                "model_repository_path": repository_experiment_model_path,
                "ensemble_voter": "arithmetic_mean"
            },
            # Base case
            {
                "ensemble_size": 3,
                "model_configurations": model_configurations,
                "model_repository_path": repository_experiment_model_path,
                "ensemble_voter": "arithmetic_mean"
            },
            # Base case
            {
                "ensemble_size": 3,
                "model_configurations": model_configurations,
                "model_repository_path": repository_experiment_model_path,
                "ensemble_voter": "arithmetic_mean"
            }
        ],
        "voter": "arithmetic_mean"
    },
    # Traditional + Median
    {
        "traditional": [
            {
                "traditional": [
                    # Base case
                    {
                        "ensemble_size": 3,
                        "model_configurations": model_configurations,
                        "model_repository_path": repository_experiment_model_path,
                        "ensemble_voter": "median"
                    },
                    # Base case
                    {
                        "ensemble_size": 3,
                        "model_configurations": model_configurations,
                        "model_repository_path": repository_experiment_model_path,
                        "ensemble_voter": "median"
                    }
                ],
                "voter": "median"
            },
            {
                "traditional": [
                    # Base case
                    {
                        "ensemble_size": 3,
                        "model_configurations": model_configurations,
                        "model_repository_path": repository_experiment_model_path,
                        "ensemble_voter": "median"
                    },
                    # Base case
                    {
                        "ensemble_size": 3,
                        "model_configurations": model_configurations,
                        "model_repository_path": repository_experiment_model_path,
                        "ensemble_voter": "median"
                    }
                ],
                "voter": "median"
            }
        ],
        "voter": "median"
    },
    {
        "traditional": [
            # Base case
            {
                "ensemble_size": 3,
                "model_configurations": model_configurations,
                "model_repository_path": repository_experiment_model_path,
                "ensemble_voter": "median"
            },
            # Base case
            {
                "ensemble_size": 3,
                "model_configurations": model_configurations,
                "model_repository_path": repository_experiment_model_path,
                "ensemble_voter": "median"
            },
            # Base case
            {
                "ensemble_size": 3,
                "model_configurations": model_configurations,
                "model_repository_path": repository_experiment_model_path,
                "ensemble_voter": "median"
            },
            # Base case
            {
                "ensemble_size": 3,
                "model_configurations": model_configurations,
                "model_repository_path": repository_experiment_model_path,
                "ensemble_voter": "median"
            }
        ],
        "voter": "median"
    },
    # Traditional + Neural network
    {
        "traditional": [
            {
                "traditional": [
                    # Base case
                    {
                        "ensemble_size": 3,
                        "model_configurations": model_configurations,
                        "model_repository_path": repository_experiment_model_path,
                        "ensemble_voter": "nn"
                    },
                    # Base case
                    {
                        "ensemble_size": 3,
                        "model_configurations": model_configurations,
                        "model_repository_path": repository_experiment_model_path,
                        "ensemble_voter": "nn"
                    }
                ],
                "voter": "nn"
            },
            {
                "traditional": [
                    # Base case
                    {
                        "ensemble_size": 3,
                        "model_configurations": model_configurations,
                        "model_repository_path": repository_experiment_model_path,
                        "ensemble_voter": "nn"
                    },
                    # Base case
                    {
                        "ensemble_size": 3,
                        "model_configurations": model_configurations,
                        "model_repository_path": repository_experiment_model_path,
                        "ensemble_voter": "nn"
                    }
                ],
                "voter": "nn"
            }
        ],
        "voter": "nn"
    },
    {
        "traditional": [
            # Base case
            {
                "ensemble_size": 3,
                "model_configurations": model_configurations,
                "model_repository_path": repository_experiment_model_path,
                "ensemble_voter": "nn"
            },
            # Base case
            {
                "ensemble_size": 3,
                "model_configurations": model_configurations,
                "model_repository_path": repository_experiment_model_path,
                "ensemble_voter": "nn"
            },
            # Base case
            {
                "ensemble_size": 3,
                "model_configurations": model_configurations,
                "model_repository_path": repository_experiment_model_path,
                "ensemble_voter": "nn"
            },
            # Base case
            {
                "ensemble_size": 3,
                "model_configurations": model_configurations,
                "model_repository_path": repository_experiment_model_path,
                "ensemble_voter": "nn"
            }
        ],
        "voter": "nn"
    },
    # # Static NCL + Arithmetic mean
    # {
    #     "static_ncl": [
    #         {
    #             "static_ncl": [
    #                 # Base case
    #                 {
    #                     "ensemble_size": 3,
    #                     "model_configurations": model_configurations,
    #                     "model_repository_path": repository_experiment_model_path,
    #                     "ensemble_voter": "arithmetic_mean"
    #                 },
    #                 # Base case
    #                 {
    #                     "ensemble_size": 3,
    #                     "model_configurations": model_configurations,
    #                     "model_repository_path": repository_experiment_model_path,
    #                     "ensemble_voter": "arithmetic_mean"
    #                 }
    #             ],
    #             "voter": "arithmetic_mean"
    #         },
    #         {
    #             "static_ncl": [
    #                 # Base case
    #                 {
    #                     "ensemble_size": 3,
    #                     "model_configurations": model_configurations,
    #                     "model_repository_path": repository_experiment_model_path,
    #                     "ensemble_voter": "arithmetic_mean"
    #                 },
    #                 # Base case
    #                 {
    #                     "ensemble_size": 3,
    #                     "model_configurations": model_configurations,
    #                     "model_repository_path": repository_experiment_model_path,
    #                     "ensemble_voter": "arithmetic_mean"
    #                 }
    #             ],
    #             "voter": "arithmetic_mean"
    #         }
    #     ],
    #     "voter": "arithmetic_mean"
    # },
    # {
    #     "static_ncl": [
    #         # Base case
    #         {
    #             "ensemble_size": 3,
    #             "model_configurations": model_configurations,
    #             "model_repository_path": repository_experiment_model_path,
    #             "ensemble_voter": "arithmetic_mean"
    #         },
    #         # Base case
    #         {
    #             "ensemble_size": 3,
    #             "model_configurations": model_configurations,
    #             "model_repository_path": repository_experiment_model_path,
    #             "ensemble_voter": "arithmetic_mean"
    #         },
    #         # Base case
    #         {
    #             "ensemble_size": 3,
    #             "model_configurations": model_configurations,
    #             "model_repository_path": repository_experiment_model_path,
    #             "ensemble_voter": "arithmetic_mean"
    #         },
    #         # Base case
    #         {
    #             "ensemble_size": 3,
    #             "model_configurations": model_configurations,
    #             "model_repository_path": repository_experiment_model_path,
    #             "ensemble_voter": "arithmetic_mean"
    #         }
    #     ],
    #     "voter": "arithmetic_mean"
    # },
    # # Static NCL + Median
    # {
    #     "static_ncl": [
    #         {
    #             "static_ncl": [
    #                 # Base case
    #                 {
    #                     "ensemble_size": 3,
    #                     "model_configurations": model_configurations,
    #                     "model_repository_path": repository_experiment_model_path,
    #                     "ensemble_voter": "median"
    #                 },
    #                 # Base case
    #                 {
    #                     "ensemble_size": 3,
    #                     "model_configurations": model_configurations,
    #                     "model_repository_path": repository_experiment_model_path,
    #                     "ensemble_voter": "median"
    #                 }
    #             ],
    #             "voter": "median"
    #         },
    #         {
    #             "static_ncl": [
    #                 # Base case
    #                 {
    #                     "ensemble_size": 3,
    #                     "model_configurations": model_configurations,
    #                     "model_repository_path": repository_experiment_model_path,
    #                     "ensemble_voter": "median"
    #                 },
    #                 # Base case
    #                 {
    #                     "ensemble_size": 3,
    #                     "model_configurations": model_configurations,
    #                     "model_repository_path": repository_experiment_model_path,
    #                     "ensemble_voter": "median"
    #                 }
    #             ],
    #             "voter": "median"
    #         }
    #     ],
    #     "voter": "median"
    # },
    # {
    #     "static_ncl": [
    #         # Base case
    #         {
    #             "ensemble_size": 3,
    #             "model_configurations": model_configurations,
    #             "model_repository_path": repository_experiment_model_path,
    #             "ensemble_voter": "median"
    #         },
    #         # Base case
    #         {
    #             "ensemble_size": 3,
    #             "model_configurations": model_configurations,
    #             "model_repository_path": repository_experiment_model_path,
    #             "ensemble_voter": "median"
    #         },
    #         # Base case
    #         {
    #             "ensemble_size": 3,
    #             "model_configurations": model_configurations,
    #             "model_repository_path": repository_experiment_model_path,
    #             "ensemble_voter": "median"
    #         },
    #         # Base case
    #         {
    #             "ensemble_size": 3,
    #             "model_configurations": model_configurations,
    #             "model_repository_path": repository_experiment_model_path,
    #             "ensemble_voter": "median"
    #         }
    #     ],
    #     "voter": "median"
    # },
    # # Static NCL + Neural network
    # {
    #     "static_ncl": [
    #         {
    #             "static_ncl": [
    #                 # Base case
    #                 {
    #                     "ensemble_size": 3,
    #                     "model_configurations": model_configurations,
    #                     "model_repository_path": repository_experiment_model_path,
    #                     "ensemble_voter": "nn"
    #                 },
    #                 # Base case
    #                 {
    #                     "ensemble_size": 3,
    #                     "model_configurations": model_configurations,
    #                     "model_repository_path": repository_experiment_model_path,
    #                     "ensemble_voter": "nn"
    #                 }
    #             ],
    #             "voter": "nn"
    #         },
    #         {
    #             "static_ncl": [
    #                 # Base case
    #                 {
    #                     "ensemble_size": 3,
    #                     "model_configurations": model_configurations,
    #                     "model_repository_path": repository_experiment_model_path,
    #                     "ensemble_voter": "nn"
    #                 },
    #                 # Base case
    #                 {
    #                     "ensemble_size": 3,
    #                     "model_configurations": model_configurations,
    #                     "model_repository_path": repository_experiment_model_path,
    #                     "ensemble_voter": "nn"
    #                 }
    #             ],
    #             "voter": "nn"
    #         }
    #     ],
    #     "voter": "nn"
    # },
    # {
    #     "static_ncl": [
    #         # Base case
    #         {
    #             "ensemble_size": 3,
    #             "model_configurations": model_configurations,
    #             "model_repository_path": repository_experiment_model_path,
    #             "ensemble_voter": "nn"
    #         },
    #         # Base case
    #         {
    #             "ensemble_size": 3,
    #             "model_configurations": model_configurations,
    #             "model_repository_path": repository_experiment_model_path,
    #             "ensemble_voter": "nn"
    #         },
    #         # Base case
    #         {
    #             "ensemble_size": 3,
    #             "model_configurations": model_configurations,
    #             "model_repository_path": repository_experiment_model_path,
    #             "ensemble_voter": "nn"
    #         },
    #         # Base case
    #         {
    #             "ensemble_size": 3,
    #             "model_configurations": model_configurations,
    #             "model_repository_path": repository_experiment_model_path,
    #             "ensemble_voter": "nn"
    #         }
    #     ],
    #     "voter": "nn"
    # },
]
# ==================================================================================


if __name__ == "__main__":
    print(f"Running runner of recursive ensemble model ...")

    # Run experiment
    print(f"Number of CUDA devices detected: {device_num_cuda}")

    experiment_configurations["config"] = config

    runner = ExperimentRunnerRecursive(
        configurations=experiment_configurations,
        architecture=architectures
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

    print(f"Recursive ensemble model runner finishes running ...")
