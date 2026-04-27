import neat
import os
from src.model.neat_ncl_ensemble_net import NEATNCLEnsembleNet
from src.neat.neat_ncl_genome import NEATNCLGenome
from src.utils import data_helper
import torch
from torch import nn

"""
NOTE: All neural networks will be run on CPUs instead of GPUs because NEAT networks have dynamic structures and cannot be optimised using GPUs
    - Good news, the improvement of moving NEAT networks to GPUs (using my algorithm) is negligible
        - One reason is that NEAT networks are usually small and GPUs are for big neural networks
"""

# ==================================================================================

# Global configuration
housing_filename = "data/housing.csv"
housing_column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
                        "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

# Hardware configuration
device_num = os.cpu_count()  # Number of CPUs/GPUs

# For splitting dataset into training and testing data
test_size = 0.2  # Expect value-range is (0, 1)
random_state = 40
batch_size = 32

# NEAT parameters
# Number of evolution generations
config_filename = f"config-housing-neat-ncl.ini"

# NCL ensemble parameters
# Number of base learners in total
ensemble_size = 2
# Number of evolutions
evoluatin_epoch = 3
learning_rate = 0.01
min_correlation_penalty_coefficient = 0.1
max_correlation_penalty_coefficient = 0.9
# Set to False for inferencing model only
is_experiment = True

# NCL base learner, i.e. the housing model
# To use a customised loss function, create a Python function and pass the function instance to train_models()
loss_function = nn.MSELoss()
relative_repository_driver_path = "data/driver/neat-ncl"
absolute_repository_driver_path = os.path.join(os.path.dirname(
    __file__), os.path.join("../", relative_repository_driver_path))
repository_driver_model_path = f"{absolute_repository_driver_path}/model"


if __name__ == "__main__":
    print(f"Running driver of NEAT NCL ensemble model ...")

    print(f"Number of CPU cores detected: {device_num}")

    config_path = os.path.join(os.path.dirname(
        __file__), os.path.join("../", config_filename))

    print(f"Using config file at {config_path}")

    config = neat.Config(NEATNCLGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    ensemble_model = NEATNCLEnsembleNet(
        config=config,
        model_repository_path=absolute_repository_driver_path
    )

    # Prepare training and testing data
    data_loader_train, data_loader_test = data_helper.get_data_loaders(
        filename=housing_filename,
        column_names=housing_column_names,
        output_column_name=housing_column_names[-1],
        test_size=test_size,
        random_state=random_state,
        batch_size=batch_size,
        normalise_data=True,
        remove_outliers=False
    )

    # NOTE: Remember to update this in experiment_neat_ncl.py so that we do not use data loader (use 1 batch of data instead) for both training and testing
    data_test = None
    for data_test_X, data_test_y in data_loader_test:
        # NOTE: Remember to change it in experiment_neat_ncl.py so that we have all data in data_test
        data_test = (data_test_X, data_test_y)
        break

    for data_train_X, data_train_y in data_loader_train:
        fitness_function_arguments = {
            "model_repository_path": repository_driver_model_path,
            # NOTE: Remember to change it in experiment_neat_ncl.py so that we have all data in data_train
            "data_train": (data_train_X, data_train_y),
            "loss_function": loss_function,
            "min_correlation_penalty_coefficient": min_correlation_penalty_coefficient,
            "max_correlation_penalty_coefficient": max_correlation_penalty_coefficient,
            "learning_rate": learning_rate,
            "is_experiment": is_experiment,
            "data_test": data_test
        }

        print(f"Evolving NEAT NCL ensemble model ...")
        ensemble_model.evolve(
            evolution_epoch=evoluatin_epoch,
            fitness_function_arguments=fitness_function_arguments
        )

        # TODO: We only train with 1 batch for now -> change by removing the use of data loader
        break

    print(f"Testing NEAT NCL ensemble model ...")
    # Test NEAT NCL manually
    # ensemble_losses = []

    # ensemble_model.eval()

    # with torch.no_grad():
    #     for batch_X, batch_y in data_loader_test:
    #         # NOTE: batch_y will be in CPU
    #         # model.eval() and torch.no_grad() will not be done NEATNCLEnsembleNet
    #         y_prediction = ensemble_model(batch_X)

    #         ensemble_losses.append(loss_function(y_prediction, batch_y))

    # average_ensemble_loss = torch.mean(torch.tensor(ensemble_losses))
    # # Assume y_prediction is a 1D tensor
    # print(
    #     f"Average NEAT NCL ensemble loss in testing: {average_ensemble_loss:.4f}")

    # -----------------------------------------------------------------------------

    # Test NEAT NCL by collecting experiment data
    experiment_data = ensemble_model.experiment_data
    print(f"Experiment data collected: {experiment_data}")

    print(f"Driver of NEAT NCL ensemble model finishes running ...")
