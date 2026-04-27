import neat
import os
from torch import nn
import torch.cuda
from src.model.neat_base_learner import NEATBaseLearner
from src.utils import data_helper


def log_genome(genome: neat.DefaultGenome) -> None:
    print(f"=================== Genome ===================")

    print(f"Node: ")
    for (node_id, node) in genome.nodes.items():
        print(
            f"ID: {node_id}; Bias: {node.bias}; Response: {node.response}; Activation: {node.activation}")

    print(f"=====================================")

    print(f"Connection: ")
    for (connection_id, connection) in genome.connections.items():
        print(
            f"ID: {connection_id}; Key: {connection.key}; Weight: {connection.weight}; Enabled: {connection.enabled}")

    print(f"=====================================")

# ==================================================================================


# Global configuration
housing_filename = "data/housing.csv"
housing_column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
                        "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]

# Hardware configuration
device_num = torch.cuda.device_count()  # Number of CPUs/GPUs

# For splitting dataset into training and testing data
test_size = 0.2  # Expect value-range is (0, 1)
random_state = 40
batch_size = 32

# NEAT parameters
evolution_epoch = 50
epoch_num = 100
learning_rate = 0.01
# TODO: Update this hyperparameter to find the best value
max_fitness = 50

# NEAT ensemble parameters
"""
These are dummy parameters for this program
They are only useful when you have an ensemble of NEAT models
"""
rank = 0
device = torch.device(f"cuda:{rank}")
device_index = 0

# To use a customised loss function, create a Python function and pass the function instance to train_models()
loss_function = nn.MSELoss()
# For training model
epoch_num_per_log = 20  # Number of epoches to log once
repository_driver_path = "data/driver/neat"
repository_driver_model_path = f"{repository_driver_path}/model"


if __name__ == "__main__":
    print(f"Running driver of NEAT model ...")

    print(f"Number of CUDA devices detected: {device_num}")

    config_path = os.path.join(os.path.dirname(
        __file__), os.path.join("../", "config-housing-neat.ini"))

    print(f"Using config file at {config_path}")

    neat_model = NEATBaseLearner.build_from_config_file(
        config_path=config_path,
        evolution_epoch=evolution_epoch,
        max_fitness=max_fitness,
        device=device,
        rank=rank,
        device_index=device_index,
        loss_function=loss_function,
        use_cpu=False,
        model_repository_path=repository_driver_model_path
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

    print(f"Training NEAT model ...")
    # TODO: Decide which one to use after experimenting with accuracy and running time
    # With more training before selection
    # neat_model.evolve(data_loader_train=data_loader_train, train_epoch=epoch_num, learning_rate=learning_rate)
    # Without training before selection
    # NOTE: batch_X and batch_y will be moved to GPU automatically in evolve(...)
    neat_model.evolve(data_loader_train=data_loader_train)

    # Log
    log_genome(neat_model.model_genome)
    print(f"=================== Check ===================")
    print(f"Number of non-input neurons: {len(neat_model.genome_nodes)}")
    print(f"Number of connections: {len(neat_model.genome_connections)}")

    print(f"Testing NEAT model ...")
    batch_losses = []
    for batch_X, batch_y in data_loader_test:
        # model.eval() and torch.no_grad() will be done NEATBaseLearner
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        y_prediction = neat_model(batch_X)

        batch_losses.append(loss_function(y_prediction, batch_y))

    average_batch_loss = torch.mean(torch.tensor(batch_losses))
    # Assume y_prediction is a 1D tensor
    print(f"Average NEAT loss in testing: {average_batch_loss:.4f}")

    print(f"Driver of NEAT model finishes running ...")
