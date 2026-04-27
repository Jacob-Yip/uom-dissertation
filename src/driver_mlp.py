import sys
from torch import nn
import torch.cuda
from src.model.mlp import MLP
from src.utils import data_helper
from src.utils.model_helper import MLPHelper

# ==================================================================================

# Global configuration
housing_filename = "data/housing.csv"
housing_column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
                        "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
housing_output_column_name = housing_column_names[-1]

# Hardware configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For splitting dataset into training and testing data
test_size = 0.2  # Expect value-range is (0, 1)
random_state = 40
batch_size = 32

# MLP model parameters
epoch_num = 100
learning_rate = 0.01

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
loss_function = nn.MSELoss()
# For training model
epoch_num_per_log = 20  # Number of epoches to log once
repository_driver_path = "data/driver/mlp"
repository_driver_model_path = f"{repository_driver_path}/model"


if __name__ == "__main__":
    print(f"Running driver of MLP ...")

    print(f"Using device {device} ...")

    if torch.cuda.is_available():
        assert device.type == "cuda", f"Invalid device (please use CUDA): {device}"
    else:
        # We do not run model
        print(f"Cannot find cuda to run model ...")
        print(f"Exiting programme ...")

        sys.exit(1)

    # Create model
    model = MLP.build_from_config(
        input_size=model_configurations["input_size"],
        hidden_sizes=model_configurations["hidden_sizes"],
        activations=model_configurations["activations"],
        activation_type=model_configurations["activation_type"],
        output_size=model_configurations["output_size"],
        dropout_rate=model_configurations["dropout_rate"],
        dropout_indices=model_configurations["dropout_indices"]
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)

    helper = MLPHelper()

    # Prepare training and testing data
    data_loader_train, data_loader_test = data_helper.get_data_loaders(
        filename=housing_filename,
        column_names=housing_column_names,
        output_column_name=housing_output_column_name,
        test_size=test_size,
        random_state=random_state,
        batch_size=batch_size,
        normalise_data=True,
        remove_outliers=False
    )

    print(f"Training MLP model ...")
    helper.train_model(
        model=model,
        epoch_num=epoch_num,
        loss_function=loss_function,
        optimizer=optimizer,
        data_loader_train=data_loader_train,
        epoch_num_per_log=epoch_num_per_log
    )

    print(f"Testing MLP model ...")
    total_loss, batch_num = helper.test_model(
        model=model,
        loss_function=loss_function,
        data_loader_test=data_loader_test
    )

    # Assume y_prediction is a 1D tensor
    print(f"Average loss in testing: {(total_loss / batch_num):.4f}")

    print(f"MLP driver finishes running ...")
