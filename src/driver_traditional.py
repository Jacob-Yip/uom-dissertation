from torch import nn
import torch.cuda
from src.model.traditional_ensemble_net import TraditionalEnsembleNet
from src.utils import data_helper

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

# Ensemble model parameters
# Number of base learners per GPU
# TODO: Expect every GPU has the same number of base learners now
ensemble_size = 2
epoch_num = 100
learning_rate = 0.01
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
loss_function = nn.MSELoss()
# For training model
epoch_num_per_log = 20  # Number of epoches to log once
repository_driver_path = "data/driver/traditional"
repository_driver_model_path = f"{repository_driver_path}/model"


if __name__ == "__main__":
    print(f"Running driver of traditional ensemble model ...")

    print(f"Number of CUDA devices detected: {device_num}")

    ensemble_model = TraditionalEnsembleNet(
        device_num=device_num,
        base_learner_nums=[ensemble_size] * device_num,
        model_configurations=model_configurations,
        dist_backend=dist_backend,
        master_address=master_address,
        master_port=master_port,
        gloo_socket_filename=gloo_socket_filename,
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

    print(f"Training traditional ensemble model ...")
    ensemble_model.train(
        data_loader_train=data_loader_train,
        epoch_num=epoch_num,
        learning_rate=learning_rate,
        loss_function=loss_function,
        epoch_num_per_log=epoch_num_per_log
    )

    print(f"Testing traditional ensemble model ...")
    ensemble_losses = []
    for batch_X, batch_y in data_loader_test:
        # model.eval() and torch.no_grad() will be done StaticNCLEnsembleNet

        # Method 1
        y_prediction = ensemble_model(batch_X)
        # Method 2
        # y_prediction = ensemble_model.evaluate(X=batch_X)

        ensemble_losses.append(loss_function(y_prediction, batch_y))

    average_ensemble_loss = torch.mean(torch.tensor(ensemble_losses))
    # Assume y_prediction is a 1D tensor
    print(
        f"Average traditional ensemble loss in testing: {average_ensemble_loss:.4f}")

    print(f"Driver of traditional ensemble model finishes running ...")
