from src.model.mlp import MLP
from src.utils import logger
from src.utils.experiment.experiment_runner import ExperimentRunner
import torch


class ExperimentRunnerMLP(ExperimentRunner):
    def __init__(self, configurations=None, **experiment_parameters):
        super().__init__(configurations, **experiment_parameters)

    def build_model(self, **experiment_parameters) -> None:
        # Instantiate MLP model
        try:
            self.model = MLP.build_from_config(
                input_size=self.configurations["input_size"],
                hidden_sizes=experiment_parameters["hidden_size"],
                activations=self.configurations["activations"],
                activation_type=self.configurations["activation_type"],
                output_size=self.configurations["output_size"],
                dropout_rate=self.configurations["dropout_rate"],
                dropout_indices=self.configurations["dropout_indices"]
            )

            self.model.to(device=self.configurations["device"])
        except Exception as e:
            raise Exception(f"Error in building MLP model: {str(e)}")

    def run(self, data_loader_train: torch.utils.data.DataLoader, data_loader_validation: torch.utils.data.DataLoader, data_loader_test: torch.utils.data.DataLoader) -> None:
        assert data_loader_validation is None, f"Expect data_loader_validation to be None (not applicable to this experiment): {data_loader_validation}"

        # Get experiment configurations
        try:
            epoch_num_per_log = self.configurations["epoch_num_per_log"]
            assert epoch_num_per_log is not None, f"Missing required configuration argument (expect not None): epoch_num_per_log"

            loss_function = self.configurations["loss_function"]
            assert loss_function is not None, f"Missing required configuration argument (expect not None): loss_function"
        except Exception as e:
            raise Exception(f"Error in getting configurations: {str(e)}")

        # Target experiment output data
        losses_test = []  # Shape: [experiment_num]
        # NOTE: We care about training loss because this is for MLP
        losses_train = []  # Shape: [experiment_num]

        for row_index, row in self.df_experiment.iterrows():
            # Get experiment parameters
            epoch_num = int(row["epoch_num"])
            learning_rate = float(row["learning_rate"])
            hidden_size = row["hidden_size"]

            self.build_model(hidden_size=hidden_size)

            assert self.model is not None, f"Missing required variable for experiment (expect not None): self.model -> invoke build_model()"

            # Log
            logger.log()
            logger.log(
                f"Experiment {row_index + 1} / {len(self.df_experiment)} parameters: ")
            logger.log(f"epoch_num: {epoch_num}")
            logger.log(f"learning_rate: {learning_rate}")
            logger.log(f"hidden_size: {hidden_size}")

            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate)

            # Train model

            self.model.train()

            for epoch in range(epoch_num):
                batch_loss_sum_train = 0

                for (batch_X_train, batch_y_train) in data_loader_train:
                    batch_X_train = batch_X_train.to(
                        self.configurations["device"])
                    batch_y_train = batch_y_train.to(
                        self.configurations["device"])

                    y_prediction_train = self.model(batch_X_train)

                    loss_train = loss_function(
                        y_prediction_train, batch_y_train)

                    # Backward pass and optimisation
                    optimizer.zero_grad()  # Reset gradients
                    loss_train.backward()  # Compute gradients
                    optimizer.step()  # Update weights

                    batch_loss_sum_train += loss_train.item()

                    # NOTE: Remove the line below for logging if you want
                    # if epoch % epoch_num_per_log == 0:
                    #     logger.log(f"Epoch: {epoch + 1} / {epoch_num}; Loss: {(batch_loss_sum_train / len(data_loader_train)):.4f}")

                # Get training loss
                if epoch == epoch_num - 1:
                    # Last epoch so we care about the current training loss
                    losses_train.append(
                        batch_loss_sum_train / len(data_loader_train))

            # Test model
            self.model.eval()

            with torch.no_grad():
                batch_loss_sum_test = 0

                for (batch_X_test, batch_y_test) in data_loader_test:
                    batch_X_test = batch_X_test.to(
                        self.configurations["device"])
                    batch_y_test = batch_y_test.to(
                        self.configurations["device"])

                    y_prediction_test = self.model(batch_X_test)

                    loss_test = loss_function(
                        y_prediction_test, batch_y_test).detach()

                    batch_loss_sum_test += loss_test.item()

            # The experiment output data we are interested in
            losses_test.append(batch_loss_sum_test / len(data_loader_test))

        self.df_experiment["loss_test"] = losses_test
        self.df_experiment["loss_train"] = losses_train
