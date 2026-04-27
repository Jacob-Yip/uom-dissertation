import ast
import neat
import torch
# Model
from src.model.mlp import MLP
from src.model.neat_base_learner import NEATBaseLearner
from src.model.neat_ncl_ensemble_net import NEATNCLEnsembleNet
from src.model.static_ncl_ensemble_net import StaticNCLEnsembleNet
from src.model.traditional_ensemble_net import TraditionalEnsembleNet
# Experiment
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import seaborn as sns
# Utils
from src.utils import data_helper
from src.utils import maths_helper
from src.utils.model_helper import MLPHelper


# Global configurations of experiment graphs
FONT_SIZE = 16
FONT_FAMILY = "serif"
FONT_SERIF = ["Times New Roman"]
AXES_TITLESIZE = 16
AXES_TITLEWEIGHT = "normal"
AXES_LABELSIZE = 16
LEGEND_FONTSIZE = 12


class ExperimentMLPHelper:
    def __init__(self, test_size: float, random_state: int, batch_size: int, experiment_model_repository_path: str, epoch_nums=[100], learning_rates=[0.01], validation_size=None):
        # Experiment hyperparameter
        self.__epoch_nums = epoch_nums
        self.__learning_rates = learning_rates

        # Configuration
        self.__test_size = test_size
        self.__validation_size = validation_size
        self.__random_state = random_state
        self.__batch_size = batch_size
        self.__experiment_model_repository_path = experiment_model_repository_path

        self.__device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # List of hyperparameters to plot against loss
        self.__hyperparameters = ["epoch_num", "learning_rate"]

        self.__model_helper = MLPHelper(self.__device)

    def get_data_loaders(self, filename: str, column_names: list, output_column_name: str, normalise_data=True, remove_outliers=False) -> tuple:
        """
        Return data loaders of training and testing data and possibly validation data
        Return (training_data_loader, testing_data_loader) or (training_data_loader, validation_data_loader, testing_data_loader)

        :return: (training_data_loader, testing_data_loader) or (training_data_loader, validation_data_loader, testing_data_loader)
        """
        return data_helper.get_data_loaders(
            filename=filename,
            column_names=column_names,
            output_column_name=output_column_name,
            test_size=self.__test_size,
            random_state=self.__random_state,
            batch_size=self.__batch_size,
            validation_size=self.__validation_size,
            normalise_data=normalise_data,
            remove_outliers=remove_outliers
        )

    def build_mlp(self, model_configurations: dict) -> MLP:
        # Instantiate MLP model
        return MLP.build_from_config(
            input_size=model_configurations["input_size"],
            hidden_sizes=model_configurations["hidden_sizes"],
            activations=model_configurations["activations"],
            activation_type=model_configurations["activation_type"],
            output_size=model_configurations["output_size"],
            dropout_rate=model_configurations["dropout_rate"],
            dropout_indices=model_configurations["dropout_indices"]
        )

    def experiment(self, model_configurations: dict, loss_function, data_loader_train, data_loader_test, epoch_num_per_log=20):
        """
        Run the experiment
        To use a customised loss function, create a Python function and pass the function instance to train_model()

        :param: loss_function: The loss function of models
        """
        # Hyperparameter combinations
        param_combinations = list(product(
            self.__epoch_nums,
            self.__learning_rates
        ))

        self.__df_experiment = pd.DataFrame(
            param_combinations, columns=self.__hyperparameters)

        # For storing experiment output data so that we can plot graphs
        # Collect average batch loss
        average_batch_losses_train = []  # shape: [epoch, 1]
        average_batch_losses_test = []  # shape: [epoch, 1]

        for row_index, row in self.__df_experiment.iterrows():
            epoch_num = int(row["epoch_num"])
            learning_rate = float(row["learning_rate"])

            # Log
            print(f"============================================", flush=True)
            print(
                f"Experiment {row_index + 1} / {len(self.__df_experiment)} parameters: ", flush=True)
            print(f"epoch_num: {epoch_num}", flush=True)
            print(f"learning_rate: {learning_rate}", flush=True)

            # Instantiate mlp model
            mlp_model = self.build_mlp(
                model_configurations=model_configurations
            )
            mlp_model.to(self.__device)  # Move to GPU

            # Instantiate optimizer
            optimizer = torch.optim.Adam(
                mlp_model.parameters(), lr=learning_rate)

            # Training
            print(
                f"Training MLP model in experiment with epoch_num={epoch_num}, learning_rate={learning_rate} ...", flush=True)
            total_loss_train, batch_num_train = self.__model_helper.train_model(
                model=mlp_model,
                epoch_num=epoch_num,
                loss_function=loss_function,
                optimizer=optimizer,
                data_loader_train=data_loader_train,
                epoch_num_per_log=epoch_num_per_log
            )
            average_batch_loss_train = total_loss_train / batch_num_train
            average_batch_losses_train.append(average_batch_loss_train)

            # ============================================================================================

            print(
                f"Testing MLP model in experiment with epoch_num={epoch_num}, learning_rate={learning_rate} ...", flush=True)
            # Testing
            total_loss_test, batch_num_test = self.__model_helper.test_model(
                model=mlp_model,
                loss_function=loss_function,
                data_loader_test=data_loader_test
            )
            average_batch_loss_test = total_loss_test / batch_num_test
            average_batch_losses_test.append(average_batch_loss_test)

        # Add columns of average_loss to the dataframe
        self.__df_experiment["average_loss_train"] = average_batch_losses_train
        self.__df_experiment["average_loss_test"] = average_batch_losses_test

    def save_csv(self, repository_experiment_csv_path: str, csv_name: str) -> None:
        assert repository_experiment_csv_path and csv_name, f"Invalid final csv path: {repository_experiment_csv_path}/{csv_name}.csv"
        self.__df_experiment.to_csv(
            f"{repository_experiment_csv_path}/{csv_name}.csv", index=False)

    def plot_graph(self, repository_experiment_img_path: str, img_name: str, save_img=True, image_resolution=300, show_img=False) -> None:
        # Set plot style for nicer plots
        sns.set_theme(style="whitegrid", rc={
            "font.size": FONT_SIZE,  # Default font size
            "font.family": FONT_FAMILY,
            "font.serif": FONT_SERIF,
            "axes.titlesize": AXES_TITLESIZE,
            "axes.titleweight": AXES_TITLEWEIGHT,
            "axes.labelsize": AXES_LABELSIZE,
            "legend.fontsize": LEGEND_FONTSIZE
        })

        plt.figure(figsize=(16, 12))  # Larger figure for 2 plots

        row_num = 1
        column_num = 2

        for i, hyperparameter in enumerate(self.__hyperparameters, 1):
            # 2 graphs are needed for 2 hyperparameters
            plt.subplot(row_num, column_num, i)
            # The 2 lines will be assigned with different colours by sns automatically
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_train",
                marker='o',
                label=f"Train"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_test",
                marker='o',
                label=f"Test"
            )
            plt.title(f"MSE vs {hyperparameter}")
            plt.xlabel(hyperparameter)
            plt.ylabel("MSE")

        plt.tight_layout()

        if save_img:
            assert repository_experiment_img_path and img_name, f"Invalid final image path: {repository_experiment_img_path}/{img_name}.png"

            # Save graph as image if applicable
            plt.savefig(f"{repository_experiment_img_path}/{img_name}.png",
                        dpi=image_resolution)

        if show_img:
            # Show graph if applicable
            # Might block the execution so we set show_img = False for non-interactive Python script
            # Jupyter notebook will still show the image if show_img = False though
            plt.show()

    @property
    def device(self):
        return self.__device

    @property
    def df_experiment(self):
        return self.__df_experiment

# ============================================================================================================


class ExperimentTraditionalHelper:
    def __init__(self, test_size: float, random_state: int, batch_size: int, experiment_model_repository_path: str, master_address: str, master_port: str, device_num: int, dist_backend: str, ensemble_sizes=[1, 2], epoch_nums=[100], learning_rates=[0.01], validation_size=None):
        # Experiment hyperparameter
        self.__ensemble_sizes = ensemble_sizes
        self.__epoch_nums = epoch_nums
        self.__learning_rates = learning_rates

        # Configuration
        self.__test_size = test_size
        self.__validation_size = validation_size
        self.__random_state = random_state
        self.__batch_size = batch_size
        self.__experiment_model_repository_path = experiment_model_repository_path
        self.__master_address = master_address
        self.__master_port = master_port
        self.__device_num = device_num
        self.__dist_backend = dist_backend

        self.__device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # List of hyperparameters to plot against loss
        self.__hyperparameters = [
            "ensemble_size", "epoch_num", "learning_rate"]

        # For MLP model in this experiment
        self.__mlp_model_helper = MLPHelper(self.__device)

    def get_data_loaders(self, filename: str, column_names: list, output_column_name: str, normalise_data=True, remove_outliers=False) -> tuple:
        """
        Return data loaders of training and testing data
        Return (training_data_loader, testing_data_loader)

        :return: (training_data_loader, testing_data_loader)
        """
        return data_helper.get_data_loaders(
            filename=filename,
            column_names=column_names,
            output_column_name=output_column_name,
            test_size=self.__test_size,
            random_state=self.__random_state,
            batch_size=self.__batch_size,
            validation_size=self.__validation_size,
            normalise_data=normalise_data,
            remove_outliers=remove_outliers
        )

    def build_ensemble_net(self, model_configurations: dict, ensemble_size: int) -> TraditionalEnsembleNet:
        # Instantiate ensemble net
        return TraditionalEnsembleNet(
            device_num=self.__device_num,
            base_learner_nums=[ensemble_size] * self.__device_num,
            model_configurations=model_configurations,
            dist_backend=self.__dist_backend,
            master_address=self.__master_address,
            master_port=self.__master_port,
            model_repository_path=self.__experiment_model_repository_path
        )

    def build_mlp(self, model_configurations: dict) -> MLP:
        # Instantiate MLP model
        return MLP.build_from_config(
            input_size=model_configurations["input_size"],
            hidden_sizes=model_configurations["hidden_sizes"],
            activations=model_configurations["activations"],
            activation_type=model_configurations["activation_type"],
            output_size=model_configurations["output_size"],
            dropout_rate=model_configurations["dropout_rate"],
            dropout_indices=model_configurations["dropout_indices"]
        )

    def test_ensemble_model(self, ensemble_model: TraditionalEnsembleNet, ensemble_loss_function, individual_loss_function, data_loader) -> tuple:
        """
        Test a traditional ensemble model
        This method might be invoked with training data
        Do not be mislead by the method name -> this function just evaluates the performance of a model against a dataset, which can be either training or testing
        Return (total_loss, batch_num, y_predictions_all, average_individual_losses)
        To get the average ensemble loss per batch, calculate (total_loss / batch_num)
        To get the average individual loss among all base learners, calculate (average_individual_losses / base_learner_num)

        :param: ensemble_model: The ensemble model instance
        :param: ensemble_loss_function: The loss function of the ensemble model (will return a scalar instead of a tensor)
        :param: individual_loss_function: The loss function of a base learner (will return a tensor instead of a scalar)
        :param: data_loader: The training/testing data loader
        :return: (total_loss, batch_num, y_predictions_all, average_individual_losses)
        """
        # model.eval() and torch.no_grad() will be done TraditionalEnsembleNet

        total_loss = 0
        y_predictions_all_raw = []
        individual_losses_raw = []

        for (batch_X, batch_y) in data_loader:
            # Shape: [base_learner_num, batch_num, 1]
            y_predictions_all_batch = ensemble_model.evaluate_all(batch_X)

            y_predictions_all_raw.append(y_predictions_all_batch)

            y_prediction = ensemble_model.vote(y_predictions_all_batch)

            ensemble_loss = ensemble_loss_function(y_prediction, batch_y)

            total_loss += ensemble_loss.item()

            # ========================================
            # Calculate individual loss
            # Shape: [base_learner_num, batch_num, 1]
            individual_losses_batch_all_data = individual_loss_function(y_predictions_all_batch, batch_y.unsqueeze(
                0).expand_as(y_predictions_all_batch))
            # Shape [base_learner_num]
            individual_losses_batch = individual_losses_batch_all_data.mean(
                dim=tuple(range(1, individual_losses_batch_all_data.dim())))

            individual_losses_raw.append(individual_losses_batch)

        # Shape: [base_learner_num, x_num, 1]
        y_predictions_all = torch.cat(y_predictions_all_raw, dim=1)

        # Shape: [base_learner_num]
        average_individual_losses = torch.stack(
            individual_losses_raw, dim=1).mean(dim=1)

        return (total_loss, len(data_loader), y_predictions_all, average_individual_losses)

    def get_predictions_std(self, y_predictions_all: torch.Tensor) -> float:
        """
        NOTE: Deprecated
        We will calcuate the standard deviation of predictions of each batch
        Then we will add the mean of standard deviations of all batches to the lists below

        :param: y_predictions_all: Expect a tensor with shape [base_learner_num, x_num, y.shape] (y.shape should be 1 in the simplest experiment)
        :return: Mean of standard deviations of each predictions per batch
        """
        y_predictions_all_std = torch.std(
            y_predictions_all, dim=0)  # Expect shape: [x_num, y.shape]
        return torch.mean(y_predictions_all_std, dim=0, dtype=float).item()

    def experiment(self, model_configurations: dict, ensemble_loss_function, individual_loss_function, data_loader_train, data_loader_test, epoch_num_per_log=20):
        """
        Run the experiment
        To use a customised loss function, create a Python function and pass the function instance to train_model()

        :param: ensemble_loss_function: The loss function of the ensemble model
        """
        # Hyperparameter combinations
        param_combinations = list(product(
            self.__ensemble_sizes,
            self.__epoch_nums,
            self.__learning_rates
        ))

        self.__df_experiment = pd.DataFrame(
            param_combinations, columns=self.__hyperparameters)

        # For storing experiment output data so that we can plot graphs
        average_batch_losses_train_traditional = []  # shape: [epoch, 1]
        average_batch_losses_test_traditional = []  # shape: [epoch, 1]
        # For comparison - expect MLP is worse than a traditional ensemble model
        average_batch_losses_train_mlp = []  # shape: [epoch, 1]
        average_batch_losses_test_mlp = []  # shape: [epoch, 1]
        """
        We will calcuate the average individual loss of all base learners
        Expected to be higher than the ensemble loss of the ensemble model
        """
        average_individual_loss_train_traditional = []  # shape: [epoch, 1]
        average_individual_loss_test_traditional = []  # shape: [epoch, 1]
        diversity_coefficients_train_traditional = []  # Shape: [experiment_num, 1]
        diversity_coefficients_test_traditional = []  # Shape: [experiment_num, 1]

        for row_index, row in self.__df_experiment.iterrows():
            ensemble_size = int(row["ensemble_size"])
            epoch_num = int(row["epoch_num"])
            learning_rate = float(row["learning_rate"])

            # Log
            print(f"============================================", flush=True)
            print(
                f"Experiment {row_index + 1} / {len(self.__df_experiment)} parameters: ", flush=True)
            print(f"ensemble_size: {ensemble_size}", flush=True)
            print(f"epoch_num: {epoch_num}", flush=True)
            print(f"learning_rate: {learning_rate}", flush=True)

            # Instantiate ensemble model
            ensemble_model = self.build_ensemble_net(
                model_configurations=model_configurations,
                ensemble_size=ensemble_size
            )

            # Instantiate MLP model
            mlp_model = self.build_mlp(
                model_configurations=model_configurations
            )
            mlp_model.to(self.__device)  # Move to GPU

            # Instantiate optimizer
            optimizer = torch.optim.Adam(
                mlp_model.parameters(), lr=learning_rate)

            # ======================================================
            # Ensemble model

            # Train ensemble model
            print(f"Training traditional ensemble model in experiment ...", flush=True)
            ensemble_model.train(
                data_loader_train=data_loader_train,
                epoch_num=epoch_num,
                learning_rate=learning_rate,
                loss_function=ensemble_loss_function,
                epoch_num_per_log=epoch_num_per_log
            )
            total_loss_train_traditional, batch_num_train_traditional, y_predictions_all_train_traditional, individual_losses_train_traditional = self.test_ensemble_model(
                ensemble_model=ensemble_model,
                ensemble_loss_function=ensemble_loss_function,
                individual_loss_function=individual_loss_function,
                data_loader=data_loader_train
            )

            average_batch_losses_train_traditional.append(
                total_loss_train_traditional / batch_num_train_traditional)

            average_individual_loss_train_traditional.append(
                individual_losses_train_traditional.mean(dim=0).item())

            diversity_coefficients_train_traditional.append(torch.mean(maths_helper.diversity_coefficient(
                y_predictions=y_predictions_all_train_traditional), dim=0).item())

            # Test ensemble model
            print(f"Testing traditional ensemble model in experiment ...", flush=True)
            total_loss_test_traditional, batch_num_test_traditional, y_predictions_all_test_traditional, individual_losses_test_traditional = self.test_ensemble_model(
                ensemble_model=ensemble_model,
                ensemble_loss_function=ensemble_loss_function,
                individual_loss_function=individual_loss_function,
                data_loader=data_loader_test
            )

            average_batch_losses_test_traditional.append(
                total_loss_test_traditional / batch_num_test_traditional)

            average_individual_loss_test_traditional.append(
                individual_losses_test_traditional.mean(dim=0).item())

            diversity_coefficients_test_traditional.append(torch.mean(maths_helper.diversity_coefficient(
                y_predictions=y_predictions_all_test_traditional), dim=0).item())

            # ======================================================
            # MLP model

            # Train + test MLP model
            # There is a more efficient way, which is to pick the largest epoch and evaluate model for each target training epoch (reduce number of times to train models as we are training the model once assuming other hyperparameters stay the same except epoch_num)
            # However, that will ruin the consistency of this code so I did not use it
            # It is a trade-off between code consistency and execution time
            print(f"Training and testing MLP model in experiment ...", flush=True)
            for epoch in range(epoch_num):
                total_loss_train_mlp = 0

                # Training
                mlp_model.train()
                for (batch_X, batch_y) in data_loader_train:
                    batch_X = batch_X.to(self.__device)
                    batch_y = batch_y.to(self.__device)

                    # y_prediction is a PyTorch tensor
                    y_prediction = mlp_model(batch_X)

                    loss = ensemble_loss_function(y_prediction, batch_y)

                    # Backward pass and optimization
                    optimizer.zero_grad()  # reset gradients
                    loss.backward()        # compute gradients
                    optimizer.step()       # update weights

                    total_loss_train_mlp += loss.item()

                average_batch_loss_train_mlp = total_loss_train_mlp / \
                    len(data_loader_train)
                if epoch % epoch_num_per_log == 0:
                    print(
                        f"Epoch: {epoch + 1} / {epoch_num}; Loss: {average_batch_loss_train_mlp:.4f}", flush=True)

                if epoch == epoch_num - 1:
                    # Last epoch
                    average_batch_losses_train_mlp.append(
                        average_batch_loss_train_mlp)

            # ============================================================================================

            # Testing
            total_loss_test_mlp, batch_num_test_mlp = self.__mlp_model_helper.test_model(
                model=mlp_model,
                loss_function=ensemble_loss_function,
                data_loader_test=data_loader_test
            )
            average_batch_loss_test_mlp = total_loss_test_mlp / batch_num_test_mlp
            average_batch_losses_test_mlp.append(average_batch_loss_test_mlp)

        # Add columns of average_loss to the dataframe
        self.__df_experiment["average_loss_train_traditional"] = average_batch_losses_train_traditional
        self.__df_experiment["average_loss_test_traditional"] = average_batch_losses_test_traditional
        self.__df_experiment["average_loss_train_mlp"] = average_batch_losses_train_mlp
        self.__df_experiment["average_loss_test_mlp"] = average_batch_losses_test_mlp
        self.__df_experiment["average_individual_loss_train_traditional"] = average_individual_loss_train_traditional
        self.__df_experiment["average_individual_loss_test_traditional"] = average_individual_loss_test_traditional
        self.__df_experiment["diversity_coefficient_train_traditional"] = diversity_coefficients_train_traditional
        self.__df_experiment["diversity_coefficient_test_traditional"] = diversity_coefficients_test_traditional

    def save_csv(self, repository_experiment_csv_path: str, csv_name: str) -> None:
        assert repository_experiment_csv_path and csv_name, f"Invalid final csv path: {repository_experiment_csv_path}/{csv_name}.csv"
        self.__df_experiment.to_csv(
            f"{repository_experiment_csv_path}/{csv_name}.csv", index=False)

    def plot_graph(self, repository_experiment_img_path: str, img_name: str, save_img=True, image_resolution=300, show_img=False) -> None:
        # Set plot style for nicer plots
        sns.set_theme(style="whitegrid", rc={
            "font.size": FONT_SIZE,  # Default font size
            "font.family": FONT_FAMILY,
            "font.serif": FONT_SERIF,
            "axes.titlesize": AXES_TITLESIZE,
            "axes.titleweight": AXES_TITLEWEIGHT,
            "axes.labelsize": AXES_LABELSIZE,
            "legend.fontsize": LEGEND_FONTSIZE
        })

        plt.figure(figsize=(16, 24))  # Larger figure for 6 plots

        row_num = len(self.__hyperparameters)
        column_num = 2

        for i, hyperparameter in enumerate(self.__hyperparameters, 1):
            if hyperparameter == "ensemble_size":
                hyperparameter_text = "ensemble size"
            elif hyperparameter == "epoch_num":
                hyperparameter_text = "train epoch"
            elif hyperparameter == "learning_rate":
                hyperparameter_text = "learning rate"
            else:
                hyperparameter_text = hyperparameter

            column_index = 0  # 0-indexed

            # Average loss
            plt.subplot(row_num, column_num, (i - 1) *
                        column_num + (column_index + 1))
            # sns.lineplot(
            #     data=self.__df_experiment,
            #     x=hyperparameter,
            #     y="average_loss_train_traditional",
            #     marker='o',
            #     label=f"Train - traditional ensemble model"
            # )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_test_traditional",
                marker='o',
                label=f"Test - traditional ensemble model"
            )
            # sns.lineplot(
            #     data=self.__df_experiment,
            #     x=hyperparameter,
            #     y="average_loss_train_mlp",
            #     marker='o',
            #     label=f"Train - MLP model"
            # )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_test_mlp",
                marker='o',
                label=f"Test - MLP model"
            )
            # sns.lineplot(
            #     data=self.__df_experiment,
            #     x=hyperparameter,
            #     y="average_individual_loss_train_traditional",
            #     marker='o',
            #     label=f"Train - base learner (average)"
            # )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_individual_loss_test_traditional",
                marker='o',
                label=f"Test - base learner (average)"
            )
            plt.title(f"MSE vs {hyperparameter_text}")
            plt.xlabel(hyperparameter_text)
            plt.ylabel("MSE")
            plt.legend()

            # ===================================================================

            # Diversity Coefficient
            column_index = 1  # 0-indexed

            plt.subplot(row_num, column_num, (i - 1) *
                        column_num + (column_index + 1))
            # sns.lineplot(
            #     data=self.__df_experiment,
            #     x=hyperparameter,
            #     y="diversity_coefficient_train_traditional",
            #     marker='o',
            #     label=f"Train - traditional ensemble model"
            # )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="diversity_coefficient_test_traditional",
                marker='o',
                label=f"Test - traditional ensemble model"
            )
            plt.title(
                f"$\\rho$ vs {hyperparameter_text}")
            plt.xlabel(hyperparameter_text)
            plt.ylabel("$\\rho$")
            plt.legend()

        plt.tight_layout()

        if save_img:
            assert repository_experiment_img_path and img_name, f"Invalid final image path: {repository_experiment_img_path}/{img_name}.png"

            # Save graph as image if applicable
            plt.savefig(f"{repository_experiment_img_path}/{img_name}.png",
                        dpi=image_resolution)

        if show_img:
            # Show graph if applicable
            # Might block the execution so we set show_img = False for non-interactive Python script
            # Jupyter notebook will still show the image if show_img = False though
            plt.show()

    @property
    def device(self):
        return self.__device

    @property
    def df_experiment(self):
        return self.__df_experiment


# ============================================================================================================


class ExperimentStaticNCLHelper:
    def __init__(self, test_size: float, random_state: int, batch_size: int, experiment_model_repository_path: str, master_address: str, master_port: str, device_num: int, dist_backend: str, correlation_penalty_coefficients=[0.5], ensemble_sizes=[1, 2], epoch_nums=[100], learning_rates=[0.01], hidden_sizes=[[6]], validation_size=None):
        # Experiment hyperparameter
        self.__correlation_penalty_coefficients = correlation_penalty_coefficients
        self.__ensemble_sizes = ensemble_sizes
        self.__epoch_nums = epoch_nums
        self.__learning_rates = learning_rates
        self.__hidden_sizes = hidden_sizes

        # Configuration
        self.__test_size = test_size
        self.__validation_size = validation_size
        self.__random_state = random_state
        self.__batch_size = batch_size
        self.__experiment_model_repository_path = experiment_model_repository_path
        self.__master_address = master_address
        self.__master_port = master_port
        self.__device_num = device_num
        self.__dist_backend = dist_backend

        self.__device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # List of hyperparameters to plot against loss
        self.__hyperparameters = [
            "correlation_penalty_coefficient", "ensemble_size", "epoch_num", "learning_rate", "hidden_size"]

        # For MLP model in this experiment
        self.__mlp_model_helper = MLPHelper(self.__device)

    def get_data_loaders(self, filename: str, column_names: list, output_column_name: str, normalise_data=True, remove_outliers=False) -> tuple:
        """
        Return data loaders of training and testing data and possibly validation data
        Return (training_data_loader, testing_data_loader) or (training_data_loader, validation_data_loader, testing_data_loader)

        :return: (training_data_loader, testing_data_loader) or (training_data_loader, validation_data_loader, testing_data_loader)
        """
        return data_helper.get_data_loaders(
            filename=filename,
            column_names=column_names,
            output_column_name=output_column_name,
            test_size=self.__test_size,
            random_state=self.__random_state,
            batch_size=self.__batch_size,
            validation_size=self.__validation_size,
            normalise_data=normalise_data,
            remove_outliers=remove_outliers
        )

    def build_static_ncl_ensemble_net(self, model_configurations: dict, ensemble_size: int) -> StaticNCLEnsembleNet:
        # Instantiate static NCL ensemble net
        return StaticNCLEnsembleNet(
            device_num=self.__device_num,
            base_learner_nums=[ensemble_size] * self.__device_num,
            model_configurations=model_configurations,
            dist_backend=self.__dist_backend,
            master_address=self.__master_address,
            master_port=self.__master_port,
            model_repository_path=self.__experiment_model_repository_path
        )

    def build_traditional_ensemble_net(self, model_configurations: dict, ensemble_size: int) -> TraditionalEnsembleNet:
        # Instantiate traditional ensemble net
        return TraditionalEnsembleNet(
            device_num=self.__device_num,
            base_learner_nums=[ensemble_size] * self.__device_num,
            model_configurations=model_configurations,
            dist_backend=self.__dist_backend,
            master_address=self.__master_address,
            master_port=self.__master_port,
            model_repository_path=self.__experiment_model_repository_path
        )

    def build_mlp(self, model_configurations: dict) -> MLP:
        # Instantiate MLP model
        return MLP.build_from_config(
            input_size=model_configurations["input_size"],
            hidden_sizes=model_configurations["hidden_sizes"],
            activations=model_configurations["activations"],
            activation_type=model_configurations["activation_type"],
            output_size=model_configurations["output_size"],
            dropout_rate=model_configurations["dropout_rate"],
            dropout_indices=model_configurations["dropout_indices"]
        )

    def test_ensemble_model_static_ncl(self, ensemble_model: StaticNCLEnsembleNet, loss_function, data_loader, lambda_ncl: float) -> tuple:
        """
        Test a static NCL ensemble model
        This method might be invoked with training data
        Do not be mislead by the method name -> this function just evaluates the performance of a model against a dataset, which can be either training or testing
        Return (total_loss, batch_num, y_predictions_all)
        To get the average loss per batch, calculate (total_loss / batch_num)

        :param: ensemble_model: The ensemble model instance
        :param: loss_function: The loss function
        :param: data_loader: The training/testing data loader
        :param: lambda_ncl: Correlation penalty coefficient
        :return: (total_loss, batch_num, y_predictions_all)
        """
        # model.eval() and torch.no_grad() will be done StaticNCLEnsembleNet

        total_loss = 0
        y_predictions_all_raw = []

        for (batch_X, batch_y) in data_loader:
            # Shape: [base_learner_num, batch_num, 1]
            y_predictions_all_batch = ensemble_model.evaluate_all_predictions(
                batch_X, lambda_ncl=lambda_ncl)

            y_predictions_all_raw.append(y_predictions_all_batch)

            y_prediction = ensemble_model.vote(y_predictions_all_batch)

            loss = loss_function(y_prediction, batch_y)

            total_loss += loss.item()

        # Shape: [base_learner_num, x_num, 1]
        y_predictions_all = torch.cat(y_predictions_all_raw, dim=1)

        return (total_loss, len(data_loader), y_predictions_all)

    def test_ensemble_model_traditional(self, ensemble_model: TraditionalEnsembleNet, loss_function, data_loader) -> tuple:
        """
        Test a traditional ensemble model
        This method might be invoked with training data
        Do not be mislead by the method name -> this function just evaluates the performance of a model against a dataset, which can be either training or testing
        Return (total_loss, batch_num, y_predictions_all)
        To get the average loss per batch, calculate (total_loss / batch_num)

        :param: ensemble_model: The ensemble model instance
        :param: loss_function: The loss function
        :param: data_loader: The training/testing data loader
        :return: (total_loss, batch_num, y_predictions_all)
        """
        # model.eval() and torch.no_grad() will be done TraditionalEnsembleNet

        total_loss = 0
        y_predictions_all_raw = []

        for (batch_X, batch_y) in data_loader:
            # Shape: [base_learner_num, batch_num, 1]
            y_predictions_all_batch = ensemble_model.evaluate_all(batch_X)

            y_predictions_all_raw.append(y_predictions_all_batch)

            y_prediction = ensemble_model.vote(y_predictions_all_batch)

            loss = loss_function(y_prediction, batch_y)

            total_loss += loss.item()

        # Shape: [base_learner_num, x_num, 1]
        y_predictions_all = torch.cat(y_predictions_all_raw, dim=1)

        return (total_loss, len(data_loader), y_predictions_all)

    def get_predictions_std(self, y_predictions_all: torch.Tensor) -> float:
        """
        NOTE: Deprecated
        We will calcuate the standard deviation of predictions of each batch
        Then we will add the mean of standard deviations of all batches to the lists below

        :param: y_predictions_all: Expect a tensor with shape [base_learner_num, x_num, y.shape] (y.shape should be 1 in the simplest experiment)
        :return: Mean of standard deviations of each predictions per batch
        """
        y_predictions_all_std = torch.std(
            y_predictions_all, dim=0)  # Expect shape: [x_num, y.shape]
        return torch.mean(y_predictions_all_std, dim=0, dtype=float).item()

    def experiment(self, model_configurations: dict, loss_function: any, data_loader_train: torch.utils.data.DataLoader, data_loader_validation: torch.utils.data.DataLoader, data_loader_test: torch.utils.data.DataLoader, epoch_num_per_log=20, validation_cycle=50, validation_epoch_window=500):
        """
        Run the experiment
        To use a customised loss function, create a Python function and pass the function instance to train_model()

        :param: loss_function: The loss function of models
        """
        # Hyperparameter combinations
        param_combinations = list(product(
            self.__correlation_penalty_coefficients,
            self.__ensemble_sizes,
            self.__epoch_nums,
            self.__learning_rates,
            self.__hidden_sizes
        ))

        self.__df_experiment = pd.DataFrame(
            param_combinations, columns=self.__hyperparameters)

        # For storing experiment output data so that we can plot graphs
        average_batch_losses_train_static_ncl = []  # Shape: [experiment_num, 1]
        average_batch_losses_test_static_ncl = []  # Shape: [experiment_num, 1]
        """
        For comparison
        - Expect when lambda_ncl = 0, static-ncl-ensemble-model is the same as traditional-ensemble-model
        - (I think) Nothing special when lambda_ncl = 0.25, 0.5, 0.75
        - Expect when lambda_ncl = 1, static-ncl-ensemble-model is like a signle estimator
        """
        average_batch_losses_train_traditional = []  # Shape: [experiment_num, 1]
        average_batch_losses_test_traditional = []  # Shape: [experiment_num, 1]
        """
        For comparison
        """
        average_batch_losses_train_mlp = []  # Shape: [experiment_num, 1]
        average_batch_losses_test_mlp = []  # Shape: [experiment_num, 1]
        """
        For plotting a graph of loss against diversity coefficient
        """
        diversity_coefficients_train_static_ncl = []  # Shape: [experiment_num, 1]
        diversity_coefficients_test_static_ncl = []  # Shape: [experiment_num, 1]
        diversity_coefficients_train_traditional = []  # Shape: [experiment_num, 1]
        diversity_coefficients_test_traditional = []  # Shape: [experiment_num, 1]
        """
        For plotting graph of MSE against correlation_penalty_coefficient
        """
        hidden_sizes = []  # Shape: [experiment_num, 1]
        ensemble_sizes = []  # Shape: [experiment_num, 1]

        for row_index, row in self.__df_experiment.iterrows():

            correlation_penalty_coefficient = float(
                row["correlation_penalty_coefficient"])
            ensemble_size = int(row["ensemble_size"])
            epoch_num = int(row["epoch_num"])
            learning_rate = float(row["learning_rate"])
            hidden_size = row["hidden_size"]

            # Log
            print(f"============================================", flush=True)
            print(
                f"Experiment {row_index + 1} / {len(self.__df_experiment)} parameters: ", flush=True)
            print(
                f"correlation_penalty_coefficient: {correlation_penalty_coefficient}", flush=True)
            print(f"ensemble_size: {ensemble_size}", flush=True)
            print(f"epoch_num: {epoch_num}", flush=True)
            print(f"learning_rate: {learning_rate}", flush=True)
            print(f"hidden_size: {hidden_size}", flush=True)

            hidden_sizes.append(hidden_size)
            ensemble_sizes.append(ensemble_size)

            # Update model_configurations as hidden_sizes is actually missing on purpose
            model_configurations["hidden_sizes"] = hidden_size

            # Instantiate static NCL ensemble model
            ensemble_model_static_ncl = self.build_static_ncl_ensemble_net(
                model_configurations=model_configurations,
                ensemble_size=ensemble_size
            )

            # Instantiate traditional ensemble model
            ensemble_model_traditional = self.build_traditional_ensemble_net(
                model_configurations=model_configurations,
                ensemble_size=ensemble_size
            )

            # Instantiate MLP model
            mlp_model = self.build_mlp(
                model_configurations=model_configurations
            )
            mlp_model.to(self.__device)  # Move to GPU

            # Instantiate optimizer
            optimizer = torch.optim.Adam(
                mlp_model.parameters(), lr=learning_rate)

            # ======================================================
            # Static NCL ensemble model

            # Train static NCL ensemble model
            # print(f"Training static NCL ensemble model in experiment ...", flush=True)

            ensemble_model_static_ncl.train(
                data_loader_train=data_loader_train,
                data_loader_validation=data_loader_validation,
                epoch_num=epoch_num,
                learning_rate=learning_rate,
                loss_function=loss_function,
                lambda_ncl=correlation_penalty_coefficient,
                epoch_num_per_log=epoch_num_per_log,
                validation_cycle=validation_cycle,
                validation_epoch_window=validation_epoch_window
            )
            total_loss_train_static_ncl, batch_num_train_static_ncl, y_predictions_all_train_static_ncl = self.test_ensemble_model_static_ncl(
                ensemble_model=ensemble_model_static_ncl,
                loss_function=loss_function,
                data_loader=data_loader_train,
                lambda_ncl=correlation_penalty_coefficient
            )

            average_batch_losses_train_static_ncl.append(
                total_loss_train_static_ncl / batch_num_train_static_ncl)

            # Test static NCL ensemble model
            # print(f"Testing static NCL ensemble model in experiment ...", flush=True)

            total_loss_test_static_ncl, batch_num_test_static_ncl, y_predictions_all_test_static_ncl = self.test_ensemble_model_static_ncl(
                ensemble_model=ensemble_model_static_ncl,
                loss_function=loss_function,
                data_loader=data_loader_test,
                lambda_ncl=correlation_penalty_coefficient
            )

            average_batch_losses_test_static_ncl.append(
                total_loss_test_static_ncl / batch_num_test_static_ncl)

            diversity_coefficients_test_static_ncl.append(torch.mean(maths_helper.diversity_coefficient(
                y_predictions=y_predictions_all_test_static_ncl), dim=0).item())

            diversity_coefficients_train_static_ncl.append(torch.mean(maths_helper.diversity_coefficient(
                y_predictions=y_predictions_all_train_static_ncl), dim=0).item())

            # ======================================================
            # Traditional ensemble model

            # Train traditional ensemble model
            # print(f"Training traditional ensemble model in experiment ...", flush=True)

            ensemble_model_traditional.train(
                data_loader_train=data_loader_train,
                epoch_num=epoch_num,
                learning_rate=learning_rate,
                loss_function=loss_function,
                epoch_num_per_log=epoch_num_per_log
            )
            total_loss_train_traditional, batch_num_train_traditional, y_predictions_all_train_traditional = self.test_ensemble_model_traditional(
                ensemble_model=ensemble_model_traditional,
                loss_function=loss_function,
                data_loader=data_loader_train
            )

            average_batch_losses_train_traditional.append(
                total_loss_train_traditional / batch_num_train_traditional)

            # Test traditional ensemble model
            # print(f"Testing traditional ensemble model in experiment ...", flush=True)

            total_loss_test_traditional, batch_num_test_traditional, y_predictions_all_test_traditional = self.test_ensemble_model_traditional(
                ensemble_model=ensemble_model_traditional,
                loss_function=loss_function,
                data_loader=data_loader_test
            )

            average_batch_losses_test_traditional.append(
                total_loss_test_traditional / batch_num_test_traditional)

            diversity_coefficients_test_traditional.append(torch.mean(maths_helper.diversity_coefficient(
                y_predictions=y_predictions_all_test_traditional), dim=0).item())

            diversity_coefficients_train_traditional.append(torch.mean(maths_helper.diversity_coefficient(
                y_predictions=y_predictions_all_train_traditional), dim=0).item())

            # ======================================================
            # MLP model

            # Train + test MLP model
            # There is a more efficient way, which is to pick the largest epoch and evaluate model for each target training epoch (reduce number of times to train models as we are training the model once assuming other hyperparameters stay the same except epoch_num)
            # However, that will ruin the consistency of this code so I did not use it
            # It is a trade-off between code consistency and execution time
            # print(f"Training and testing MLP model in experiment ...", flush=True)

            for epoch in range(epoch_num):
                total_loss_train_mlp = 0

                # Training
                mlp_model.train()
                for (batch_X, batch_y) in data_loader_train:
                    batch_X = batch_X.to(self.__device)
                    batch_y = batch_y.to(self.__device)

                    # y_prediction is a PyTorch tensor
                    y_prediction = mlp_model(batch_X)

                    loss = loss_function(y_prediction, batch_y)

                    # Backward pass and optimization
                    optimizer.zero_grad()  # reset gradients
                    loss.backward()        # compute gradients
                    optimizer.step()       # update weights

                    total_loss_train_mlp += loss.item()

                average_batch_loss_train_mlp = total_loss_train_mlp / \
                    len(data_loader_train)

                # Log
                # if epoch % epoch_num_per_log == 0:
                #     print(
                #         f"Epoch: {epoch + 1} / {epoch_num}; Loss: {average_batch_loss_train_mlp:.4f}", flush=True)

                if epoch == epoch_num - 1:
                    # Last epoch
                    average_batch_losses_train_mlp.append(
                        average_batch_loss_train_mlp)

            # ============================================================================================

            # Testing
            total_loss_test_mlp, batch_num_test_mlp = self.__mlp_model_helper.test_model(
                model=mlp_model,
                loss_function=loss_function,
                data_loader_test=data_loader_test
            )
            average_batch_loss_test_mlp = total_loss_test_mlp / batch_num_test_mlp
            average_batch_losses_test_mlp.append(average_batch_loss_test_mlp)

        # Add columns to the dataframe
        self.__df_experiment["average_loss_train_static_ncl"] = average_batch_losses_train_static_ncl
        self.__df_experiment["average_loss_test_static_ncl"] = average_batch_losses_test_static_ncl
        self.__df_experiment["average_loss_train_traditional"] = average_batch_losses_train_traditional
        self.__df_experiment["average_loss_test_traditional"] = average_batch_losses_test_traditional
        self.__df_experiment["average_loss_train_mlp"] = average_batch_losses_train_mlp
        self.__df_experiment["average_loss_test_mlp"] = average_batch_losses_test_mlp
        self.__df_experiment["diversity_coefficient_train_static_ncl"] = diversity_coefficients_train_static_ncl
        self.__df_experiment["diversity_coefficient_test_static_ncl"] = diversity_coefficients_test_static_ncl
        self.__df_experiment["diversity_coefficient_train_traditional"] = diversity_coefficients_train_traditional
        self.__df_experiment["diversity_coefficient_test_traditional"] = diversity_coefficients_test_traditional
        self.__df_experiment["hidden_size"] = hidden_sizes
        # For plotting graph -> cannot plot list -> must be a string
        self.__df_experiment["hidden_size"] = self.__df_experiment["hidden_size"].apply(
            str)
        self.__df_experiment["ensemble_size"] = ensemble_sizes

    def save_csv(self, repository_experiment_csv_path: str, csv_name: str) -> None:
        assert repository_experiment_csv_path and csv_name, f"Invalid final csv path: {repository_experiment_csv_path}/{csv_name}.csv"
        self.__df_experiment.to_csv(
            f"{repository_experiment_csv_path}/{csv_name}.csv", index=False)

    def plot_graph(self, repository_experiment_img_path: str, img_name: str, save_img=True, image_resolution=300, show_img=False) -> None:
        # Set plot style for nicer plots
        sns.set_theme(style="whitegrid", rc={
            "font.size": FONT_SIZE,  # Default font size
            "font.family": FONT_FAMILY,
            "font.serif": FONT_SERIF,
            "axes.titlesize": AXES_TITLESIZE,
            "axes.titleweight": AXES_TITLEWEIGHT,
            "axes.labelsize": AXES_LABELSIZE,
            "legend.fontsize": LEGEND_FONTSIZE
        })

        plt.figure(figsize=(16, 24))  # Larger figure for 10 plots

        row_num = len(self.__hyperparameters)
        column_num = 3

        for i, hyperparameter in enumerate(self.__hyperparameters, 1):
            hyperparameter_text = hyperparameter
            if hyperparameter == "correlation_penalty_coefficient":
                hyperparameter_text = "$\\lambda$"
            elif hyperparameter == "ensemble_size":
                hyperparameter_text = "ensemble size"
            elif hyperparameter == "epoch_num":
                hyperparameter_text = "train epoch"
            elif hyperparameter == "learning_rate":
                hyperparameter_text = "learning rate"

            column_index = 0  # 0-indexed

            # Average loss
            plt.subplot(row_num, column_num, (i - 1) *
                        column_num + (column_index + 1))
            # sns.lineplot(
            #     data=self.__df_experiment,
            #     x=hyperparameter,
            #     y="average_loss_train_static_ncl",
            #     marker='o',
            #     label=f"Train - static NCL ensemble model"
            # )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_test_static_ncl",
                marker='o',
                label=f"Test - static NCL ensemble model"
            )
            # sns.lineplot(
            #     data=self.__df_experiment,
            #     x=hyperparameter,
            #     y="average_loss_train_traditional",
            #     marker='o',
            #     label=f"Train - traditional ensemble model"
            # )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_test_traditional",
                marker='o',
                label=f"Test - traditional ensemble model"
            )
            # sns.lineplot(
            #     data=self.__df_experiment,
            #     x=hyperparameter,
            #     y="average_loss_train_mlp",
            #     marker='o',
            #     label=f"Train - MLP model"
            # )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_test_mlp",
                marker='o',
                label=f"Test - MLP model"
            )
            plt.title(f"MSE vs {hyperparameter_text}")
            plt.xlabel(hyperparameter_text)
            plt.ylabel("MSE")
            plt.legend()

            # ===================================================================

            # Diversity Coefficient
            column_index = 1  # 0-indexed

            plt.subplot(row_num, column_num, (i - 1) *
                        column_num + (column_index + 1))
            # sns.lineplot(
            #     data=self.__df_experiment,
            #     x=hyperparameter,
            #     y="diversity_coefficient_train_static_ncl",
            #     marker='o',
            #     label=f"Train - static NCL ensemble model"
            # )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="diversity_coefficient_test_static_ncl",
                marker='o',
                label=f"Test - static NCL ensemble model"
            )
            # sns.lineplot(
            #     data=self.__df_experiment,
            #     x=hyperparameter,
            #     y="diversity_coefficient_train_traditional",
            #     marker='o',
            #     label=f"Train - traditional ensemble model"
            # )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="diversity_coefficient_test_traditional",
                marker='o',
                label=f"Test - traditional ensemble model"
            )
            plt.title(
                f"$\\rho$ vs {hyperparameter_text}")
            plt.xlabel(hyperparameter_text)
            plt.ylabel("$\\rho$")
            plt.legend()

            # ===================================================================

            # Loss against diversity coefficient
            column_index = 2  # 0-indexed

            if i == 1:
                # We only want to plot 1 graph for all hyparameters for this graph
                # Remember i starts with 1 instead of 0

                plt.subplot(row_num, column_num, (i - 1) *
                            column_num + (column_index + 1))
                # sns.lineplot(
                #     data=self.__df_experiment,
                #     x=f"diversity_coefficient_train_static_ncl",
                #     y="average_loss_train_static_ncl",
                #     marker='o',
                #     label=f"Train - static NCL ensemble model"
                # )
                sns.lineplot(
                    data=self.__df_experiment,
                    x=f"diversity_coefficient_test_static_ncl",
                    y="average_loss_test_static_ncl",
                    marker='o',
                    label=f"Test - static NCL ensemble model"
                )
                # sns.lineplot(
                #     data=self.__df_experiment,
                #     x=f"diversity_coefficient_train_traditional",
                #     y="average_loss_train_traditional",
                #     marker='o',
                #     label=f"Train - traditional ensemble model"
                # )
                sns.lineplot(
                    data=self.__df_experiment,
                    x=f"diversity_coefficient_test_traditional",
                    y="average_loss_test_traditional",
                    marker='o',
                    label=f"Test - traditional ensemble model"
                )
                plt.title(
                    f"MSE vs $\\rho$")
                plt.xlabel(f"$\\rho$")
                plt.ylabel("MSE")
                plt.legend()

            # ===================================================================

            # Loss against lambda (fixed hidden node num)
            column_index = 2  # 0-indexed

            if i == 2:
                # We only want to plot 1 graph for all hyparameters for this graph
                # Remember i starts with 1 instead of 0
                fixed_hidden_sizes = [2, 2, 2]
                target_ensemble_sizes = [2, 6, 12]

                """
                NOTE: If we read from csv, df_experiment["hidden_size"] will be of type class "str"
                Deprecated: If we create the csv, df_experiment["hidden_size"] will be of type class "list"
                NOTE: If we create the csv, df_experiment["hidden_size"] will be of type class "str" as I convert it to string before saving to csv for graph plotting
                """
                self.__df_experiment["hidden_size_list"] = self.__df_experiment["hidden_size"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                df_filtered_hidden_size = self.__df_experiment[self.__df_experiment["hidden_size_list"].apply(
                    lambda x: x == fixed_hidden_sizes
                )]
                df_filtered_hidden_size = df_filtered_hidden_size[df_filtered_hidden_size["ensemble_size"].apply(
                    lambda x: x in target_ensemble_sizes
                )]

                plt.subplot(row_num, column_num, (i - 1) *
                            column_num + (column_index + 1))
                sns.lineplot(
                    data=df_filtered_hidden_size,
                    x="correlation_penalty_coefficient",
                    y="average_loss_test_static_ncl",
                    hue="ensemble_size",
                    marker='o',
                    palette="tab10"
                )
                sns.lineplot(
                    data=self.__df_experiment,
                    x="correlation_penalty_coefficient",
                    y="average_loss_test_mlp",
                    marker='o',
                    label=f"Test - MLP model"
                )
                plt.title(
                    f"MSE vs $\\lambda$ \n({sum(fixed_hidden_sizes)} hidden nodes per base learner)")
                plt.xlabel(f"$\\lambda$")
                plt.ylabel("MSE")
                plt.legend()

            # ===================================================================

            # Loss against lambda (fixed base learner num)
            column_index = 2  # 0-indexed

            if i == 3:
                # We only want to plot 1 graph for all hyparameters for this graph
                # Remember i starts with 1 instead of 0
                fixed_ensemble_size = 6
                target_hidden_sizes = [[2], [2, 2, 2], [2, 2, 2, 2, 2, 2]]

                df_filtered_ensemble_size = self.__df_experiment[self.__df_experiment["ensemble_size"].apply(
                    lambda x: x == fixed_ensemble_size
                )]
                df_filtered_ensemble_size = df_filtered_ensemble_size[df_filtered_ensemble_size["hidden_size_list"].apply(
                    lambda x: x in target_hidden_sizes
                )]

                plt.subplot(row_num, column_num, (i - 1) *
                            column_num + (column_index + 1))
                sns.lineplot(
                    data=df_filtered_ensemble_size,
                    x="correlation_penalty_coefficient",
                    y="average_loss_test_static_ncl",
                    hue="hidden_size",
                    marker='o',
                    palette="tab10"
                )
                sns.lineplot(
                    data=self.__df_experiment,
                    x="correlation_penalty_coefficient",
                    y="average_loss_test_mlp",
                    marker='o',
                    label=f"Test - MLP model"
                )
                plt.title(
                    f"MSE vs $\\lambda$ \n({fixed_ensemble_size} base learners per ensemble model)")
                plt.xlabel(f"$\\lambda$")
                plt.ylabel("MSE")
                plt.legend()

            # ===================================================================

            # Diversity coefficient against lambda (fixed hidden node num)
            column_index = 2  # 0-indexed

            if i == 4:
                # We only want to plot 1 graph for all hyparameters for this graph
                # Remember i starts with 1 instead of 0
                fixed_hidden_sizes = [2, 2, 2]
                target_ensemble_sizes = [2, 6, 12]

                """
                NOTE: If we read from csv, df_experiment["hidden_size"] will be of type class "str"
                Deprecated: If we create the csv, df_experiment["hidden_size"] will be of type class "list"
                NOTE: If we create the csv, df_experiment["hidden_size"] will be of type class "str" as I convert it to string before saving to csv for graph plotting
                """
                self.__df_experiment["hidden_size_list"] = self.__df_experiment["hidden_size"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                df_filtered_hidden_size = self.__df_experiment[self.__df_experiment["hidden_size_list"].apply(
                    lambda x: x == fixed_hidden_sizes
                )]
                df_filtered_hidden_size = df_filtered_hidden_size[df_filtered_hidden_size["ensemble_size"].apply(
                    lambda x: x in target_ensemble_sizes
                )]

                plt.subplot(row_num, column_num, (i - 1) *
                            column_num + (column_index + 1))
                sns.lineplot(
                    data=df_filtered_hidden_size,
                    x="correlation_penalty_coefficient",
                    y="diversity_coefficient_test_static_ncl",
                    hue="ensemble_size",
                    marker='o',
                    palette="tab10"
                )
                plt.title(
                    f"$\\rho$ vs $\\lambda$ \n({sum(fixed_hidden_sizes)} hidden nodes per base learner)")
                plt.xlabel(f"$\\lambda$")
                plt.ylabel("$\\rho$")
                plt.legend()

            # ===================================================================

            # Loss against lambda (fixed base learner num)
            column_index = 2  # 0-indexed

            if i == 5:
                # We only want to plot 1 graph for all hyparameters for this graph
                # Remember i starts with 1 instead of 0
                fixed_ensemble_size = 6
                target_hidden_sizes = [[2], [2, 2, 2], [2, 2, 2, 2, 2, 2]]

                df_filtered_ensemble_size = self.__df_experiment[self.__df_experiment["ensemble_size"].apply(
                    lambda x: x == fixed_ensemble_size
                )]
                df_filtered_ensemble_size = df_filtered_ensemble_size[df_filtered_ensemble_size["hidden_size_list"].apply(
                    lambda x: x in target_hidden_sizes
                )]

                plt.subplot(row_num, column_num, (i - 1) *
                            column_num + (column_index + 1))
                sns.lineplot(
                    data=df_filtered_ensemble_size,
                    x="correlation_penalty_coefficient",
                    y="diversity_coefficient_test_static_ncl",
                    hue="hidden_size",
                    marker='o',
                    palette="tab10"
                )
                plt.title(
                    f"$\\rho$ vs $\\lambda$ \n({fixed_ensemble_size} base learners per ensemble model)")
                plt.xlabel(f"$\\lambda$")
                plt.ylabel("$\\rho$")
                plt.legend()

        plt.tight_layout()

        if save_img:
            assert repository_experiment_img_path and img_name, f"Invalid final image path: {repository_experiment_img_path}/{img_name}.png"

            # Save graph as image if applicable
            plt.savefig(f"{repository_experiment_img_path}/{img_name}.png",
                        dpi=image_resolution)

        if show_img:
            # Show graph if applicable
            # Might block the execution so we set show_img = False for non-interactive Python script
            # Jupyter notebook will still show the image if show_img = False though
            plt.show()

    @property
    def device(self):
        return self.__device

    @property
    def df_experiment(self):
        return self.__df_experiment

# ============================================================================================================


class ExperimentNEATHelper:
    def __init__(self, test_size: float, random_state: int, batch_size: int, experiment_model_repository_path: str, config_path: str, epoch_nums=[100], learning_rates=[0.01], evolution_epoches=[50], max_fitnesses=[20], validation_size=None):
        # Experiment hyperparameter
        self.__epoch_nums = epoch_nums
        self.__learning_rates = learning_rates
        self.__evolution_epoches = evolution_epoches
        self.__max_fitnesses = max_fitnesses

        # Configuration
        self.__test_size = test_size
        self.__validation_size = validation_size
        self.__random_state = random_state
        self.__batch_size = batch_size
        self.__experiment_model_repository_path = experiment_model_repository_path
        self.__config_path = config_path

        self.__device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # List of hyperparameters to plot against loss
        self.__hyperparameters = [
            "epoch_num", "learning_rate", "evolution_epoch", "max_fitness"]

        # For MLP model in this experiment
        self.__mlp_model_helper = MLPHelper(self.__device)

    def get_data_loaders(self, filename: str, column_names: list, output_column_name: str, normalise_data=True, remove_outliers=False) -> tuple:
        """
        Return data loaders of training and testing data
        Return (training_data_loader, testing_data_loader)

        :return: (training_data_loader, testing_data_loader)
        """
        return data_helper.get_data_loaders(
            filename=filename,
            column_names=column_names,
            output_column_name=output_column_name,
            test_size=self.__test_size,
            random_state=self.__random_state,
            batch_size=self.__batch_size,
            validation_size=self.__validation_size,
            normalise_data=normalise_data,
            remove_outliers=remove_outliers
        )

    def build_mlp(self, model_configurations: dict) -> MLP:
        # Instantiate MLP model
        return MLP.build_from_config(
            input_size=model_configurations["input_size"],
            hidden_sizes=model_configurations["hidden_sizes"],
            activations=model_configurations["activations"],
            activation_type=model_configurations["activation_type"],
            output_size=model_configurations["output_size"],
            dropout_rate=model_configurations["dropout_rate"],
            dropout_indices=model_configurations["dropout_indices"]
        )

    def build_neat(self, evolution_epoch: int, max_fitness: float, loss_function: any) -> NEATBaseLearner:
        # Instantiate NEAT model
        return NEATBaseLearner.build_from_config_file(
            config_path=self.__config_path,
            evolution_epoch=evolution_epoch,
            max_fitness=max_fitness,
            device=self.__device,
            # Does not matter
            rank=0,
            # Does not matter
            device_index=0,
            loss_function=loss_function,
            use_cpu=False,
            model_repository_path=self.__experiment_model_repository_path
        )

    def test_neat(self, neat_model: NEATBaseLearner, loss_function, data_loader_test) -> tuple:
        """
        Test a NEAT model
        Return (total_loss, batch_num)
        To get the average loss per batch, calculate (total_loss / batch_num)

        :param: neat_model: The NEATBaseLearner model instance
        :param: loss_function: The loss function
        :param: data_loader_test: The testing data loader
        :return: (total_loss, batch_num)
        """
        batch_losses = []

        # eval() and torch.no_grad() will be set automatically in NEATBaseLearner

        for batch_X, batch_y in data_loader_test:
            batch_X = batch_X.to(self.__device)
            batch_y = batch_y.to(self.__device)

            y_prediction = neat_model(batch_X)

            loss = loss_function(y_prediction, batch_y)

            batch_losses.append(loss.item())

            return (sum(batch_losses), len(data_loader_test))

    def experiment(self, model_configurations: dict, loss_function: any, data_loader_train, data_loader_test, epoch_num_per_log=20):
        """
        Run the experiment
        To use a customised loss function, create a Python function and pass the function instance to train_model()

        :param: loss_function: The loss function of models
        """
        # Hyperparameter combinations
        param_combinations = list(product(
            self.__epoch_nums,
            self.__learning_rates,
            self.__evolution_epoches,
            self.__max_fitnesses
        ))

        self.__df_experiment = pd.DataFrame(
            param_combinations, columns=self.__hyperparameters)

        # For storing experiment output data so that we can plot graphs
        # Collect average batch loss
        # NOTE: Will not have calculate average batch loss for training as we might just use the weights trained by neat-python directly
        average_batch_losses_test_neat = []  # shape: [epoch, 1]
        average_batch_losses_test_mlp = []  # shape: [epoch, 1]
        node_nums_neat = []  # shape: [epoch, 1]
        connection_nums_neat = []  # shape: [epoch, 1]

        for row_index, row in self.__df_experiment.iterrows():
            epoch_num = int(row["epoch_num"])
            learning_rate = float(row["learning_rate"])
            evolution_epoch = int(row["evolution_epoch"])
            max_fitness = float(row["max_fitness"])

            # Log
            print(f"============================================", flush=True)
            print(
                f"Experiment {row_index + 1} / {len(self.__df_experiment)} parameters: ", flush=True)
            print(f"epoch_num: {epoch_num}", flush=True)
            print(f"learning_rate: {learning_rate}", flush=True)
            print(f"evolution_epoch: {evolution_epoch}", flush=True)
            print(f"max_fitness: {max_fitness}", flush=True)

            # Instantiate NEAT model
            neat_model = self.build_neat(
                evolution_epoch=evolution_epoch,
                max_fitness=max_fitness,
                loss_function=loss_function
            )

            # Instantiate MLP model
            mlp_model = self.build_mlp(
                model_configurations=model_configurations
            )
            mlp_model.to(self.__device)  # Move to GPU

            # Instantiate optimizer
            optimizer = torch.optim.Adam(
                mlp_model.parameters(), lr=learning_rate)

            # ======================================================

            # Train NEAT model
            print(f"Training NEAT model in experiment ...", flush=True)

            # NOTE: We will do training again before selection
            # NOTE: batch_X and batch_y will be moved to GPU automatically in evolve(...)
            neat_model.evolve(data_loader_train=data_loader_train,
                              train_epoch=epoch_num, learning_rate=learning_rate)

            # ============================================================================================

            # Testing
            total_loss_test_neat, batch_num_test_neat = self.test_neat(
                neat_model=neat_model,
                loss_function=loss_function,
                data_loader_test=data_loader_test
            )
            average_batch_loss_test_neat = total_loss_test_neat / batch_num_test_neat
            average_batch_losses_test_neat.append(average_batch_loss_test_neat)

            # Expect neat_model.model_genome is not None
            node_nums_neat.append(len(neat_model.genome_nodes))
            connection_nums_neat.append(len(neat_model.genome_connections))

            # ============================================================================================

            # MLP model

            # Train + test MLP model
            # There is a more efficient way, which is to pick the largest epoch and evaluate model for each target training epoch (reduce number of times to train models as we are training the model once assuming other hyperparameters stay the same except epoch_num)
            # However, that will ruin the consistency of this code so I did not use it
            # It is a trade-off between code consistency and execution time
            print(f"Training and testing MLP model in experiment ...", flush=True)
            for epoch in range(epoch_num):
                # Training
                mlp_model.train()

                for (batch_X, batch_y) in data_loader_train:
                    batch_X = batch_X.to(self.__device)
                    batch_y = batch_y.to(self.__device)

                    # y_prediction is a PyTorch tensor
                    y_prediction = mlp_model(batch_X)

                    loss = loss_function(y_prediction, batch_y)

                    # Backward pass and optimization
                    optimizer.zero_grad()  # reset gradients
                    loss.backward()        # compute gradients
                    optimizer.step()       # update weights

                # We do not care about loss in training

            # ============================================================================================

            # Testing
            total_loss_test_mlp, batch_num_test_mlp = self.__mlp_model_helper.test_model(
                model=mlp_model,
                loss_function=loss_function,
                data_loader_test=data_loader_test
            )
            average_batch_loss_test_mlp = total_loss_test_mlp / batch_num_test_mlp
            average_batch_losses_test_mlp.append(average_batch_loss_test_mlp)

            # ============================================================================================

            # NOTE: We need this cause right now -> every time the experiment is run, it is terminated
            print(
                f"average_loss_test_neat: {average_batch_loss_test_neat}", flush=True)
            print(
                f"average_loss_test_mlp: {average_batch_loss_test_mlp}", flush=True)
            print(f"node_num_neat: {len(neat_model.genome_nodes)}", flush=True)
            print(
                f"connection_num_neat: {len(neat_model.genome_connections)}", flush=True)

        # Add columns of average_loss to the dataframe
        self.__df_experiment["average_loss_test_neat"] = average_batch_losses_test_neat
        self.__df_experiment["average_loss_test_mlp"] = average_batch_losses_test_mlp
        self.__df_experiment["node_num_neat"] = node_nums_neat
        self.__df_experiment["connection_num_neat"] = connection_nums_neat

    def save_csv(self, repository_experiment_csv_path: str, csv_name: str) -> None:
        assert repository_experiment_csv_path and csv_name, f"Invalid final csv path: {repository_experiment_csv_path}/{csv_name}.csv"
        self.__df_experiment.to_csv(
            f"{repository_experiment_csv_path}/{csv_name}.csv", index=False)

    def plot_graph(self, repository_experiment_img_path: str, img_name: str, save_img=True, image_resolution=300, show_img=False) -> None:
        # Set plot style for nicer plots
        sns.set_theme(style="whitegrid", rc={
            "font.size": FONT_SIZE,  # Default font size
            "font.family": FONT_FAMILY,
            "font.serif": FONT_SERIF,
            "axes.titlesize": AXES_TITLESIZE,
            "axes.titleweight": AXES_TITLEWEIGHT,
            "axes.labelsize": AXES_LABELSIZE,
            "legend.fontsize": LEGEND_FONTSIZE
        })

        plt.figure(figsize=(24, 32))  # Larger figure for 12 plots

        row_num = len(self.__hyperparameters)
        column_num = 3

        for i, hyperparameter in enumerate(self.__hyperparameters, 1):
            column_index = 0  # 0-indexed

            # Average loss
            plt.subplot(row_num, column_num, (i - 1) *
                        column_num + (column_index + 1))
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_test_neat",
                marker='o',
                label=f"Test - NEAT model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_test_mlp",
                marker='o',
                label=f"Test - MLP model"
            )
            plt.title(f"Average Loss vs {hyperparameter}")
            plt.xlabel(hyperparameter)
            plt.ylabel("Average Loss")
            plt.legend()

            # ===================================================================
            column_index = 1  # 0-indexed

            # Node num
            plt.subplot(row_num, column_num, (i - 1) *
                        column_num + (column_index + 1))
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="node_num_neat",
                marker='o',
                label=f"NEAT model - number of nodes"
            )
            plt.title(f"Number of Non-input Nodes vs {hyperparameter}")
            plt.xlabel(hyperparameter)
            plt.ylabel("Number of Non-input Nodes")
            plt.legend()

            # ===================================================================
            column_index = 2  # 0-indexed

            # Connection num
            plt.subplot(row_num, column_num, (i - 1) *
                        column_num + (column_index + 1))
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="connection_num_neat",
                marker='o',
                label=f"NEAT model - number of connections"
            )
            plt.title(f"Number of Connections vs {hyperparameter}")
            plt.xlabel(hyperparameter)
            plt.ylabel("Number of Connections")
            plt.legend()

        plt.tight_layout()

        if save_img:
            assert repository_experiment_img_path and img_name, f"Invalid final image path: {repository_experiment_img_path}/{img_name}.png"

            # Save graph as image if applicable
            plt.savefig(f"{repository_experiment_img_path}/{img_name}.png",
                        dpi=image_resolution)

        if show_img:
            # Show graph if applicable
            # Might block the execution so we set show_img = False for non-interactive Python script
            # Jupyter notebook will still show the image if show_img = False though
            plt.show()

    @property
    def device(self):
        return self.__device

    @property
    def df_experiment(self):
        return self.__df_experiment

# ============================================================================================================


class ExperimentNEATNCLHelper:
    def __init__(self, test_size: float, random_state: int, batch_size: int, experiment_model_repository_path: str, master_address: str, master_port: str, device_num: int, dist_backend: str, config: neat.Config, correlation_penalty_coefficients=[0.5], ensemble_sizes=[1, 2], epoch_nums=[100], learning_rates=[0.01], evolution_epoches=[100], max_fitnesses=[50], validation_size=None):
        # Experiment hyperparameter
        self.__correlation_penalty_coefficients = correlation_penalty_coefficients
        self.__ensemble_sizes = ensemble_sizes
        self.__epoch_nums = epoch_nums
        self.__learning_rates = learning_rates
        self.__evolution_epoches = evolution_epoches
        self.__max_fitnesses = max_fitnesses

        # Configuration
        self.__test_size = test_size
        self.__validation_size = validation_size
        self.__random_state = random_state
        self.__batch_size = batch_size
        self.__experiment_model_repository_path = experiment_model_repository_path
        self.__master_address = master_address
        self.__master_port = master_port
        self.__device_num = device_num
        self.__dist_backend = dist_backend
        # For NEAT
        self.__config = config

        self.__device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # List of hyperparameters to plot against loss
        self.__hyperparameters = [
            "correlation_penalty_coefficient", "ensemble_size", "epoch_num", "learning_rate", "evolution_epoch", "max_fitness"]

        # For MLP model in this experiment
        self.__mlp_model_helper = MLPHelper(self.__device)

    def get_data_loaders(self, filename: str, column_names: list, output_column_name: str, normalise_data=True, remove_outliers=False) -> tuple:
        """
        Return data loaders of training and testing data
        Return (training_data_loader, testing_data_loader)

        :return: (training_data_loader, testing_data_loader)
        """
        return data_helper.get_data_loaders(
            filename=filename,
            column_names=column_names,
            output_column_name=output_column_name,
            test_size=self.__test_size,
            random_state=self.__random_state,
            batch_size=self.__batch_size,
            validation_size=self.__validation_size,
            normalise_data=normalise_data,
            remove_outliers=remove_outliers
        )

    def build_neat_ncl_ensemble_net(self, config: neat.Config, evolution_epoch: int, max_fitness: float, ensemble_size: int) -> NEATNCLEnsembleNet:
        # Instantiate NEAT NCL ensemble net
        return NEATNCLEnsembleNet(
            config=config,
            evolution_epoch=evolution_epoch,
            max_fitness=max_fitness,
            device_num=self.__device_num,
            base_learner_nums=[ensemble_size] * self.__device_num,
            dist_backend=self.__dist_backend,
            master_address=self.__master_address,
            master_port=self.__master_port,
            model_repository_path=self.__experiment_model_repository_path
        )

    def build_static_ncl_ensemble_net(self, model_configurations: dict, ensemble_size: int) -> StaticNCLEnsembleNet:
        # Instantiate static NCL ensemble net
        return StaticNCLEnsembleNet(
            device_num=self.__device_num,
            base_learner_nums=[ensemble_size] * self.__device_num,
            model_configurations=model_configurations,
            dist_backend=self.__dist_backend,
            master_address=self.__master_address,
            master_port=self.__master_port,
            model_repository_path=self.__experiment_model_repository_path
        )

    def build_traditional_ensemble_net(self, model_configurations: dict, ensemble_size: int) -> TraditionalEnsembleNet:
        # Instantiate traditional ensemble net
        return TraditionalEnsembleNet(
            device_num=self.__device_num,
            base_learner_nums=[ensemble_size] * self.__device_num,
            model_configurations=model_configurations,
            dist_backend=self.__dist_backend,
            master_address=self.__master_address,
            master_port=self.__master_port,
            model_repository_path=self.__experiment_model_repository_path
        )

    def build_mlp(self, model_configurations: dict) -> MLP:
        # Instantiate MLP model
        return MLP.build_from_config(
            input_size=model_configurations["input_size"],
            hidden_sizes=model_configurations["hidden_sizes"],
            activations=model_configurations["activations"],
            activation_type=model_configurations["activation_type"],
            output_size=model_configurations["output_size"],
            dropout_rate=model_configurations["dropout_rate"],
            dropout_indices=model_configurations["dropout_indices"]
        )

    def evaluate_ensemble_model_static_ncl(self, ensemble_model: StaticNCLEnsembleNet, loss_function, data_loader, lambda_ncl: float) -> tuple:
        """
        Test a static NCL ensemble model
        This method might be invoked with training data
        Do not be mislead by the method name -> this function just evaluates the performance of a model against a dataset, which can be either training or testing
        Return (total_loss, batch_num, y_predictions_all)
        To get the average loss per batch, calculate (total_loss / batch_num)

        :param: ensemble_model: The ensemble model instance
        :param: loss_function: The loss function
        :param: data_loader: The training/testing data loader
        :param: lambda_ncl: Correlation penalty coefficient
        :return: (total_loss, batch_num, y_predictions_all)
        """
        # model.eval() and torch.no_grad() will be done StaticNCLEnsembleNet

        total_loss = 0
        y_predictions_all_raw = []

        for (batch_X, batch_y) in data_loader:
            # Shape: [base_learner_num, batch_num, 1]
            y_predictions_all_batch = ensemble_model.evaluate_all_predictions(
                batch_X, lambda_ncl=lambda_ncl)

            y_predictions_all_raw.append(y_predictions_all_batch)

            y_prediction = ensemble_model.vote(y_predictions_all_batch)

            loss = loss_function(y_prediction, batch_y)

            total_loss += loss.item()

        # Shape: [base_learner_num, x_num, 1]
        y_predictions_all = torch.cat(y_predictions_all_raw, dim=1)

        return (total_loss, len(data_loader), y_predictions_all)

    def evaluate_ensemble_model_neat_ncl(self, ensemble_model: NEATNCLEnsembleNet, loss_function, data_loader) -> tuple:
        """
        Test a NEAT NCL ensemble model
        This method might be invoked with training data
        Do not be mislead by the method name -> this function just evaluates the performance of a model against a dataset, which can be either training or testing
        Return (total_loss, batch_num, y_predictions_all)
        To get the average loss per batch, calculate (total_loss / batch_num)
        NOTE: Technically, this method is the same as evalaute_ensemble_model_static_ncl() but because lambda_ncl is required in evaluate_all() of Static NCL (lambda_ncl can be removed from the method after refactoring), we create this separate method

        :param: ensemble_model: The ensemble model instance
        :param: loss_function: The loss function
        :param: data_loader: The training/testing data loader
        :return: (total_loss, batch_num, y_predictions_all)
        """
        # model.eval() and torch.no_grad() will be done StaticNCLEnsembleNet

        total_loss = 0
        y_predictions_all_raw = []

        for (batch_X, batch_y) in data_loader:
            # Shape: [base_learner_num, batch_num, 1]
            y_predictions_all_batch = ensemble_model.evaluate_all(batch_X)

            y_predictions_all_raw.append(y_predictions_all_batch)

            y_prediction = ensemble_model.vote(y_predictions_all_batch)

            loss = loss_function(y_prediction, batch_y)

            total_loss += loss.item()

        # Shape: [base_learner_num, x_num, 1]
        y_predictions_all = torch.cat(y_predictions_all_raw, dim=1)

        return (total_loss, len(data_loader), y_predictions_all)

    def evaluate_ensemble_model_traditional(self, ensemble_model: TraditionalEnsembleNet, loss_function, data_loader) -> tuple:
        """
        Test a traditional ensemble model
        This method might be invoked with training data
        Do not be mislead by the method name -> this function just evaluates the performance of a model against a dataset, which can be either training or testing
        Return (total_loss, batch_num, y_predictions_all)
        To get the average loss per batch, calculate (total_loss / batch_num)

        :param: ensemble_model: The ensemble model instance
        :param: loss_function: The loss function
        :param: data_loader: The training/testing data loader
        :return: (total_loss, batch_num, y_predictions_all)
        """
        # model.eval() and torch.no_grad() will be done TraditionalEnsembleNet

        total_loss = 0
        y_predictions_all_raw = []

        for (batch_X, batch_y) in data_loader:
            # Shape: [base_learner_num, batch_num, 1]
            y_predictions_all_batch = ensemble_model.evaluate_all(batch_X)

            y_predictions_all_raw.append(y_predictions_all_batch)

            y_prediction = ensemble_model.vote(y_predictions_all_batch)

            loss = loss_function(y_prediction, batch_y)

            total_loss += loss.item()

        # Shape: [base_learner_num, x_num, 1]
        y_predictions_all = torch.cat(y_predictions_all_raw, dim=1)

        return (total_loss, len(data_loader), y_predictions_all)

    def get_predictions_std(self, y_predictions_all: torch.Tensor) -> float:
        """
        NOTE: Deprecated
        We will calcuate the standard deviation of predictions of each batch
        Then we will add the mean of standard deviations of all batches to the lists below

        :param: y_predictions_all: Expect a tensor with shape [base_learner_num, x_num, y.shape] (y.shape should be 1 in the simplest experiment)
        :return: Mean of standard deviations of each predictions per batch
        """
        y_predictions_all_std = torch.std(
            y_predictions_all, dim=0)  # Expect shape: [x_num, y.shape]
        return torch.mean(y_predictions_all_std, dim=0, dtype=float).item()

    def experiment(self, model_configurations: dict, loss_function, data_loader_train, data_loader_test, epoch_num_per_log=20):
        """
        Run the experiment
        To use a customised loss function, create a Python function and pass the function instance to train_model()

        :param: loss_function: The loss function of models
        """
        # Hyperparameter combinations
        param_combinations = list(product(
            self.__correlation_penalty_coefficients,
            self.__ensemble_sizes,
            self.__epoch_nums,
            self.__learning_rates,
            self.__evolution_epoches,
            self.__max_fitnesses
        ))

        self.__df_experiment = pd.DataFrame(
            param_combinations, columns=self.__hyperparameters)

        # For storing experiment output data so that we can plot graphs
        average_batch_losses_train_neat_ncl = []  # Shape: [experiment_num, 1]
        average_batch_losses_test_neat_ncl = []  # Shape: [experiment_num, 1]
        average_batch_losses_train_static_ncl = []  # Shape: [experiment_num, 1]
        average_batch_losses_test_static_ncl = []  # Shape: [experiment_num, 1]
        """
        For comparison
        - Expect when lambda_ncl = 0, static-ncl-ensemble-model is the same as traditional-ensemble-model
        - (I think) Nothing special when lambda_ncl = 0.25, 0.5, 0.75
        - Expect when lambda_ncl = 1, static-ncl-ensemble-model is like a signle estimator
        """
        average_batch_losses_train_traditional = []  # Shape: [experiment_num, 1]
        average_batch_losses_test_traditional = []  # Shape: [experiment_num, 1]
        """
        For comparison
        """
        average_batch_losses_train_mlp = []  # Shape: [experiment_num, 1]
        average_batch_losses_test_mlp = []  # Shape: [experiment_num, 1]
        """
        For plotting a graph of loss against diversity coefficient
        """
        diversity_coefficients_train_neat_ncl = []  # Shape: [experiment_num, 1]
        diversity_coefficients_test_neat_ncl = []  # Shape: [experiment_num, 1]
        diversity_coefficients_train_static_ncl = []  # Shape: [experiment_num, 1]
        diversity_coefficients_test_static_ncl = []  # Shape: [experiment_num, 1]
        diversity_coefficients_train_traditional = []  # Shape: [experiment_num, 1]
        diversity_coefficients_test_traditional = []  # Shape: [experiment_num, 1]

        for row_index, row in self.__df_experiment.iterrows():
            correlation_penalty_coefficient = row["correlation_penalty_coefficient"].astype(
                float)
            ensemble_size = int(row["ensemble_size"])
            epoch_num = int(row["epoch_num"])
            learning_rate = float(row["learning_rate"])
            evolution_epoch = int(row["evolution_epoch"])
            max_fitness = float(row["max_fitness"])

            # Log
            print(f"============================================", flush=True)
            print(
                f"Experiment {row_index + 1} / {len(self.__df_experiment)} parameters: ", flush=True)
            print(
                f"correlation_penalty_coefficient: {correlation_penalty_coefficient}", flush=True)
            print(f"ensemble_size: {ensemble_size}", flush=True)
            print(f"epoch_num: {epoch_num}", flush=True)
            print(f"learning_rate: {learning_rate}", flush=True)
            print(f"evolution_epoch: {evolution_epoch}", flush=True)
            print(f"max_fitness: {max_fitness}", flush=True)

            # Instantiate NEAT NCL ensemble model
            ensemble_model_neat_ncl = self.build_neat_ncl_ensemble_net(
                config=self.__config,
                evolution_epoch=evolution_epoch,
                max_fitness=max_fitness,
                ensemble_size=ensemble_size
            )

            # Instantiate static NCL ensemble model
            ensemble_model_static_ncl = self.build_static_ncl_ensemble_net(
                model_configurations=model_configurations,
                ensemble_size=ensemble_size
            )

            # Instantiate traditional ensemble model
            ensemble_model_traditional = self.build_traditional_ensemble_net(
                model_configurations=model_configurations,
                ensemble_size=ensemble_size
            )

            # Instantiate MLP model
            mlp_model = self.build_mlp(
                model_configurations=model_configurations
            )
            mlp_model.to(self.__device)  # Move to GPU

            # Instantiate optimizer
            optimizer = torch.optim.Adam(
                mlp_model.parameters(), lr=learning_rate)

            # ======================================================
            # NEAT NCL ensemble model

            # Train NEAT NCL ensemble model
            print(f"Training NEAT NCL ensemble model in experiment ...", flush=True)
            ensemble_model_neat_ncl.train(
                data_loader_train=data_loader_train,
                epoch_num=epoch_num,
                learning_rate=learning_rate,
                loss_function=loss_function,
                lambda_ncl=correlation_penalty_coefficient,
                epoch_num_per_log=epoch_num_per_log
            )
            total_loss_train_neat_ncl, batch_num_train_neat_ncl, y_predictions_all_train_neat_ncl = self.evaluate_ensemble_model_neat_ncl(
                ensemble_model=ensemble_model_neat_ncl,
                loss_function=loss_function,
                data_loader=data_loader_train
            )

            average_batch_losses_train_neat_ncl.append(
                total_loss_train_neat_ncl / batch_num_train_neat_ncl)

            # Test NEAT NCL ensemble model
            print(f"Testing NEAT NCL ensemble model in experiment ...", flush=True)
            total_loss_test_neat_ncl, batch_num_test_neat_ncl, y_predictions_all_test_neat_ncl = self.evaluate_ensemble_model_neat_ncl(
                ensemble_model=ensemble_model_neat_ncl,
                loss_function=loss_function,
                data_loader=data_loader_test
            )

            average_batch_losses_test_neat_ncl.append(
                total_loss_test_neat_ncl / batch_num_test_neat_ncl)

            diversity_coefficients_test_neat_ncl.append(torch.mean(maths_helper.diversity_coefficient(
                y_predictions=y_predictions_all_test_neat_ncl), dim=0).item())

            diversity_coefficients_train_neat_ncl.append(torch.mean(maths_helper.diversity_coefficient(
                y_predictions=y_predictions_all_train_neat_ncl), dim=0).item())

            # ======================================================
            # Static NCL ensemble model

            # Train static NCL ensemble model
            print(f"Training static NCL ensemble model in experiment ...", flush=True)
            ensemble_model_static_ncl.train(
                data_loader_train=data_loader_train,
                epoch_num=epoch_num,
                learning_rate=learning_rate,
                loss_function=loss_function,
                lambda_ncl=correlation_penalty_coefficient,
                epoch_num_per_log=epoch_num_per_log
            )
            total_loss_train_static_ncl, batch_num_train_static_ncl, y_predictions_all_train_static_ncl = self.evaluate_ensemble_model_static_ncl(
                ensemble_model=ensemble_model_static_ncl,
                loss_function=loss_function,
                data_loader=data_loader_train,
                lambda_ncl=correlation_penalty_coefficient
            )

            average_batch_losses_train_static_ncl.append(
                total_loss_train_static_ncl / batch_num_train_static_ncl)

            # Test static NCL ensemble model
            print(f"Testing static NCL ensemble model in experiment ...", flush=True)
            total_loss_test_static_ncl, batch_num_test_static_ncl, y_predictions_all_test_static_ncl = self.evaluate_ensemble_model_static_ncl(
                ensemble_model=ensemble_model_static_ncl,
                loss_function=loss_function,
                data_loader=data_loader_test,
                lambda_ncl=correlation_penalty_coefficient
            )

            average_batch_losses_test_static_ncl.append(
                total_loss_test_static_ncl / batch_num_test_static_ncl)

            diversity_coefficients_test_static_ncl.append(torch.mean(maths_helper.diversity_coefficient(
                y_predictions=y_predictions_all_test_static_ncl), dim=0).item())

            diversity_coefficients_train_static_ncl.append(torch.mean(maths_helper.diversity_coefficient(
                y_predictions=y_predictions_all_train_static_ncl), dim=0).item())

            # ======================================================
            # Traditional ensemble model

            # Train traditional ensemble model
            print(f"Training traditional ensemble model in experiment ...", flush=True)
            ensemble_model_traditional.train(
                data_loader_train=data_loader_train,
                epoch_num=epoch_num,
                learning_rate=learning_rate,
                loss_function=loss_function,
                epoch_num_per_log=epoch_num_per_log
            )
            total_loss_train_traditional, batch_num_train_traditional, y_predictions_all_train_traditional = self.evaluate_ensemble_model_traditional(
                ensemble_model=ensemble_model_traditional,
                loss_function=loss_function,
                data_loader=data_loader_train
            )

            average_batch_losses_train_traditional.append(
                total_loss_train_traditional / batch_num_train_traditional)

            # Test traditional ensemble model
            print(f"Testing traditional ensemble model in experiment ...", flush=True)
            total_loss_test_traditional, batch_num_test_traditional, y_predictions_all_test_traditional = self.evaluate_ensemble_model_traditional(
                ensemble_model=ensemble_model_traditional,
                loss_function=loss_function,
                data_loader=data_loader_test
            )

            average_batch_losses_test_traditional.append(
                total_loss_test_traditional / batch_num_test_traditional)

            diversity_coefficients_test_traditional.append(torch.mean(maths_helper.diversity_coefficient(
                y_predictions=y_predictions_all_test_traditional), dim=0).item())

            diversity_coefficients_train_traditional.append(torch.mean(maths_helper.diversity_coefficient(
                y_predictions=y_predictions_all_train_traditional), dim=0).item())

            # ======================================================
            # MLP model

            # Train + test MLP model
            # There is a more efficient way, which is to pick the largest epoch and evaluate model for each target training epoch (reduce number of times to train models as we are training the model once assuming other hyperparameters stay the same except epoch_num)
            # However, that will ruin the consistency of this code so I did not use it
            # It is a trade-off between code consistency and execution time
            print(f"Training and testing MLP model in experiment ...", flush=True)
            for epoch in range(epoch_num):
                total_loss_train_mlp = 0

                # Training
                mlp_model.train()
                for (batch_X, batch_y) in data_loader_train:
                    batch_X = batch_X.to(self.__device)
                    batch_y = batch_y.to(self.__device)

                    # y_prediction is a PyTorch tensor
                    y_prediction = mlp_model(batch_X)

                    loss = loss_function(y_prediction, batch_y)

                    # Backward pass and optimization
                    optimizer.zero_grad()  # reset gradients
                    loss.backward()        # compute gradients
                    optimizer.step()       # update weights

                    total_loss_train_mlp += loss.item()

                average_batch_loss_train_mlp = total_loss_train_mlp / \
                    len(data_loader_train)
                if epoch % epoch_num_per_log == 0:
                    print(
                        f"Epoch: {epoch + 1} / {epoch_num}; Loss: {average_batch_loss_train_mlp:.4f}", flush=True)

                if epoch == epoch_num - 1:
                    # Last epoch
                    average_batch_losses_train_mlp.append(
                        average_batch_loss_train_mlp)

            # ============================================================================================

            # Testing
            total_loss_test_mlp, batch_num_test_mlp = self.__mlp_model_helper.test_model(
                model=mlp_model,
                loss_function=loss_function,
                data_loader_test=data_loader_test
            )
            average_batch_loss_test_mlp = total_loss_test_mlp / batch_num_test_mlp
            average_batch_losses_test_mlp.append(average_batch_loss_test_mlp)

        # Add columns of average_loss to the dataframe
        self.__df_experiment["average_loss_train_neat_ncl"] = average_batch_losses_train_neat_ncl
        self.__df_experiment["average_loss_test_neat_ncl"] = average_batch_losses_test_neat_ncl
        self.__df_experiment["average_loss_train_static_ncl"] = average_batch_losses_train_static_ncl
        self.__df_experiment["average_loss_test_static_ncl"] = average_batch_losses_test_static_ncl
        self.__df_experiment["average_loss_train_traditional"] = average_batch_losses_train_traditional
        self.__df_experiment["average_loss_test_traditional"] = average_batch_losses_test_traditional
        self.__df_experiment["average_loss_train_mlp"] = average_batch_losses_train_mlp
        self.__df_experiment["average_loss_test_mlp"] = average_batch_losses_test_mlp
        self.__df_experiment["diversity_coefficient_train_neat_ncl"] = diversity_coefficients_train_neat_ncl
        self.__df_experiment["diversity_coefficient_test_neat_ncl"] = diversity_coefficients_test_neat_ncl
        self.__df_experiment["diversity_coefficient_train_static_ncl"] = diversity_coefficients_train_static_ncl
        self.__df_experiment["diversity_coefficient_test_static_ncl"] = diversity_coefficients_test_static_ncl
        self.__df_experiment["diversity_coefficient_train_traditional"] = diversity_coefficients_train_traditional
        self.__df_experiment["diversity_coefficient_test_traditional"] = diversity_coefficients_test_traditional

    def save_csv(self, repository_experiment_csv_path: str, csv_name: str) -> None:
        assert repository_experiment_csv_path and csv_name, f"Invalid final csv path: {repository_experiment_csv_path}/{csv_name}.csv"
        self.__df_experiment.to_csv(
            f"{repository_experiment_csv_path}/{csv_name}.csv", index=False)

    def plot_graph(self, repository_experiment_img_path: str, img_name: str, save_img=True, image_resolution=300, show_img=False) -> None:
        # Set plot style for nicer plots
        sns.set_theme(style="whitegrid", rc={
            "font.size": FONT_SIZE,  # Default font size
            "font.family": FONT_FAMILY,
            "font.serif": FONT_SERIF,
            "axes.titlesize": AXES_TITLESIZE,
            "axes.titleweight": AXES_TITLEWEIGHT,
            "axes.labelsize": AXES_LABELSIZE,
            "legend.fontsize": LEGEND_FONTSIZE
        })

        plt.figure(figsize=(16, 24))  # Larger figure for 8 plots

        row_num = len(self.__hyperparameters)
        column_num = 3

        for i, hyperparameter in enumerate(self.__hyperparameters, 1):
            column_index = 0  # 0-indexed

            # Average loss
            plt.subplot(row_num, column_num, (i - 1) *
                        column_num + (column_index + 1))
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_train_neat_ncl",
                marker='o',
                label=f"Train - NEAT NCL ensemble model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_test_neat_ncl",
                marker='o',
                label=f"Test - NEAT NCL ensemble model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_train_static_ncl",
                marker='o',
                label=f"Train - static NCL ensemble model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_test_static_ncl",
                marker='o',
                label=f"Test - static NCL ensemble model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_train_traditional",
                marker='o',
                label=f"Train - traditional ensemble model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_test_traditional",
                marker='o',
                label=f"Test - traditional ensemble model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_train_mlp",
                marker='o',
                label=f"Train - MLP model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="average_loss_test_mlp",
                marker='o',
                label=f"Test - MLP model"
            )
            plt.title(f"MSE vs {hyperparameter}")
            plt.xlabel(hyperparameter)
            plt.ylabel("MSE")
            plt.legend()

            # ===================================================================

            # Diversity Coefficient
            column_index = 1  # 0-indexed

            plt.subplot(row_num, column_num, (i - 1) *
                        column_num + (column_index + 1))
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="diversity_coefficient_train_neat_ncl",
                marker='o',
                label=f"Train - NEAT NCL ensemble model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="diversity_coefficient_test_neat_ncl",
                marker='o',
                label=f"Test - NEAT NCL ensemble model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="diversity_coefficient_train_static_ncl",
                marker='o',
                label=f"Train - static NCL ensemble model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="diversity_coefficient_test_static_ncl",
                marker='o',
                label=f"Test - static NCL ensemble model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="diversity_coefficient_train_traditional",
                marker='o',
                label=f"Train - traditional ensemble model"
            )
            sns.lineplot(
                data=self.__df_experiment,
                x=hyperparameter,
                y="diversity_coefficient_test_traditional",
                marker='o',
                label=f"Test - traditional ensemble model"
            )
            plt.title(
                f"Diversity Coefficient vs {hyperparameter}")
            plt.xlabel(hyperparameter)
            plt.ylabel("Diversity Coefficient")
            plt.legend()

            # ===================================================================

            # Loss against diversity coefficient
            column_index = 2  # 0-indexed

            if i == 1:
                # We only want to plot 1 graph for all hyparameters for this graph
                # Remember i starts with 1 instead of 0

                plt.subplot(row_num, column_num, (i - 1) *
                            column_num + (column_index + 1))
                sns.lineplot(
                    data=self.__df_experiment,
                    x=f"diversity_coefficient_train_neat_ncl",
                    y="average_loss_train_neat_ncl",
                    marker='o',
                    label=f"Train - NEAT NCL ensemble model"
                )
                sns.lineplot(
                    data=self.__df_experiment,
                    x=f"diversity_coefficient_test_neat_ncl",
                    y="average_loss_test_neat_ncl",
                    marker='o',
                    label=f"Test - NEAT NCL ensemble model"
                )
                sns.lineplot(
                    data=self.__df_experiment,
                    x=f"diversity_coefficient_train_static_ncl",
                    y="average_loss_train_static_ncl",
                    marker='o',
                    label=f"Train - static NCL ensemble model"
                )
                sns.lineplot(
                    data=self.__df_experiment,
                    x=f"diversity_coefficient_test_static_ncl",
                    y="average_loss_test_static_ncl",
                    marker='o',
                    label=f"Test - static NCL ensemble model"
                )
                sns.lineplot(
                    data=self.__df_experiment,
                    x=f"diversity_coefficient_train_traditional",
                    y="average_loss_train_traditional",
                    marker='o',
                    label=f"Train - traditional ensemble model"
                )
                sns.lineplot(
                    data=self.__df_experiment,
                    x=f"diversity_coefficient_test_traditional",
                    y="average_loss_test_traditional",
                    marker='o',
                    label=f"Test - traditional ensemble model"
                )
                plt.title(
                    f"MSE vs Diversity Coefficient")
                plt.xlabel(f"Diversity coefficient")
                plt.ylabel("MSE")
                plt.legend()

        plt.tight_layout()

        if save_img:
            assert repository_experiment_img_path and img_name, f"Invalid final image path: {repository_experiment_img_path}/{img_name}.png"

            # Save graph as image if applicable
            plt.savefig(f"{repository_experiment_img_path}/{img_name}.png",
                        dpi=image_resolution)

        if show_img:
            # Show graph if applicable
            # Might block the execution so we set show_img = False for non-interactive Python script
            # Jupyter notebook will still show the image if show_img = False though
            plt.show()

    @property
    def device(self):
        return self.__device

    @property
    def df_experiment(self):
        return self.__df_experiment

    @property
    def config(self) -> neat.Config:
        return self.__config
