from src.model.traditional_ensemble_net import TraditionalEnsembleNet
from src.utils import logger, maths_helper
from src.utils.experiment.experiment_runner import ExperimentRunner
import torch


class ExperimentRunnerTraditional(ExperimentRunner):
    def __init__(self, configurations=None, **experiment_parameters):
        super().__init__(configurations, **experiment_parameters)

    def build_model(self, **experiment_parameters) -> None:
        # Instantiate traditional ensemble net
        try:
            model_configurations = {
                "input_size": self.configurations["input_size"],
                "hidden_sizes": experiment_parameters["hidden_size"],
                "activations": self.configurations["activations"],
                "activation_type": self.configurations["activation_type"],
                "output_size": self.configurations["output_size"],
                "dropout_rate": self.configurations["dropout_rate"],
                "dropout_indices": self.configurations["dropout_indices"]
            }

            self.model = TraditionalEnsembleNet(
                device_num=self.configurations["device_num"],
                base_learner_nums=[
                    experiment_parameters["ensemble_size"]] * self.configurations["device_num"],
                model_configurations=model_configurations,
                dist_backend=self.configurations["dist_backend"],
                master_address=self.configurations["master_address"],
                master_port=self.configurations["master_port"],
                model_repository_path=self.configurations["model_repository_path"], 
                voter=self.configurations["voter"]
            )
        except Exception as e:
            raise Exception(
                f"Error in building Traditional model: {str(e)}")

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
        # NOTE: We do not care about training loss
        diversity_coefficients_test = []  # Shape: [experiment_num]
        # Average base learner testing loss
        losses_test_base_learner = []  # Shape: [experiment_num]

        for row_index, row in self.df_experiment.iterrows():
            # Get experiment parameters
            ensemble_size = int(row["ensemble_size"])
            epoch_num = int(row["epoch_num"])
            learning_rate = float(row["learning_rate"])
            hidden_size = row["hidden_size"]

            self.build_model(ensemble_size=ensemble_size,
                             hidden_size=hidden_size)

            assert self.model is not None, f"Missing required variable for experiment (expect not None): self.model -> invoke build_model()"

            # Log
            logger.log()
            logger.log(
                f"Experiment {row_index + 1} / {len(self.df_experiment)} parameters: ")
            logger.log(f"ensemble_size: {ensemble_size}")
            logger.log(f"epoch_num: {epoch_num}")
            logger.log(f"learning_rate: {learning_rate}")
            logger.log(f"hidden_size: {hidden_size}")

            # Train model

            self.model.train(
                data_loader_train=data_loader_train,
                epoch_num=epoch_num,
                learning_rate=learning_rate,
                loss_function=loss_function,
                epoch_num_per_log=epoch_num_per_log
            )

            # Test model
            # model.eval() and torch.no_grad() will be done TraditionalEnsembleNet

            batch_loss_sum_test = 0
            y_predictions_all_raw = []
            # Loss of all base learners of each batch
            losses_base_learner_raw = []

            for (batch_X_test, batch_y_test) in data_loader_test:
                # Calcualte ensemble testing loss
                # Shape: [base_learner_num, batch_size, 1]
                y_predictions_all_batch = self.model.evaluate_all(batch_X_test)

                y_predictions_all_raw.append(y_predictions_all_batch)

                y_prediction = self.model.vote(y_predictions_all_batch)

                batch_loss_test = loss_function(y_prediction, batch_y_test)

                batch_loss_sum_test += batch_loss_test.item()

                # Calculate average base learner testing loss
                # TODO: Refactor code so that this is not hard-coded (for the current project, this seems to be good enough)
                # Loss function for 1 base learner
                # Give a tensor instead of a scalar as the final output
                base_learner_loss_function = torch.nn.MSELoss(reduction="none")

                # Shape: [base_learner_num, batch_num, 1]
                losses_base_learner_batch_all_data = base_learner_loss_function(
                    y_predictions_all_batch, batch_y_test.unsqueeze(0).expand_as(y_predictions_all_batch))

                # Shape: [base_learner_num]
                losses_base_learner = losses_base_learner_batch_all_data.mean(
                    dim=tuple(range(1, losses_base_learner_batch_all_data.dim())))

                losses_base_learner_raw.append(losses_base_learner)

            # Shape: [base_learner_num, x_num, 1]
            y_predictions_all = torch.cat(y_predictions_all_raw, dim=1)

            # Shape: [base_learner_num]
            # Average loss of each base learners of all batches
            average_losses_base_learner = torch.stack(
                losses_base_learner_raw, dim=1).mean(dim=1)

            # The experiment output data we are interested in
            losses_test.append(batch_loss_sum_test / len(data_loader_test))
            diversity_coefficients_test.append(torch.mean(
                maths_helper.diversity_coefficient(y_predictions=y_predictions_all), dim=0).item())
            losses_test_base_learner.append(
                average_losses_base_learner.mean(dim=0).item())

        self.df_experiment["loss_test"] = losses_test
        self.df_experiment["diversity_coefficient_test"] = diversity_coefficients_test
        self.df_experiment["loss_test_base_learner"] = losses_test_base_learner
