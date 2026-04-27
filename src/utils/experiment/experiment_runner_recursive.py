from src.model.recursive_ensemble_net import RecursiveEnsembleNet
from src.utils import logger
from src.utils.experiment.experiment_runner import ExperimentRunner
import torch


class ExperimentRunnerRecursive(ExperimentRunner):
    def __init__(self, configurations=None, **experiment_parameters):
        super().__init__(configurations, **experiment_parameters)

    def build_model(self, **experiment_parameters) -> None:
        # Instantiate recursive ensemble net
        try:
            self.model = RecursiveEnsembleNet(
                architecture=experiment_parameters["architecture"]
            )
        except Exception as e:
            raise Exception(f"Error in building recursive model: {str(e)}")

    def run(self, data_loader_train: torch.utils.data.DataLoader, data_loader_validation: torch.utils.data.DataLoader, data_loader_test: torch.utils.data.DataLoader) -> None:
        assert data_loader_validation is not None, f"Missing required variable for experiment (expect not None): data_loader_validation"

        # Get experiment configurations
        try:
            epoch_num_per_log = self.configurations["epoch_num_per_log"]
            assert epoch_num_per_log is not None, f"Missing required configuration argument (expect not None): epoch_num_per_log"

            loss_function = self.configurations["loss_function"]
            assert loss_function is not None, f"Missing required configuration argument (expect not None): loss_function"

            device = self.configurations["device"]
            assert device is not None, f"Missing required configuration argument (expect not None): device"

            learning_rate = self.configurations["learning_rate"]
            assert learning_rate is not None, f"Missing required configuration argument (expect not None): learning_rate"

            model_repository_path = self.configurations["model_repository_path"]
            assert model_repository_path is not None, f"Missing required configuration argument (expect not None): model_repository_path"

            # Only will be used by static NCL
            correlation_penalty_coefficient = self.configurations["correlation_penalty_coefficient"]
            assert correlation_penalty_coefficient is not None, f"Missing required configuration argument (expect not None): correlation_penalty_coefficient"

            # Different from evolution_epoch in NEAT NCL
            train_epoch = self.configurations["train_epoch"]
            assert train_epoch is not None, f"Missing required configuration argument (expect not None): train_epoch"

            # Only for NEAT NCL
            min_correlation_penalty_coefficient = self.configurations[
                "min_correlation_penalty_coefficient"]
            assert min_correlation_penalty_coefficient is not None, f"Missing required configuration argument (expect not None): min_correlation_penalty_coefficient"

            # Only for NEAT NCL
            max_correlation_penalty_coefficient = self.configurations[
                "max_correlation_penalty_coefficient"]
            assert max_correlation_penalty_coefficient is not None, f"Missing required configuration argument (expect not None): max_correlation_penalty_coefficient"

            # NOTE: Technically, we can check min_correlation_penalty_coefficient < max_correlation_penalty_coefficient

            # Only for NEAT NCL
            evolution_epoch = self.configurations["evolution_epoch"]
            assert evolution_epoch is not None, f"Missing required configuration argument (expect not None): evolution_epoch"

            # Only for NEAT NCL
            # NOTE: Might not be used
            evolution_epoch_step_size = self.configurations["evolution_epoch_step_size"]
            assert evolution_epoch_step_size is not None, f"Missing required configuration argument (expect not None): evolution_epoch_step_size"

            # Only for NEAT NCL
            population_size = self.configurations["population_size"]
            assert population_size is not None, f"Missing required configuration argument (expect not None): population_size"
        except Exception as e:
            raise Exception(f"Error in getting configurations: {str(e)}")

        # For NEAT NCL
        self.configurations["config"].pop_size = population_size

        # Target experiment output data
        losses_test = []  # Shape: [experiment_num]
        # NOTE: We do not care about training loss

        for row_index, row in self.df_experiment.iterrows():
            try:
                # Get experiment parameters
                architecture = dict(row["architecture"])

                self.build_model(architecture=architecture)

                assert self.model is not None, f"Missing required variable for experiment (expect not None): self.model -> invoke build_model()"

                # NOTE: we use recursive_to(...) here instead of to(...)
                self.model.recursive_to(device=device)

                # Log
                logger.log()
                logger.log(
                    f"Experiment {row_index + 1} / {len(self.df_experiment)} parameters: ")
                logger.log(
                    f"architecture: {architecture}")

                # Train model

                self.model.train()

                if ExperimentRunnerRecursive.is_architecture_neat_ncl_related(architecture):
                    # NEAT NCL
                    # This section extracts the 1 batch of training and testing data
                    # Specifically required for NEAT NCL because the NEAT NCL model does not expect a tensor batch
                    data_train = None
                    assert len(
                        data_loader_train) == 1, f"Invalid number of training batches (expect 1 because all training data should be in the same batch): {len(data_loader_train)}"
                    for data_train_X, data_train_y in data_loader_train:
                        data_train = (data_train_X, data_train_y)
                        break

                    data_test = None
                    assert len(
                        data_loader_test) == 1, f"Invalid number of testing batches (expect 1 because all testing data should be in the same batch): {len(data_loader_test)}"
                    for data_test_X, data_test_y in data_loader_test:
                        data_test = (data_test_X, data_test_y)
                        break

                    # Train + test model by evoling the model
                    fitness_function_arguments = {
                        "model_repository_path": model_repository_path,
                        "data_train": data_train,
                        "loss_function": loss_function,
                        "min_correlation_penalty_coefficient": min_correlation_penalty_coefficient,
                        "max_correlation_penalty_coefficient": max_correlation_penalty_coefficient,
                        "learning_rate": learning_rate,
                        "is_experiment": True,
                        "data_test": data_test
                    }

                    # NOTE: Expect data loader instead of tensor of data batch
                    self.model.recursive_train(
                        X=data_train[0],
                        y=data_train[1],
                        learning_rate=learning_rate,
                        correlation_penalty_coefficient=correlation_penalty_coefficient,
                        evolution_epoch=evolution_epoch,
                        fitness_function_arguments=fitness_function_arguments
                    )
                else:
                    # Assume to be traditional/(static NCL)
                    for epoch in range(train_epoch):
                        for (batch_X_train, batch_y_train) in data_loader_train:
                            batch_X_train = batch_X_train.to(device)
                            batch_y_train = batch_y_train.to(device)

                            self.model.recursive_train(
                                X=batch_X_train,
                                y=batch_y_train,
                                learning_rate=learning_rate,
                                correlation_penalty_coefficient=correlation_penalty_coefficient
                            )

                # Test model

                self.model.eval()

                batch_loss_sum_test = 0

                with torch.no_grad():
                    for (batch_X_test, batch_y_test) in data_loader_test:
                        batch_X_test = batch_X_test.to(device)
                        batch_y_test = batch_y_test.to(device)

                        # Shape: [base_learner_num, batch_size, 1]
                        y_prediction_ensemble = self.model(
                            batch_X_test).detach()

                        batch_loss_test = loss_function(
                            y_prediction_ensemble, batch_y_test)

                        batch_loss_sum_test += batch_loss_test.cpu().item()

                # The experiment output data we are interested in
                losses_test.append(batch_loss_sum_test / len(data_loader_test))
            except Exception as e:
                logger.error(f"Error encountered: {e}")
                logger.error(f"Skipping experiment ...")
                logger.error()

                raise e

        self.df_experiment["loss_test"] = losses_test

    @classmethod
    def is_architecture_neat_ncl_related(cls, architecture: dict) -> bool:
        """
        This method is created NEAT NCL expects a dataloader instead of a tensor of data batch
        This method will be incorrect once each recursive ensemble network has more than 1 type of base learner
            - But this will never in this stage of project
        """
        if "neat_ncl" in architecture:
            return True
        else:
            return False
