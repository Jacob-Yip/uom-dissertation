import pandas as pd
from src.model.neat_ncl_ensemble_net import NEATNCLEnsembleNet
from src.utils import logger, maths_helper
from src.utils.experiment.experiment_runner import ExperimentRunner
import torch


class ExperimentRunnerNEATNCL(ExperimentRunner):
    def __init__(self, configurations=None, **experiment_parameters):
        super().__init__(configurations, **experiment_parameters)

    def build_model(self, **experiment_parameters) -> None:
        # Instantiate NEAT NCL ensemble net
        try:
            self.model = NEATNCLEnsembleNet(
                config=self.configurations["config"],
                model_repository_path=self.configurations["model_repository_path"], 
                voter=self.configurations["voter"]
            )
        except Exception as e:
            raise Exception(f"Error in building NEAT NCL model: {str(e)}")

    def run(self, data_loader_train: torch.utils.data.DataLoader, data_loader_validation: torch.utils.data.DataLoader, data_loader_test: torch.utils.data.DataLoader) -> None:
        assert data_loader_validation is None, f"Expect data_loader_validation to be None (not applicable to this experiment): {data_loader_validation}"

        # Get experiment configurations
        try:
            loss_function = self.configurations["loss_function"]
            assert loss_function is not None, f"Missing required configuration argument (expect not None): loss_function"

            min_correlation_penalty_coefficient = self.configurations[
                "min_correlation_penalty_coefficient"]
            assert min_correlation_penalty_coefficient is not None, f"Missing required configuration argument (expect not None): min_correlation_penalty_coefficient"

            max_correlation_penalty_coefficient = self.configurations[
                "max_correlation_penalty_coefficient"]
            assert max_correlation_penalty_coefficient is not None, f"Missing required configuration argument (expect not None): max_correlation_penalty_coefficient"

            # NOTE: Technically, we can check min_correlation_penalty_coefficient <= max_correlation_penalty_coefficient here but my hunch says no for now

            # NOTE: Technically, this is an experiment parameter but because of how .evolve() is implemented, we only need the maximum value of evolution_epoch to get all relevant data
            evolution_epoch = self.configurations["evolution_epoch"]
            assert evolution_epoch is not None, f"Missing required configuration argument (expect not None): evolution_epoch"

            # Number of evolution epoches for 1 output data to be stored
            evolution_epoch_step_size = self.configurations["evolution_epoch_step_size"]
            assert evolution_epoch_step_size is not None, f"Missing required configuration argument (expect not None): evolution_epoch_step_size"
        except Exception as e:
            raise Exception(f"Error in getting configurations: {str(e)}")

        # Target experiment output data
        losses_test = []  # Shape: [experiment_num * generation_num]
        # NOTE: We do not care about training loss
        average_active_hidden_node_nums = []  # Shape: [experiment_num * generation_num]
        diversity_coefficients_test = []  # Shape: [experiment_num * generation_num]
        population_diversities_test = []  # Shape: [experiment_num * generation_num]
        max_population_diversities_test = []  # Shape: [experiment_num * generation_num]
        correlation_penalty_coefficients_test = []  # Shape: [experiment_num * generation_num]
        subpopulation_nums_test = []  # Shape: [experiment_num * generation_num]
        average_niche_radiuses_test = []  # Shape: [experiment_num * generation_num]
        average_sharing_factors_test = []  # Shape: [experiment_num * generation_num]
        average_raw_fitnesses_test = []  # Shape: [experiment_num * generation_num]
        average_adjusted_fitnesses_test = []  # Shape: [experiment_num * generation_num]
        """
        These variables are created because we need to expand the original dataframe
        Because of evolution_epoch, each row of the original dataframe (having parameters ensemble_size and learning_rate only) needs to be expanded to n rows for the new dataframe (having parameters ensemble_size, learning_rate and generation_index)
        """
        generation_indices = []  # Shape: [experiment_num * generation_num]
        ensemble_sizes = []  # Shape: [experiment_num * generation_num]
        learning_rates = []  # Shape: [experiment_num * generation_num]

        for row_index, row in self.df_experiment.iterrows():
            try:
                # Get experiment parameters
                ensemble_size = int(row["ensemble_size"])
                learning_rate = float(row["learning_rate"])

                # Update the config instance based on ensemble_size
                # ensemble_size should be equal to population_size in the configuration file
                self.configurations["config"].pop_size = ensemble_size

                # NOTE: Maybe the config instance should be passed as kwargs instead of instance variable
                self.build_model()

                assert self.model is not None, f"Missing required variable for experiment (expect not None): self.model -> invoke build_model()"

                # Log
                logger.log()
                logger.log(
                    f"Experiment {row_index + 1} / {len(self.df_experiment)} parameters: ")
                logger.log(f"ensemble_size: {ensemble_size}")
                logger.log(f"learning_rate: {learning_rate}")

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
                    "model_repository_path": self.configurations["model_repository_path"],
                    "data_train": data_train,
                    "loss_function": loss_function,
                    "min_correlation_penalty_coefficient": min_correlation_penalty_coefficient,
                    "max_correlation_penalty_coefficient": max_correlation_penalty_coefficient,
                    "learning_rate": learning_rate,
                    "is_experiment": True,
                    "data_test": data_test
                }
                self.model.evolve(
                    evolution_epoch=evolution_epoch,
                    fitness_function_arguments=fitness_function_arguments
                )

                # Collect raw experiment data in the form of {generation_index: {output_data_name: output_data_value}}
                all_experiment_data = self.model.experiment_data

                # NOTE: We will take into account evolution_epoch_step_size to reduce the number of output data generated
                for generation_index, experiment_data in all_experiment_data.items():
                    # NOTE: We have assumed that generation_index goes like 0, 1, 2, ..., n
                    if generation_index % evolution_epoch_step_size != 0:
                        # We skip this set of experiment data in this generation
                        continue

                    # Actual experiment output data we get from NEAT NCL
                    losses_test.append(experiment_data["ensemble_loss"])
                    average_active_hidden_node_nums.append(
                        experiment_data["average_active_hidden_node_num"])
                    diversity_coefficients_test.append(
                        experiment_data["diversity_coefficient"])
                    population_diversities_test.append(
                        experiment_data["population_diversity"])
                    max_population_diversities_test.append(
                        experiment_data["max_population_diversity"])
                    correlation_penalty_coefficients_test.append(
                        experiment_data["correlation_penalty_coefficient"])
                    subpopulation_nums_test.append(experiment_data["subpopulation_num"])
                    """
                    Calculate average of list of target experiment data

                    Experiment data in the form of a list: 
                    - niche_radiuses
                    - sharing_factors
                    - raw_fitnesses
                    - adjusted_fitnesses
                    """
                    average_niche_radiuses_test.append(maths_helper.average(data=experiment_data["niche_radiuses"]))
                    average_sharing_factors_test.append(maths_helper.average(data=experiment_data["sharing_factors"]))
                    average_raw_fitnesses_test.append(maths_helper.average(data=experiment_data["raw_fitnesses"]))
                    average_adjusted_fitnesses_test.append(maths_helper.average(data=experiment_data["adjusted_fitnesses"]))
                    # For updating the original dataframe
                    generation_indices.append(generation_index)
                    ensemble_sizes.append(ensemble_size)
                    learning_rates.append(learning_rate)
            except Exception as e:
                logger.error(f"Error encountered: {e}")
                logger.error(f"Skipping experiment ...")
                logger.error()

        # Reset dataframe
        # Otherwise, adding experiment output data will throw errors
        self.df_experiment = pd.DataFrame()

        # For updating the original dataframe
        self.df_experiment["generation_index"] = generation_indices
        self.df_experiment["ensemble_size"] = ensemble_sizes
        self.df_experiment["learning_rate"] = learning_rates
        # Actual experiment output data we get from NEAT NCL
        self.df_experiment["loss_test"] = losses_test
        self.df_experiment["average_active_hidden_node_num"] = average_active_hidden_node_nums
        self.df_experiment["diversity_coefficient_test"] = diversity_coefficients_test
        self.df_experiment["population_diversity_test"] = population_diversities_test
        self.df_experiment["max_population_diversity_test"] = max_population_diversities_test
        self.df_experiment["correlation_penalty_coefficient_test"] = correlation_penalty_coefficients_test
        self.df_experiment["subpopulation_num"] = subpopulation_nums_test
        self.df_experiment["average_niche_radius"] = average_niche_radiuses_test
        self.df_experiment["average_sharing_factor"] = average_sharing_factors_test
        self.df_experiment["average_raw_fitness"] = average_raw_fitnesses_test
        self.df_experiment["average_adjusted_fitness"] = average_adjusted_fitnesses_test
