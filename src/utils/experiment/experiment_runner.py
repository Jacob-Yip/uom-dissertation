"""
Run the experiment of 1 model and generate the dataframe of the experiment outputs, e.g. average training/testing loss
An abstract parent class to be implemented
"""

from abc import ABC, abstractmethod
from itertools import product
import os
import pandas as pd
import torch


class ExperimentRunner(ABC):
    """
    The dictionary of configurations for everything, e.g. model, experiment
        - Fixed learning rate
        - Fixed number of hidden layers
    In the form of {configuration_name: configuration_value}
    """
    configurations: dict
    """
    The dictionary of experiment parameters
    In the form of {parameter_name: [parameter_value]}
    """
    parameters: dict
    """
    List of experiment parameter names
    """
    parameter_names: list
    df_experiment: pd.DataFrame
    model: any

    def __init__(self, configurations=None, **experiment_parameters):
        super().__init__()

        self.configurations = configurations
        self.parameters = experiment_parameters
        self.parameter_names = list(experiment_parameters.keys())

        parameter_combinations = list(product(*experiment_parameters.values()))
        self.df_experiment = pd.DataFrame(
            parameter_combinations,
            columns=self.parameter_names
        )

        self.model = None

    @abstractmethod
    def build_model(self, **experiment_parameters) -> None:
        """
        Create a model instance for 1 experiment of a particular combination of experiment parameters
        Update self.model

        :param experiment_parameters: Any necessary additional experiment parameters for building the model
        """
        pass

    @abstractmethod
    def run(self, data_loader_train: torch.utils.data.DataLoader, data_loader_validation: torch.utils.data.DataLoader, data_loader_test: torch.utils.data.DataLoader) -> None:
        """
        Update self.df_experiment

        :param data_loader_train: Data loader of training data
        :param data_loader_validation: Data loader of validation data
        :param data_loader_test: Data loader of testing data
        """
        pass

    def save_csv(self, absolute_folder_path: str, csv_file_name: str) -> str:
        """
        Save self.df_experiment to a csv file

        :param absolute_folder_path: Absolute path to the folder of the csv file
        :param csv_file_name: Name of the csv file to be created and saved (without file extension)
        :return: The absolute path to the csv file
        :rtype: str
        """
        assert absolute_folder_path and csv_file_name, f"Invalid final csv path: {absolute_folder_path}/{csv_file_name}.csv"

        absolute_file_path = os.path.join(
            absolute_folder_path, f"{csv_file_name}.csv")

        assert self.df_experiment is not None, f"Invalid required variable (expect not None): self.df_experiment"
        self.df_experiment.to_csv(absolute_file_path, index=False)

        return absolute_file_path
