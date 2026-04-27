import torch
from src.model.mlp import MLP
from src.voter.voter import Voter
from src.voter.voter_arithmetic_mean import ArithmeticMeanVoter
from src.voter.voter_median import MedianVoter
from src.voter.voter_nn import NNVoter

"""
An ensemble model for stage 6
Replacing traditional ensemble network
Distributed computing is not used here. Instead a simple for-loop is used to simplify the design
"""


class RecursiveTraditionalEnsembleNet:
    def __init__(self, ensemble_size: int, model_configurations: dict, model_repository_path="data/model", voter="arithmetic_mean"):
        """
        :param voter: The type of voter used
            - arithmetic_mean (default)
            - median
            - nn
        """
        assert ensemble_size > 1, f"Invalid ensemble_size (expect > 1): {ensemble_size}"
        self.__ensemble_size = ensemble_size
        self.__model_repository_path = model_repository_path

        self.__base_learners = []
        for _ in range(self.__ensemble_size):
            self.__base_learners.append(MLP.build_from_config(
                input_size=model_configurations[f"input_size"],
                hidden_sizes=model_configurations[f"hidden_sizes"],
                activations=model_configurations[f"activations"],
                activation_type=model_configurations[f"activation_type"],
                output_size=model_configurations[f"output_size"],
                dropout_rate=model_configurations[f"dropout_rate"],
                dropout_indices=model_configurations[f"dropout_indices"]
            ))

        if voter == "arithmetic_mean":
            self.__voter = ArithmeticMeanVoter()
        elif voter == "median":
            self.__voter = MedianVoter()
        elif voter == "nn":
            self.__voter = NNVoter(ensemble_size=self.__ensemble_size)
        else:
            raise Exception(f"Invalid voter: {voter}")

    def train(self, mode: bool = True):
        for base_learner in self.__base_learners:
            base_learner.train(mode=mode)

        if isinstance(self.__voter, NNVoter):
            self.__voter.train(mode=mode)

    def eval(self, mode: bool = True):
        for base_learner in self.__base_learners:
            base_learner.eval(mode=mode)

        if isinstance(self.__voter, NNVoter):
            self.__voter.eval(mode=mode)

    def to(self, device: any):
        for base_learner in self.__base_learners:
            base_learner.to(device=device)

        if isinstance(self.__voter, NNVoter):
            self.__voter.to(device=device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :return: The output tensor (dimension of the output tensor depends on the problem type)
        """
        # Shape: [base_learner_num, batch_size, 1]
        base_learner_outputs = []

        for base_learner in self.__base_learners:
            base_learner_outputs.append(base_learner(x))

        return self.__voter.vote(y_predictions=torch.stack(base_learner_outputs, dim=0))

    @property
    def model_repository_path(self) -> str:
        return self.__model_repository_path

    @property
    def ensemble_size(self) -> int:
        return self.__ensemble_size

    @property
    def base_learners(self) -> list:
        return self.__base_learners

    @property
    def voter(self) -> Voter:
        return self.__voter
