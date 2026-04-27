from src.model.mlp import MLP
from src.voter.voter import Voter
from torch import nn
import torch


class NNVoter(Voter):
    def __init__(self, ensemble_size: int, model_configurations=None):
        super().__init__()

        assert ensemble_size > 0, f"Invalid ensemble_size (expect ensemble_size > 0): {ensemble_size}"
        self.ensemble_size = ensemble_size

        """
        Some fixed hyperparameters, e.g. architecture design of nn-voter
        """
        if model_configurations is None:
            self.model_configurations = {
                "input_size": ensemble_size,
                "hidden_sizes": [],
                "activations": None,
                "activation_type": nn.ReLU(),
                "output_size": 1,
                "dropout_rate": None,
                "dropout_indices": []
            }
        else:
            self.model_configurations = model_configurations
            self.model_configurations["input_size"] = ensemble_size

        self.nn = MLP.build_from_config(
            input_size=self.model_configurations["input_size"],
            hidden_sizes=self.model_configurations["hidden_sizes"],
            activations=self.model_configurations["activations"],
            activation_type=self.model_configurations["activation_type"],
            output_size=self.model_configurations["output_size"],
            dropout_rate=self.model_configurations["dropout_rate"],
            dropout_indices=self.model_configurations["dropout_indices"]
        )

    def vote(self, y_predictions: torch.Tensor) -> torch.Tensor:
        """
        :param y_predictions: Shape [base_learner_num, batch_size, 1]
        """
        assert y_predictions.dim() == 3, f"Invalid dimension of y_predictions (expect [base_learner_num, batch_size, 1]): {y_predictions.shape}"

        # [base_learner_num, batch_size, 1] -> [batch_size, base_learner_num, 1] -> [batch_size, base_learner_num]
        return self.nn(y_predictions.transpose(0, 1).squeeze(-1))

    def parameters(self) -> list:
        """
        Override
        """
        return list(self.nn.parameters())

    def to(self, device: any):
        self.nn.to(device=device)

    @property
    def device(self) -> any:
        return self.nn.device