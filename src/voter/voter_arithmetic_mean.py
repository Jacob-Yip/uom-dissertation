from src.voter.voter import Voter
import torch


class ArithmeticMeanVoter(Voter):
    def __init__(self):
        super().__init__()

    def vote(self, y_predictions: torch.Tensor) -> torch.Tensor:
        # y_bar
        return y_predictions.mean(dim=0, keepdim=False, dtype=torch.float)