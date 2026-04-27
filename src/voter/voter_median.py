from src.voter.voter import Voter
import torch


class MedianVoter(Voter):
    def __init__(self):
        super().__init__()

    def vote(self, y_predictions: torch.Tensor) -> torch.Tensor:
        """
        torch.median(...) returns 2 things, <indices> and <values>
        We are only interested in the attribute <values>
        """
        return torch.median(y_predictions, dim=0).values