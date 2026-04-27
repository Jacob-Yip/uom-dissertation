from abc import ABC, abstractmethod
import torch


class Voter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def vote(self, y_predictions: torch.Tensor) -> torch.Tensor:
        """
        Aggregate predictions from base learner and return the ensemble prediction

        :param y_predictions: The predictions of base learners
        :return: The ensemble prediction
        """
        pass
    
    def parameters(self) -> list:
        return []