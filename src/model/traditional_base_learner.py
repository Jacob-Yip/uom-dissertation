import neat
import torch
from src.model.base_model import BaseModel
from src.model.mlp import MLP
from src.neat.neat_ncl_genome import NEATNCLGenome

"""
A wrapper for any model, e.g. mlp
Contains functions, e.g. training
Base learner for traditional ensemble model, i.e. without NCL
Technically, you can just use MLP.build_from_config() instead but I am too lazy so I just copy and paste code
"""


class TraditionalBaseLearner:
    def __init__(self, model: BaseModel, device, rank: int, device_index: int, optimizer, loss_function, use_cpu=False, model_repository_path="data/model"):
        """
        Create a base learner for traditional ensemble learner
        world_size = device_num * model_num_per_gpu

        :param: rank: Unique ID of the GPU running this base learner among the ensemble model; Start from 0 inclusively; Expect value [0, device_num)
        :param: device_index: Unique local ID of this base learner in this GPU; Final unique ID is rank_device_index; Start from 0 inclusively
        :param: use_cpu: True if we are training the model with a CPU
        """
        assert not use_cpu, f"We do not support CPU currently"

        # Model metadata
        self.__model = model
        self.__device = device
        self.__rank = rank
        self.__device_index = device_index
        self.__stream = torch.cuda.Stream(device)

        if not self.__model is None:
            # If self.__model is None, that means it has not been loaded
            # E.g. during evaluation instead of training
            self.__model.to(self.__device)  # Move device to GPU

        # For training
        self.__optimizer = optimizer
        self.__loss_function = loss_function

        # Other configuration data
        # TODO: Will not be used now
        self.__use_cpu = use_cpu
        self.__model_repository_path = model_repository_path

    @classmethod
    def build_from_config(cls, model: BaseModel, device, rank: int, device_index: int, optimizer, loss_function, use_cpu=False, model_repository_path="data/model"):
        return TraditionalBaseLearner(
            model=model,
            device=device,
            rank=rank,
            device_index=device_index,
            optimizer=optimizer,
            loss_function=loss_function,
            use_cpu=use_cpu,
            model_repository_path=model_repository_path
        )

    @classmethod
    def build_from_genome(cls, genome: NEATNCLGenome, config: neat.Config, device, rank: int, device_index: int, learning_rate: float, loss_function, use_cpu=False, model_repository_path="data/model"):
        # NOTE: We have assumed that this is MLP
        model = MLP.build_from_genome(genome=genome, config=config)

        optimizer = None
        if not learning_rate is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        return TraditionalBaseLearner(
            model=model,
            device=device,
            rank=rank,
            device_index=device_index,
            optimizer=optimizer,
            loss_function=loss_function,
            use_cpu=use_cpu,
            model_repository_path=model_repository_path
        )

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def rank(self) -> int:
        return self.__rank

    @property
    def device_index(self) -> int:
        return self.__device_index

    @property
    def model_repository_path(self) -> str:
        return self.__model_repository_path

    @property
    def optimizer(self) -> any:
        return self.__optimizer

    @property
    def model(self) -> BaseModel:
        return self.__model

    @model.setter
    def model(self, model: BaseModel) -> None:
        self.__model = model

    @property
    def layer_heights(self) -> list:
        return self.__model.layer_heights

    def train_base_learner(self, y_prediction: torch.Tensor, y_true: torch.Tensor, **kwargs) -> None:
        """
        Train this base learner with normal ensemble learning, i.e. each base learner is trained independently
        Will save the trained model
        Assume we are using GPU
        Expect all tensors have been moved to the device in caller
        Will set to train mode

        :param: kwargs: Expect (save_model)
        """
        assert "save_model" in kwargs, f"Missing required argument(s): save_model"
        # No need to cast type of kwargs items
        save_model = kwargs["save_model"]

        self.train()  # Set to train mode

        self.__optimizer.zero_grad()

        base_learner_loss = self.__loss_function(y_prediction, y_true)

        base_learner_loss.backward()
        self.__optimizer.step()

        # Save model for future use, e.g. evaluation
        if save_model:
            self.save_model()

    def __call__(self, *args, **kwargs) -> any:
        """
        Wrapper function of PyTorch model
        with torch.cuda.stream(self.__stream) is wrapped around method
        Expect all tensors have been moved to the device in caller
        NOTE: Need to manually set mode, i.e. train/eval, in caller

        :return: prediction-tensor of model
        """
        with torch.cuda.stream(self.__stream):
            return self.__model(*args, **kwargs)

    def eval(self) -> None:
        """
        Wrapper function of PyTorch model
        """
        self.__model.eval()

    def train(self) -> None:
        """
        Wrapper function of PyTorch model
        """
        self.__model.train()

    def load_state_dict(self, state_dict, strict=True, assign=False, **kwargs) -> None:
        """
        Wrapper function of PyTorch model

        :param: kwargs: Expect {"model_configurations": dictionary of model configurations} if self.__model is None
        """
        if self.__model is None:
            assert "model_configurations" in kwargs, f"Missing required model configurations to create model: {kwargs}"

            model_configurations = kwargs["model_configurations"]

            self.__model = MLP.build_from_config(
                input_size=model_configurations["input_size"],
                hidden_sizes=model_configurations["hidden_sizes"],
                activations=model_configurations["activations"],
                activation_type=model_configurations["activation_type"],
                output_size=model_configurations["output_size"],
                dropout_rate=model_configurations["dropout_rate"],
                dropout_indices=model_configurations["dropout_indices"]
            )

        self.__model.load_state_dict(
            state_dict=state_dict, strict=strict, assign=assign)

        # Move device to the correct GPU
        assert not self.__device is None, f"Expect device set during instantiation: {self.__device}"
        self.__model.to(self.__device)

    def save_model(self) -> None:
        """
        Save model at self.__model_repository_path
        """
        torch.save(self.__model.state_dict(),
                   f"{self.__model_repository_path}/base_learner_traditional_{self.__rank}_{self.__device_index}.pt")
