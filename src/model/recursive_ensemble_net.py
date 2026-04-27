from src.model.recursive_traditional_ensemble_net import RecursiveTraditionalEnsembleNet
from src.model.recursive_static_ncl_ensemble_net import RecursiveStaticNCLEnsembleNet
from src.model.neat_ncl_ensemble_net import NEATNCLEnsembleNet
from src.voter.voter import Voter
from src.voter.voter_arithmetic_mean import ArithmeticMeanVoter
from src.voter.voter_median import MedianVoter
from src.voter.voter_nn import NNVoter
import torch
from torch import nn

"""
Technically, our current design supports running different types of ensemble models as base learners. However, for the sake of my project and experiment report, I will assume they are of the same type
"""


class RecursiveEnsembleNet(nn.Module):
    def __init__(self, architecture: list):
        """
        :param architecture: The recursive data structure indicating what to create in the the below form: 
            {
                "<ensemble_net_type>": [<sub_architecture_dict>]
                "voter": <voter_type>
            }
            - If <architecture_dict> does not have the key "voter", it is for the base case. Otherwise, it is for the step case, i.e. RecursiveEnsembleNet
        """
        super().__init__()

        # List of base learners from the current level of perspective
        # In the form of [RecursiveEnsembleNet]
        self.__base_learners = []

        for key, value in architecture.items():
            # voter
            if key.lower() == "voter":
                # We need to do voter at the end in case it is a NNVoter and we need to know how many base learners we need
                continue
            # <ensemble_net_type>
            elif key.lower() == "traditional":
                for sub_architecture in value:
                    assert isinstance(
                        sub_architecture, dict), f"Invalid type of sub_architecture (expect dict): {sub_architecture} has type {type(sub_architecture)}"

                    if not f"voter" in sub_architecture:
                        # Base case - RecursiveTraditionalEnsembleNet
                        self.__base_learners.append(RecursiveTraditionalEnsembleNet(
                            ensemble_size=sub_architecture["ensemble_size"],
                            model_configurations=sub_architecture["model_configurations"],
                            model_repository_path=sub_architecture["model_repository_path"],
                            voter=sub_architecture["ensemble_voter"]
                        ))
                    else:
                        # Step case - RecursiveEnsembleNet
                        self.__base_learners.append(
                            RecursiveEnsembleNet(architecture=sub_architecture))
            elif key.lower() == "static_ncl":
                for sub_architecture in value:
                    assert isinstance(
                        sub_architecture, dict), f"Invalid type of sub_architecture (expect dict): {sub_architecture} has type {type(sub_architecture)}"

                    if not f"voter" in sub_architecture:
                        # Base case - RecursiveStaticNCLEnsembleNet
                        self.__base_learners.append(RecursiveStaticNCLEnsembleNet(
                            ensemble_size=sub_architecture["ensemble_size"],
                            model_configurations=sub_architecture["model_configurations"],
                            model_repository_path=sub_architecture["model_repository_path"],
                            voter=sub_architecture["ensemble_voter"]
                        ))
                    else:
                        # Step case - RecursiveEnsembleNet
                        self.__base_learners.append(
                            RecursiveEnsembleNet(architecture=sub_architecture))
            elif key.lower() == "neat_ncl":
                for sub_architecture in value:
                    assert isinstance(
                        sub_architecture, dict), f"Invalid type of sub_architecture (expect dict): {sub_architecture} has type {type(sub_architecture)}"

                    if not f"voter" in sub_architecture:
                        # Base case - NEATNCLEnsembleNet
                        self.__base_learners.append(NEATNCLEnsembleNet(
                            config=sub_architecture["config"],
                            model_repository_path=sub_architecture["model_repository_path"],
                            voter=sub_architecture["ensemble_voter"]
                        ))
                    else:
                        # Step case - RecursiveEnsembleNet
                        self.__base_learners.append(
                            RecursiveEnsembleNet(architecture=sub_architecture))
            else:
                raise Exception(
                    f"Invalid architecture-key (expect voter/traditional/static_ncl/neat_ncl): {key}")

        # Number of base learners at the current level, which is for the voter if the voter is a NNVoter
        base_learner_num = 0
        if f"traditional" in architecture:
            base_learner_num += len(architecture[f"traditional"])
        if f"static_ncl" in architecture:
            base_learner_num += len(architecture[f"static_ncl"])
        if f"neat_ncl" in architecture:
            base_learner_num += len(architecture[f"neat_ncl"])

        self.__voter = self.get_voter(
            voter_value=architecture[f"voter"], ensemble_size=base_learner_num)

    def get_voter(self, voter_value: str, ensemble_size=-1) -> Voter:
        """
        :param voter_value: The type of voter
            - arithmetic_mean
            - median
            - nn
        :param ensemble_size: The number of input features for the voter if the voter is a NNVoter
        """
        assert isinstance(
            voter_value, str), f"Invalid type of voter_value (expect str): {voter_value} has type {type(voter_value)}"

        if voter_value.lower() == "arithmetic_mean":
            return ArithmeticMeanVoter()
        elif voter_value.lower() == "median":
            return MedianVoter()
        elif voter_value.lower() == "nn":
            return NNVoter(
                ensemble_size=ensemble_size,
                # We pass None so that the voter creates model configurations for us automatically
                model_configurations=None
            )
        
    def recursive_to(self, device: any):
        for base_learner in self.__base_learners:
            if isinstance(base_learner, RecursiveEnsembleNet):
                base_learner.recursive_to(device=device)
            elif isinstance(base_learner, RecursiveTraditionalEnsembleNet):
                base_learner.to(device=device)
            elif isinstance(base_learner, RecursiveStaticNCLEnsembleNet):
                base_learner.to(device=device)
            elif isinstance(base_learner, NEATNCLEnsembleNet):
                # Do nothing
                pass
            else:
                raise Exception(f"Invalid type of base learner (expect RecursiveEnsembleNet/RecursiveTraditionalEnsembleNet/RecursiveStaticNCLEnsembleNet/NEATNCLEnsembleNet): {base_learner} has type {type(base_learner)}")
        
        if isinstance(self.__voter, NNVoter):
            self.__voter.to(device)


    def recursive_train(self, X: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        NOTE: If we are doing recursive NEAT NCL, X and y will be ignored (data will be passed via fitness_function_arguments in kwargs instead)
        NOTE: If we are doing traditional/(static NCL), X and y should tensors of 1 batch


        NOTE: \hat{y} is the ensemble prediction and \hat{y}_i is the prediction from the ith base learner

        RecursiveTraditionalEnsembleNet (when voter == NNVoter()): 
        - Loss function of base learner: 
            - Aggregator: Loss = (\hat{y} - y)^2
            - Base learner: Loss = (\hat{y}_i - y)^2

        RecursiveStaticNCLEnsembleNet (when voter == NNVoter()):
        - Loss function of base learner: 
            - Aggregator: Loss = (\hat{y} - y)^2
            - Base learner Loss = (\hat{y} - y)^2 - \lambda * (\hat{y}_i - \hat{y})^2

        :param X: The training X data
        :param y: The training y data
        """
        base_learner_outputs = []

        for i in range(len(self.__base_learners)):
            base_learner = self.__base_learners[i]

            if isinstance(base_learner, RecursiveEnsembleNet):
                output = base_learner.recursive_train(X=X, y=y, **kwargs)

                base_learner_outputs.append(output)
            elif isinstance(base_learner, RecursiveTraditionalEnsembleNet):
                # \hat{y}
                y_prediction_ensemble = base_learner(x=X)

                # Update aggregator
                voter = base_learner.voter

                if isinstance(voter, NNVoter):
                    optimizer_voter = torch.optim.Adam(
                        voter.parameters(), lr=kwargs["learning_rate"])

                    optimizer_voter.zero_grad()

                    # Compute loss of aggregator
                    loss_voter = nn.MSELoss()(y_prediction_ensemble, y)

                    loss_voter.backward()

                    optimizer_voter.step()

                # Update base learner of this ensemble model
                for mlp_traditional in base_learner.base_learners:
                    # \hat{y}_i
                    y_prediction_i = mlp_traditional(x=X)

                    optimizer_mlp_traditional = torch.optim.Adam(
                        mlp_traditional.parameters(), lr=kwargs["learning_rate"])

                    optimizer_mlp_traditional.zero_grad()

                    # Compute loss of MLP in the traditional ensemble model
                    loss_mlp_traditional = nn.MSELoss()(y_prediction_i, y)

                    loss_mlp_traditional.backward()

                    optimizer_mlp_traditional.step()

                base_learner_outputs.append(y_prediction_ensemble.detach())
                # TODO: Confirm correctness
            elif isinstance(base_learner, RecursiveStaticNCLEnsembleNet):
                # \hat{y}
                y_prediction_ensemble = base_learner(x=X)

                # Update aggregator
                voter = base_learner.voter

                if isinstance(voter, NNVoter):
                    optimizer_voter = torch.optim.Adam(
                        voter.parameters(), lr=kwargs["learning_rate"])

                    optimizer_voter.zero_grad()

                    # Compute loss of aggregator
                    loss_voter = nn.MSELoss()(y_prediction_ensemble, y)

                    loss_voter.backward()

                    optimizer_voter.step()

                # Update base learner of this ensemble model
                for mlp_static_ncl in base_learner.base_learners:
                    # \hat{y}_i
                    y_prediction_i = mlp_static_ncl(x=X)

                    optimizer_mlp_static_ncl = torch.optim.Adam(
                        mlp_static_ncl.parameters(), lr=kwargs["learning_rate"])

                    optimizer_mlp_static_ncl.zero_grad()

                    # Compute loss of MLP in the static NCL ensemble model
                    loss_mlp_static_ncl = nn.MSELoss()(y_prediction_i, y) - \
                        kwargs["correlation_penalty_coefficient"] * \
                        nn.MSELoss()(y_prediction_i, y_prediction_ensemble.detach())

                    loss_mlp_static_ncl.backward()

                    optimizer_mlp_static_ncl.step()

                base_learner_outputs.append(y_prediction_ensemble.detach())
                # TODO: Confirm correctness
            elif isinstance(base_learner, NEATNCLEnsembleNet):
                voter = base_learner.voter

                # For error checking
                assert isinstance(voter, ArithmeticMeanVoter) or isinstance(
                    voter, MedianVoter), f"Invalid voter for NEATNCLEnsembleNet (expect ArithmeticMeanVoter/MedianVoter): {voter} has type {type(voter)}"

                base_learner.evolve(
                    evolution_epoch=kwargs["evolution_epoch"],
                    fitness_function_arguments=kwargs["fitness_function_arguments"]
                )

                base_learner_outputs.append(base_learner(X))

                # Maybe the above implementation is incorrect because evolve should be run after base_learner(X)
            else:
                raise Exception(
                    f"Invalid type of base_learners[{i}]: {base_learner} has type {type(base_learner)}")

        self_output = self.__voter.vote(y_predictions=torch.stack(base_learner_outputs, dim=0))

        if isinstance(self.__voter, NNVoter):
            optimizer_self = torch.optim.Adam(self.__voter.parameters(), lr=kwargs["learning_rate"])

            optimizer_self.zero_grad()

            loss_self = nn.MSELoss()(self_output, y)
            loss_self.backward()

            optimizer_self.step()

        return self_output.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_learner_outputs = []

        for base_learner in self.__base_learners:
            if isinstance(base_learner, RecursiveEnsembleNet):
                output = base_learner(x)

                base_learner_outputs.append(output)
            elif isinstance(base_learner, RecursiveTraditionalEnsembleNet):
                output = base_learner(x)

                base_learner_outputs.append(output)
            elif isinstance(base_learner, RecursiveStaticNCLEnsembleNet):
                output = base_learner(x)

                base_learner_outputs.append(output)
            elif isinstance(base_learner, NEATNCLEnsembleNet):
                output = base_learner(x)

                base_learner_outputs.append(output)
            else:
                raise Exception(
                    f"Invalid type of base_learner (expect RecursiveEnsembleNet/RecursiveTraditionalEnsembleNet, RecursiveStaticNCLEnsembleNet, NEATNCLEnsembleNet): {base_learner} has type {type(base_learner)}")

        return self.__voter.vote(y_predictions=torch.stack(base_learner_outputs, dim=0))
