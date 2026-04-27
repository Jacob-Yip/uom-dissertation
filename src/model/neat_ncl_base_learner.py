import neat
import os
import pickle
from src.neat.neat_ncl_genome import NEATNCLGenome
import torch
from torch import nn
import torch.nn.functional as F

"""
Base learner for a NEAT-NCL model

In the caller code, you can keep this instance alive for the whole generation/training or you create/dispose this instance on request
"""


class NEATNCLBaseLearner(nn.Module):
    def __init__(self, genome: NEATNCLGenome, config: neat.Config, genome_id: int, model_repository_path="data/model/neat"):
        """
        Create a base learner for NEAT ensemble learner
        """
        super().__init__()

        # Checking
        assert genome is not None, f"Missing requirement genome: {genome}"
        assert config is not None, f"Missing requirement config: {config}"

        self.__genome = genome
        self.__config = config
        self.__phenome = neat.nn.FeedForwardNetwork.create(
            self.__genome, self.__config)
        # ID of this base learner
        # Used when storing its genome to the local file system
        self.__genome_id = genome_id
        # List of active hidden nodes, i.e. nodes with connections
        # In the form of {node_id: dummy_data}
        self.__active_hidden_node_indices = {}

        self.__device = torch.device(
            "cpu")  # NEAT base learner must run on CPU (cannot run on GPU)
        # TODO: Allow developers to customise tensor dtype -> follow NEAT configurations if possible (I am not sure whether you can customise float precision in NEAT)
        self.__dtype = torch.float
        # A dictionary of PyTorch trainable weights for every connection (will be updated by PyTorch backpropagation -> copied to NEAT)
        # In the form of {"connection_(start_node_index)_(end_node_index)": weight}
        self.__weights = nn.ParameterDict()
        for connection_key, connection in self.__genome.connections.items():
            # connection_key is in the form of (start_node_index, end_node_index)
            # connection is an instance
            if connection.enabled:
                self.__weights[f"connection_{(connection_key[0])}_{connection_key[1]}"] = nn.Parameter(
                    torch.tensor(connection.weight, dtype=self.__dtype, requires_grad=True))

                # Record active hidden nodes

                if (connection_key[0] not in self.__phenome.input_nodes) and (connection_key[0] not in self.__phenome.output_nodes) and (connection_key[0] not in self.__active_hidden_node_indices):
                    # Add this new active hidden node
                    # None is used as the dummy data
                    self.__active_hidden_node_indices[connection_key[0]] = None

                if (connection_key[1] not in self.__phenome.input_nodes) and (connection_key[1] not in self.__phenome.output_nodes) and (connection_key[1] not in self.__active_hidden_node_indices):
                    # Add this new active hidden node
                    # None is used as the dummy data
                    self.__active_hidden_node_indices[connection_key[1]] = None

        # Other configuration data
        # Absolute path to model repository
        self.__model_repository_path = model_repository_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expect all tensors have been moved to the device in caller
        The output of the node is determined by activation(bias + (response * aggregation(all_input_values)))

        :param x: The input 2D tensor; Unfortunately, I do not think it is possible to do batch processing because we are using my algorithm + NEAT, which handles tensors sequentially; Expect [[batch_1_input_value_1, batch_1_input_value_2, ...]]
        :return: A 2D prediction-tensor of model in the form of [[batch_1_y_1, batch_1_y_2, ...]]
        """
        assert x.ndim == 2, f"Expect a 2D input tensor but received: {x}"
        assert len(
            self.__phenome.input_nodes) == x.shape[1], f"Expect {len(self.__phenome.input_nodes)} input features but received: {x.shape[1]}"

        y_raw = []

        for x_1_sample in x:
            # A dictionary for storing the value of each node (not just input nodes, but also hidden and output nodes)
            # NOTE: Each input node is expected to have negative index (this is defined by NEAT, not me), i.e. -1, -2, -3, ..., -n
            node_values = {-(input_node_index + 1): input_node_value.detach().clone().requires_grad_(True)
                           for input_node_index, input_node_value in enumerate(x_1_sample)}  # Get the innermost elements

            # Follow NEAT's topological evaluation order
            for node_index, activation_function, aggregation_function, bias, response, links in self.__phenome.node_evals:
                node_input_values = []

                for start_node_index, connection_weight in links:
                    # Update the weights because this genome is updated
                    # torch.no_grad() is used here to make sure the change below is not recorded and considered for backpropagation
                    # nn.Parameter() should only be assigned once and should not be modified -> we are using assignment on the attribute "data" instead
                    with torch.no_grad():
                        self.__weights[f"connection_{start_node_index}_{node_index}"].data = torch.tensor(
                            connection_weight, dtype=self.__dtype, requires_grad=True)

                    # Use trainable weight parameter from PyTorch instead of the static non-updatable weight from NEAT
                    node_input_values.append(
                        node_values[start_node_index] * self.__weights[f"connection_{start_node_index}_{node_index}"])

                if len(node_input_values) == 0:
                    # This node has no incoming connections
                    node_input_values = torch.tensor(
                        [0], dtype=self.__dtype, requires_grad=True)
                else:
                    node_input_values = torch.stack(node_input_values)

                # The output of the node is determined by activation(bias + (response * aggregation(inputs)))
                # Must use PyTorch activation and aggregation functions -> otherwise the output will be a float instead of a tensor
                node_output = NEATNCLBaseLearner.to_pytorch_function(activation_function)(
                    bias + response * NEATNCLBaseLearner.to_pytorch_function(aggregation_function)(node_input_values))
                node_values[node_index] = node_output

            # Return the output nodes for this 1 sample
            # NOTE: Order matters (I think the implementation below is correct)
            # Return a 1D tensor
            y_raw.append(torch.stack([node_values[output_node_index]
                         for output_node_index in self.__phenome.output_nodes]))

        return torch.stack(y_raw)

    def synchronise(self) -> None:
        """
        Update genome weights so that they match the PyTorch weights updated via PyTorch backpropagation
        Expected to be invoked after PyTorch backpropagation and before NEAT reproduction
        NOTE: We have assumed that this method is invoked correctly -> no change in structure of genome
        """
        for key, connection in self.__genome.connections.items():
            connection_key = f"connection_{key[0]}_{key[1]}"

            if connection.enabled:
                # Update the actual genome object with the tensor's value
                connection.weight = self.__weights[connection_key].detach(
                ).item()
            else:
                # Extra checking
                # NOTE: This can be removed if you prioritise speed over carefulness/correctness (though I do not think there will be a lot of speed improvement from such removal)
                if connection_key in self.__weights.keys():
                    # We have used this connection before
                    # NOTE: If the above boolean statement is false, it is expected and we just ignore it
                    pytorch_connection_weight = self.__weights[connection_key].detach(
                    ).item()
                    assert pytorch_connection_weight == 0, f"Expect PyTorch weight (of a disabled genome connection) to be 0 but received: {pytorch_connection_weight}"

    @classmethod
    def to_pytorch_function(cls, neat_function: any) -> any:
        """
        Convert a NEAT function to a PyTorch function, which always expects a tensor
        We need this because NEAT function will convert a PyTorch tensor to a float, which will mess up future calculations

        :param neat_function: NEAT function to be converted
        :type neat_function: any
        :return: The equivalent PyTorch function (if there is)
        :rtype: any
        """
        if neat_function is neat.activations.relu_activation:
            return F.relu
        elif neat_function is neat.aggregations.sum_aggregation:
            return torch.sum
        else:
            raise Exception(
                f"Invalid neat_function to be converted: {neat_function}")

    def load_genome(self) -> None:
        """
        Load model genome from self.genome_file_path
        """
        with open(self.genome_file_path, f"rb") as file:
            self.__genome = pickle.load(file)

    def save_genome(self) -> None:
        """
        Save model genome at self.genome_file_path
        """
        # Save genome to a file
        with open(self.genome_file_path, f"wb") as file:
            pickle.dump(self.__genome, file)

    # ================================= Getter =================================
    @property
    def genome(self):
        return self.__genome

    @property
    def config(self):
        return self.__config

    @property
    def phenome(self) -> neat.nn.FeedForwardNetwork:
        return self.__phenome

    @property
    def rank(self) -> int:
        return self.__genome_id

    @property
    def device(self) -> torch.device:
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    @property
    def weights(self) -> nn.ParameterDict:
        return self.__weights

    @property
    def model_repository_path(self) -> str:
        """
        Absolute path to model repository

        :return: Absolute path to model repository
        :rtype: str
        """
        return self.__model_repository_path

    @property
    def genome_file_name(self) -> str:
        assert self.__genome_id is not None and self.__genome_id >= 0, f"Invalid genome_id (expect >= 0): {self.__genome_id}"

        return f"genome_neat_ncl_{self.__genome_id}.pkl"

    @property
    def genome_file_path(self) -> str:
        return os.path.join(self.model_repository_path, self.genome_file_name)

    @property
    def all_hidden_node_num(self) -> int:
        """
        Return number of all hidden nodes (including inactive hidden nodes and active hidden nodes)
        Hidden nodes are nodes that are not input nodes and not output nodes

        :return: Number of all hidden nodes
        :rtype: int
        """
        assert self.__phenome is not None, f"Missing required variable phenome (expect to be not None): {self.__phenome}"

        all_hidden_node_num = len(
            self.__genome.nodes) - len(self.__phenome.input_nodes) - len(self.__phenome.output_nodes)

        assert all_hidden_node_num >= 0, f"Invalid value of active_hidden_node_num (expect >= 0): {len(self.__genome.nodes)} - {len(self.__phenome.input_nodes)} - {len(self.__phenome.output_nodes)} = {all_hidden_node_num}"

        return all_hidden_node_num

    @property
    def active_hidden_node_num(self) -> int:
        """
        Return number of active hidden nodes (excluding inactive hidden nodes)
        Hidden nodes are nodes that are not input nodes and not output nodes

        :return: Number of active hidden nodes
        :rtype: int
        """
        assert self.__phenome is not None, f"Missing required variable phenome (expect to be not None): {self.__phenome}"

        active_hidden_node_num = len(self.__active_hidden_node_indices)

        # Just in case
        assert active_hidden_node_num >= 0, f"Invalid value of active_hidden_node_num (expect >= 0): {active_hidden_node_num}"

        return active_hidden_node_num
