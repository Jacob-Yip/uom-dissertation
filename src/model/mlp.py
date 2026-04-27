import neat
from src.model.base_model import BaseModel
from src.neat.neat_ncl_genome import NEATNCLGenome
import src.utils.genome_helper as genome_helper
import src.utils.graph_helper as graph_helper
import src.utils.mlp_helper as mlp_helper
from torch import nn


class MLP(BaseModel):
    def __init__(self, layers=None, activations=None, dropout_rate=None, dropout_indices=[]) -> None:
        super().__init__()

        assert not layers is None and not activations is None, f"Missing required parameter(s) layers and activations: {layers} and {activations}"

        self.__dropout_rate = dropout_rate
        self.__dropout = None
        if not dropout_rate is None:
            self.__dropout = nn.Dropout(p=self.__dropout_rate)
        self.__dropout_indices = dropout_indices

        self.__activations = activations
        self.__layer_heights = []
        self.__layers = nn.ModuleList()

        for layer_index, layer in enumerate(layers):
            self.__layers.append(layer)

            first_layer_height = layer.in_features
            self.__layer_heights.append(first_layer_height)

            if layer_index == len(layers) - 1:
                # Last layer
                self.__layer_heights.append(layer.out_features)

    @property
    def layer_heights(self) -> list:
        return self.__layer_heights

    @property
    def layers(self) -> list:
        return self.__layers

    @property
    def activations(self) -> list:
        return self.__activations

    @classmethod
    def build_from_config(cls, input_size=1, hidden_sizes=[], activations=None, activation_type=nn.ReLU(), output_size=1, dropout_rate=None, dropout_indices=[]):
        # Type of all activation functions in this neural network
        assert not activation_type is None, f"Expect activation_type other than None"

        if activations is None:
            # NOTE: Unlike NEAT, the output layer of a MLP should not have an activation function
            activations = [activation_type] * \
                len(hidden_sizes) + [nn.Identity()]
        else:
            assert len(hidden_sizes) == len(
                activations), f"Inconsistent number of hidden layers and activation functions"

        layers = []

        input_neuron_num = input_size  # Local number of input neurons
        for _, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(input_neuron_num, hidden_size))
            input_neuron_num = hidden_size

        layers.append(nn.Linear(input_neuron_num, output_size))

        return MLP(layers=layers, activations=activations, dropout_rate=dropout_rate, dropout_indices=dropout_indices)

    @classmethod
    def build_from_linear_layers(cls, layers: list, layer_heights: list, activations=None, dropout_rate=None, dropout_indices=[]):
        # TODO: We build activations manually here -> maybe change to something else in the future
        activation_type = nn.ReLU()
        if activations is None:
            # NOTE: Expected to be for NEAT (not the main factory method though) -> output layer should have an activation function
            activations = [activation_type] * len(layers)
        else:
            assert len(activations) == len(
                layers), f"Inconsistent number of layers and activation functions: {len(layers)} != {len(activations)}"

        model = MLP(layers=layers, activations=activations,
                    dropout_rate=dropout_rate, dropout_indices=dropout_indices)

        # NOTE: This is removed due to high computation
        # assert layer_heights == model.layer_heights, f"Incorrect layer_heights: {layer_heights} != {model.layer_heights}"

        return model

    @classmethod
    def build_from_genome(cls, genome: NEATNCLGenome, config: neat.Config, dropout_rate=None, dropout_indices=[]):
        genome_nodes, genome_edges, genome_activation = graph_helper.get_genome_graph(
            genome=genome,
            input_keys=config.genome_config.input_keys
        )

        # TODO: Allow developers to customise activation_type based on the tsring genome_activation
        activation_type = nn.ReLU()
        # NOTE: We are using sigmoid here because sigmoid is better for this example but in our actual project, we should use nn.ReLU()
        # NOTE: Cannot use sigmoid as the algorithm only works for indempotent activation functions
        # activation_type = nn.Sigmoid()

        sorted_node_ids = graph_helper.topological_sort(nodes=genome_nodes)

        linear_layers = mlp_helper.build_linear_layers(
            topological_sort=sorted_node_ids, nodes=genome_nodes, edges=genome_edges)

        # NOTE: Expected to be for NEAT -> output layer should have an activation function
        return MLP(layers=linear_layers, activations=[activation_type] * len(linear_layers), dropout_rate=dropout_rate, dropout_indices=dropout_indices)

    def forward(self, x):
        assert len(self.__layers) == len(
            self.__activations), f"Number of layers does not match number of activation functions: {len(self.__layers)} != {len(self.__activations)}"

        layer_index = 0

        for layer, activation in zip(self.__layers, self.__activations):
            x = activation(layer(x))

            if not self.__dropout is None and layer_index in self.__dropout_indices:
                # Apply dropout
                x = self.__dropout(x)

            layer_index += 1

        return x
