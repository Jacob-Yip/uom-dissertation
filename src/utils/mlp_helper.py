from src.graph.edge import Edge
from src.graph.node import Node
from torch import nn


def build_linear_layers(topological_sort: list, nodes: dict, edges: dict) -> list:
    layer_num = len(nodes) - 1
    # In the form of {(linear_layer_index, (start_node_index, end_node_index)): non_zero_weight}
    non_zero_weights = {}
    # In the form of {(linear_layer_index, node_index): non_zero_bias}
    non_zero_biases = {}
    # Initially, each tower has 1 node, i.e. height is 1 unit
    # NOTE: It will be 1 more than number of layer_num
    linear_layer_heights = [1 for _ in range(layer_num + 1)]

    # Append to edges so that calculate_linear_layer_edge() will also consider (topological_sort[0], input_node_id)
    for start_node in nodes.values():
        if not start_node.is_start_node or start_node.node_id >= 0:
            continue
        else:
            assert not (
                topological_sort[0], start_node.node_id) in edges, f"Expect edges does not have this key: {(topological_sort[0], start_node.node_id)}"

            # Add additional nodes at layers before the ith start node
            data = build_linear_layer_edge_start_node(
                start_node=start_node, topological_sort=topological_sort, linear_layer_heights=linear_layer_heights)

            for ((linear_layer_index, (start_nodex_index, end_node_index)), connection_weight) in data:
                assert not (linear_layer_index, (start_nodex_index, end_node_index)
                            ) in non_zero_weights, f"Expect non_zero_weights does not contain this non-zero weight: {((linear_layer_index, (start_nodex_index, end_node_index)), connection_weight)}"

                non_zero_weights[(
                    linear_layer_index, (start_nodex_index, end_node_index))] = connection_weight

    for edge in edges.values():
        # NOTE: The assertion is removed because we might have intentionally added 0-weight edge for input neurons that are connected to any nodes
        # assert edge.weight != 0, f"Expect edge has weight: {edge}"
        """
        if edge.weight == 0:
            # Special case -> do nothing
            pass
        """

        data = calculate_linear_layer_edge(
            edge=edge,
            topological_sort=topological_sort,
            linear_layer_heights=linear_layer_heights
        )

        for ((linear_layer_index, (start_nodex_index, end_node_index)), connection_weight) in data:
            assert not (linear_layer_index, (start_nodex_index, end_node_index)
                        ) in non_zero_weights, f"Expect non_zero_weights does not contain this non-zero weight: {((linear_layer_index, (start_nodex_index, end_node_index)), connection_weight)}"

            non_zero_weights[(
                linear_layer_index, (start_nodex_index, end_node_index))] = connection_weight

    # linear_layer_heights is updated

    for node in nodes.values():
        if node.bias == 0:
            # Only care about nodes with bias
            continue

        # NOTE: We have assumed that input neurons will never have bias
        linear_layer_index = get_linear_layer_index(
            node_id=node.node_id, topological_sort=topological_sort) - 1  # Only the end-nodes of a linear layer have a bias
        non_zero_biases[(linear_layer_index, 0)] = node.bias

    # Build linear layers
    linear_layers = []

    for i in range(len(linear_layer_heights) - 1):
        start_neuron_num = linear_layer_heights[i]
        end_neuron_num = linear_layer_heights[i + 1]

        current_layer = nn.Linear(start_neuron_num, end_neuron_num)

        # All weights and biases initially are 0
        nn.init.constant_(current_layer.weight, 0.0)
        nn.init.constant_(current_layer.bias, 0.0)

        linear_layers.append(current_layer)

    # Update weights
    for (layer_index, (start_node_index, end_node_index)), weight in non_zero_weights.items():
        # NOTE: nn.Linear stores the layer as a matrix of shape [end_neuron_num, start_neuron_num]
        linear_layers[layer_index].weight.data[end_node_index,
                                               start_node_index] = weight

    # Update biases
    for (layer_index, node_index), bias in non_zero_biases.items():
        linear_layers[layer_index].bias.data[node_index] = bias

    return linear_layers


def calculate_linear_layer_edge(edge: Edge, topological_sort: list, linear_layer_heights: list) -> list:
    """
    :return: [((linear_layer_index, (start_node_index, end_node_index)), edge_weight)]
    """
    # NOTE: The assertion is removed because we might have intentionally added 0-weight edge for input neurons that are connected to any nodes
    # assert edge.weight != 0, f"Expect edge has weight: {edge}"

    start_linear_layer_index = get_linear_layer_index(
        node_id=edge.key[0], topological_sort=topological_sort)
    end_linear_layer_index = get_linear_layer_index(
        node_id=edge.key[1], topological_sort=topological_sort)

    assert end_linear_layer_index > start_linear_layer_index, f"Invalid edge used to determine linear layer indices: {edge}"

    if end_linear_layer_index - start_linear_layer_index == 1:
        # No need to create additional nodes
        if start_linear_layer_index == 0:
            # First node of this layer is an input neuron
            return [((start_linear_layer_index, (get_start_node_linear_layer_index(node=Node(node_id=edge.key[0])), 0)), edge.weight)]
        else:
            return [((start_linear_layer_index, (0, 0)), edge.weight)]
    else:
        # Need to add additional nodes
        data = []

        for linear_layer_index in range(start_linear_layer_index, end_linear_layer_index):
            # No need to create additional node -> create 1 additional node for end_node for each scan instead
            if linear_layer_index == start_linear_layer_index:
                # Originate from the node with index of 0
                start_node_index = 0

                if linear_layer_index == 0:
                    # This node is a start node - special handling -> might not have index of 0
                    # Create a temporary Node instance representing the start node - does not matter in this case
                    start_node_index = get_start_node_linear_layer_index(
                        node=Node(node_id=edge.key[0]))
            else:
                start_node_index = linear_layer_heights[linear_layer_index] - 1

            current_connection_weight = 1  # We need to make sure input value will not be changed

            if linear_layer_index == end_linear_layer_index - 1:
                # Since we are at the final scan, we want to merge all additional nodes to the original node -> do not create an additional node
                # We are merging to the existing node, which always has an index of 0
                end_node_index = 0

                # Last connection before reaching the target node
                # Current connection weight should be changed to edge.weight
                current_connection_weight = edge.weight
            else:
                # As promised, we add 1 additional node to the layer of end_node
                end_node_index = linear_layer_heights[linear_layer_index + 1]
                # Update linear_layer_heights
                # No need to return it as it is a reference to a list variable
                linear_layer_heights[linear_layer_index + 1] += 1

            data.append(((linear_layer_index, (start_node_index,
                        end_node_index)), current_connection_weight))

        return data


def build_linear_layer_edge_start_node(start_node: Node, topological_sort: list, linear_layer_heights: list) -> list:
    """
    Build linear layers
    For input neurons only

    :return: [((linear_layer_index, (start_node_index, end_node_index)), edge_weight)]
    """
    # We need to make sure input value will not be changed
    connection_weight = 1
    start_linear_layer_index = 0
    end_linear_layer_index = get_linear_layer_index(
        node_id=start_node.node_id, topological_sort=topological_sort)

    if end_linear_layer_index == 0:
        # This start node is already at the correct layer, i.e. 0th layer -> no need to do anything
        return []
    else:
        # Need to add additional nodes
        data = []

        for linear_layer_index in range(start_linear_layer_index, end_linear_layer_index):
            if linear_layer_index == 0:
                # We have a specific order of nodes for the 0th layer as it is a layer for all input neurons
                start_node_index = get_start_node_linear_layer_index(
                    node=start_node)

                assert start_node_index >= 0, f"Invalid start_node_index (node={start_node}) calculated: {start_node_index}"

                # Remember that end_linera_layer_index != 0 implies that this input neuron is added
                linear_layer_heights[linear_layer_index] += 1
            else:
                start_node_index = linear_layer_heights[linear_layer_index] - 1

            if linear_layer_index == end_linear_layer_index - 1:
                # We are merging to the start node, which always has an index of 0
                end_node_index = 0
            else:
                # As promised, we add 1 additional node to the layer of end_node
                end_node_index = linear_layer_heights[linear_layer_index + 1]
                # Update linear_layer_heights
                # No need to return it as it is a reference to a list variable
                linear_layer_heights[linear_layer_index + 1] += 1

            data.append(
                ((linear_layer_index, (start_node_index, end_node_index)), connection_weight))

        return data


def get_linear_layer_index(node_id: int, topological_sort: list) -> int:
    """
    Error will be raised if node_id is not in topological_sort
    """
    return topological_sort.index(node_id)


def get_start_node_linear_layer_index(node: Node) -> int:
    assert node.node_id < 0 and node.is_start_node, f"Expect node to be a start node with negative node_id: {node}"

    return -1 * (node.node_id + 1)
