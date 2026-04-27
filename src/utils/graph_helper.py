import neat
from src.graph.node import Node
from src.graph.edge import Edge
from src.neat.neat_ncl_genome import NEATNCLGenome


def get_genome_graph(genome: NEATNCLGenome, input_keys: list) -> tuple:
    """
    Get the graph representation of this genome
    NOTE: nodes will be sorted in ascending order of node_id

    :return: (nodes, edges, activation)
    """
    activation = None
    # Dictionary is used instead of list to prevent duplicate node instances
    # In the form of {node_id: Node instance}
    nodes = {}
    # In the form of {edge_connection_key: Edge instance}
    edges = {}

    # Update nodes
    for (node_id, node) in genome.nodes.items():
        assert node.response == 1, f"Expect 1 as node response: {node.response}"
        if activation is None:
            # Update activation
            activation = node.activation
        else:
            # Error checking
            assert node.activation == activation, f"Expect consistent node activations: {activation} (expected) != {node.activation} (current node)"

        assert not node_id in nodes, f"Expect a new node: {node_id}"

        # Add this new node to the dictionary
        nodes[node_id] = Node(node_id=node_id, bias=node.bias)

        # Will add successor later

    # Update edges
    for (connection_id, connection) in genome.connections.items():
        if not connection.enabled:
            # To reduce computation, we do not process edges that are disabled as they will have a weight of 0
            continue
        else:
            start_node_id = connection.key[0]
            end_node_id = connection.key[1]

            if not start_node_id in nodes:
                # Can be an input node or a bias node
                assert start_node_id < 0, f"Expect input node or bias node as start-node with negative node id: {start_node_id}"

                start_node = Node(node_id=start_node_id)
                nodes[start_node_id] = start_node

            if not end_node_id in nodes:
                # Not likely - cannot think of a case where this happens but just in case
                # Should not be input node
                # Could be bias node
                assert end_node_id < 0, f"Expect input node or bias node as end-node with negative node id: {end_node_id}"
                end_node = Node(node_id=end_node_id)
                nodes[end_node_id] = end_node

            # ===================================================================================

            assert not connection.key in edges, f"Expect a new edge: {connection.key}"

            # Add this new edge to the dictionary
            edges[connection.key] = Edge(
                key=connection.key, weight=connection.weight)

            # Add successor to start_node
            nodes[start_node_id].add_successor(nodes[end_node_id])

    """
    There might be a chance that some input neurons are not connected to any nodes and thus, are omitted
    However, those input neurons must exist for our algorithm to work
    """
    for hidden_node_id in input_keys:
        if hidden_node_id in nodes:
            # We have already created the corresponding Node instance
            continue
        else:
            nodes[hidden_node_id] = Node(node_id=hidden_node_id)

            # We need to link (with weight 0) these input neurons to the current first non-input neuron in the topological sort
            if len(nodes) > 1:
                # Get the first node in the topological sort
                current_sorted_node_ids = topological_sort(nodes=nodes)
                first_non_input_node_id = -1
                for current_sorted_node_id in current_sorted_node_ids:
                    if current_sorted_node_id >= 0:
                        # This is a non-input neuron
                        first_non_input_node_id = current_sorted_node_id
                        break

                assert first_non_input_node_id >= 0, f"Unable to find the first non-input neuron from the topological sort of this genome: {current_sorted_node_ids}"

                edges[(hidden_node_id, first_non_input_node_id)] = Edge(
                    key=(hidden_node_id, first_non_input_node_id), weight=0)

                nodes[hidden_node_id].add_successor(
                    nodes[first_non_input_node_id])
            else:
                # This hidden node is the only node
                # Happen when your neural network only has input neurons but no hidden/output neurons
                # In this case, we do not add edge as there is no point -> topological sort will still work for 1 node
                # len(nodes) <= 1 is the same as len(nodes) == 1 here
                pass

    return (dict(sorted(nodes.items())), edges, activation)


def depth_first_search(node: Node, node_states: dict, sorted_node_ids: list) -> None:
    node_state = node_states[node.node_id]

    if node_state == "N":
        # Ignore as it is expected
        node_states[node.node_id] = "O"  # Exploring this node
    elif node_state == "O":
        # Cycle detected
        raise Exception(
            f"Detected cycle in graph, which passes through this node: {str(node)}")
    elif node_state == "F":
        # Do nothing
        return node
    else:
        # Invalid node state
        raise Exception(f"Invalid node state: {node_state}")

    for successor in node.successors:
        depth_first_search(node=successor, node_states=node_states,
                           sorted_node_ids=sorted_node_ids)

    node_states[node.node_id] = "F"  # Finish exploring this node

    # Prepend current node to the beginning of the list
    sorted_node_ids.insert(0, node.node_id)


def topological_sort(nodes: dict) -> list:
    """
    Expect nodes to be sorted already

    :param: nodes: The dictionary of nodes in the form of {node_id: node_instance}
    :return: a list of node_ids in topological sort
    """
    sorted_node_ids = []  # The topological sort
    """
    In the form of {node_id: state}
    state: 
    - N: not explored (default state)
    - O: exploring (ongoing)
    - F: finished
    """
    node_states = {}

    # We want to do depth first search for nodes with larger node IDs
    # In this case, the topological sort is likely to start with nodes with negative node IDs, i.e. input neurons (this is essential in future debugging)
    nodes = dict(sorted(nodes.items(), reverse=True))

    # Set up
    for node in nodes.values():
        node_states[node.node_id] = "N"

    # Expect nodes to be sorted already
    # Nodes with negative id will all be starting nodes

    while len(sorted_node_ids) < len(nodes):
        # We have not finished sorting yet
        for start_node_id, start_node in nodes.items():
            if node_states[start_node_id] == "F":  # We assume it will not be O
                # We have explored from this starting node already -> move on to the next starting node
                continue
            else:
                # Start from this starting node
                assert node_states[
                    start_node_id] == "N", f"Expect starting node to have state 'N': {node_states[start_node_id]}"

                depth_first_search(
                    node=start_node, node_states=node_states, sorted_node_ids=sorted_node_ids)

    return sorted_node_ids
