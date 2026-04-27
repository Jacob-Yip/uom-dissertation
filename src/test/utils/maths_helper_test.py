import math
import neat
import os
import pytest
from src.model.neat_ncl_base_learner import NEATNCLBaseLearner
from src.neat.innovation_tracker import InnovationTracker
from src.neat.neat_ncl_genome import NEATNCLGenome
from src.utils import genome_helper, maths_helper
from src.utils.data_container import Table
import torch

"""
Run: python -m src.test.utils.maths_helper_test
"""


@pytest.fixture
def config() -> neat.Config:
    # Load config file
    config_path = os.path.join(
        os.path.dirname(__file__), os.path.join("./", "config-test.ini"))

    config = neat.Config(NEATNCLGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    config.genome_config.innovation_tracker = InnovationTracker()

    return config


@pytest.fixture
def genome_factory() -> any:
    def build_genome(config: neat.Config, genome_id: int) -> NEATNCLGenome:
        assert genome_id >= 0, f"Invalid genome_id (expect >= 0): {genome_id}"

        genome = NEATNCLGenome(genome_id)
        genome.configure_new(config.genome_config)

        return genome

    return build_genome


@pytest.fixture
def fixed_genomes_factory() -> any:
    def build_fixed_genomes(config: neat.Config) -> list:
        """
        Return [(genome_id, genome)] of fixed genomes
        NOTE: Innovation number goes from 1, 2, ...
        NOTE: We will use the data below for all testings

        Genome data: 
        1. genome_1
            - Input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                - True value: [17]
                - Prediction: [71]
            - Input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                - True value: [47]
                - Prediction: [74]
            - Input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                - True value: [157]
                - Prediction: [751]
        2. genome_2
            - Input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                - True value: [17]
                - Prediction: [7]
            - Input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                - True value: [47]
                - Prediction: [3]
            - Input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                - True value: [157]
                - Prediction: [701]

        :param config: Configuration instance
        :type config: neat.Config
        :return: [(genome_id, genome)] of fixed genomes
        :rtype: list
        """
        innovation_number_counter = 1  # Start from 1

        # Genome 1
        # The number passed is genome_id
        genome_1 = config.genome_type(1)

        # There are 13 input features
        # Input nodes are not included in genome.nodes
        # Hidden node
        genome_1.nodes[1] = genome_1.create_node(config.genome_config, 1)
        genome_1.nodes[1].bias = 1
        genome_1.nodes[2] = genome_1.create_node(config.genome_config, 2)
        genome_1.nodes[2].bias = 2
        # Output node
        # Node_0 is the output node
        genome_1.nodes[0] = genome_1.create_node(config.genome_config, 0)
        genome_1.nodes[0].bias = 0

        # Configure the connections, which is in the form of (from_node_id, to_node_id)
        genome_1_connection_1_key = (-1, 1)
        genome_1_connection_1 = genome_1.create_connection(
            config.genome_config, genome_1_connection_1_key[0], genome_1_connection_1_key[1], innovation_number_counter)
        innovation_number_counter += 1
        genome_1_connection_1.weight = 10
        genome_1_connection_1.enabled = True
        genome_1.connections[genome_1_connection_1_key] = genome_1_connection_1

        genome_1_connection_2_key = (-2, 2)
        genome_1_connection_2 = genome_1.create_connection(
            config.genome_config, genome_1_connection_2_key[0], genome_1_connection_2_key[1], innovation_number_counter)
        innovation_number_counter += 1
        genome_1_connection_2.weight = 20
        genome_1_connection_2.enabled = True
        genome_1.connections[genome_1_connection_2_key] = genome_1_connection_2

        genome_1_connection_3_key = (1, 0)
        genome_1_connection_3 = genome_1.create_connection(
            config.genome_config, genome_1_connection_3_key[0], genome_1_connection_3_key[1], innovation_number_counter)
        innovation_number_counter += 1
        genome_1_connection_3.weight = 3
        genome_1_connection_3.enabled = True
        genome_1.connections[genome_1_connection_3_key] = genome_1_connection_3

        genome_1_connection_4_key = (2, 0)
        genome_1_connection_4 = genome_1.create_connection(
            config.genome_config, genome_1_connection_4_key[0], genome_1_connection_4_key[1], innovation_number_counter)
        innovation_number_counter += 1
        genome_1_connection_4.weight = 7
        genome_1_connection_4.enabled = True
        genome_1.connections[genome_1_connection_4_key] = genome_1_connection_4

        # Genome 2
        # The number passed is genome_id
        genome_2 = config.genome_type(2)

        # There are 13 input features
        # Input nodes are not included in genome.nodes
        # Hidden node
        genome_2.nodes[1] = genome_2.create_node(config.genome_config, 1)
        genome_2.nodes[1].bias = 1
        genome_2.nodes[2] = genome_2.create_node(config.genome_config, 2)
        genome_2.nodes[2].bias = 0
        genome_2.nodes[3] = genome_2.create_node(config.genome_config, 3)
        genome_2.nodes[3].bias = 1
        # Output node
        # Node_0 is the output node
        genome_2.nodes[0] = genome_2.create_node(config.genome_config, 0)
        genome_2.nodes[0].bias = 0

        # Configure the connections, which is in the form of (from_node_id, to_node_id)
        genome_2_connection_1_key = (-1, 1)
        genome_2_connection_1 = genome_2.create_connection(
            config.genome_config, genome_2_connection_1_key[0], genome_2_connection_1_key[1], innovation_number_counter)
        innovation_number_counter += 1
        genome_2_connection_1.weight = 10
        genome_2_connection_1.enabled = True
        genome_2.connections[genome_2_connection_1_key] = genome_2_connection_1

        genome_2_connection_2_key = (-1, 2)
        genome_2_connection_2 = genome_2.create_connection(
            config.genome_config, genome_2_connection_2_key[0], genome_2_connection_2_key[1], innovation_number_counter)
        innovation_number_counter += 1
        genome_2_connection_2.weight = 1
        genome_2_connection_2.enabled = True
        genome_2.connections[genome_2_connection_2_key] = genome_2_connection_2

        genome_2_connection_3_key = (-2, 3)
        genome_2_connection_3 = genome_2.create_connection(
            config.genome_config, genome_2_connection_3_key[0], genome_2_connection_3_key[1], innovation_number_counter)
        innovation_number_counter += 1
        genome_2_connection_3.weight = 20
        genome_2_connection_3.enabled = True
        genome_2.connections[genome_2_connection_3_key] = genome_2_connection_3

        genome_2_connection_4_key = (1, 0)
        genome_2_connection_4 = genome_2.create_connection(
            config.genome_config, genome_2_connection_4_key[0], genome_2_connection_4_key[1], innovation_number_counter)
        innovation_number_counter += 1
        genome_2_connection_4.weight = 2
        genome_2_connection_4.enabled = True
        genome_2.connections[genome_2_connection_4_key] = genome_2_connection_4

        genome_2_connection_5_key = (2, 0)
        genome_2_connection_5 = genome_2.create_connection(
            config.genome_config, genome_2_connection_5_key[0], genome_2_connection_5_key[1], innovation_number_counter)
        innovation_number_counter += 1
        genome_2_connection_5.weight = 3
        genome_2_connection_5.enabled = True
        genome_2.connections[genome_2_connection_5_key] = genome_2_connection_5

        genome_2_connection_6_key = (3, 0)
        genome_2_connection_6 = genome_2.create_connection(
            config.genome_config, genome_2_connection_6_key[0], genome_2_connection_6_key[1], innovation_number_counter)
        innovation_number_counter += 1
        genome_2_connection_6.weight = 5
        genome_2_connection_6.enabled = True
        genome_2.connections[genome_2_connection_6_key] = genome_2_connection_6

        return [(1, genome_1), (2, genome_2)]

    return build_fixed_genomes


test_diversity_coefficient_data = [
    (
        # Shape: [base_learner_num, batch_size, 1]
        torch.tensor([
            [
                [5], [2]
            ],
            [
                [15], [1]
            ],
            [
                [5], [3]
            ],
            [
                [3], [6]
            ]
        ], dtype=torch.float32),
        # Shape: [batch_size, 1]
        torch.tensor([
            [22],
            [3.5]
        ], dtype=torch.float32)
    ),
    (
        # Shape: [base_learner_num, batch_size, 1]
        torch.tensor([
            [
                [1], [2], [3]
            ],
            [
                [4], [5], [6]
            ],
            [
                [7], [8], [9]
            ],
            [
                [10], [11], [12]
            ]
        ], dtype=torch.float32),
        # Shape: [batch_size, 1]
        torch.tensor([
            [11.25],
            [11.25],
            [11.25]
        ], dtype=torch.float32)
    )
]

test_get_dynamic_correlation_penalty_coefficient = [
    # In the form of (min_correlation_penalty_coefficient: float, max_correlation_penalty_coefficient: float, current_population_diversity: float, max_population_diversity: float, expected_correlation_penalty_coefficient: float)
    (0.1, 0.9, 0, 10.0, 0.9),
    (0.1, 0.9, 2.5, 10.0, 0.7),
    (0.1, 0.9, 5.0, 10.0, 0.5),
    (0.1, 0.9, 7.5, 10.0, 0.3),
    (0.1, 0.9, 10.0, 10.0, 0.1),
    (0.0, 1.0, 0, 10.0, 1.0),
    (0.0, 1.0, 2.5, 10.0, 0.75),
    (0.0, 1.0, 5.0, 10.0, 0.5),
    (0.0, 1.0, 7.5, 10.0, 0.25),
    (0.0, 1.0, 10.0, 10.0, 0.0)

]

test_get_genetic_distance_commutativity_genome_ids = [
    (
        0, 1
    )
]

test_get_weight_decay_data = [
    # Expect only 1 tuple (order matters)
    # Each element in the tuple corresponds to the weight decay of each genome in fixed_genomes, i.e. (genome_1_weight_decay, genome_2_weight_decay, ...)
    (
        # Formula: (10^2 / (1 + 10^2)) + (3^2 / (1 + 3^2)) + (20^2 / (1 + 20^2)) + (7^2 / (1 + 7^2))
        3.867605244,
        # Formula: (10^2 / (1 + 10^2)) + (1^2 / (1 + 1^2)) + (20^2 / (1 + 20^2)) + (2^2 / (1 + 2^2)) + (3^2 / (1 + 3^2)) + (5^2 / (1 + 5^2))
        5.149143706
    )
]

test_get_genome_fitness_data = [
    # In the form of (genome_loss: torch.Tensor, penalty: torch.Tensor, weight_decay: float, expected_genome_fitness: torch.Tensor)
    # NOTE: The negative sign has been applied to penalty already
    (
        torch.tensor([500], dtype=torch.float),
        torch.tensor([-20], dtype=torch.float),
        float(1),
        torch.tensor([-481], dtype=torch.float)
    ),
    (
        torch.tensor([100], dtype=torch.float),
        torch.tensor([-10], dtype=torch.float),
        float(120),
        torch.tensor([-210], dtype=torch.float)
    ),
    (
        torch.tensor([10], dtype=torch.float),
        torch.tensor([-7], dtype=torch.float),
        float(16),
        torch.tensor([-19], dtype=torch.float)
    ),
    (
        torch.tensor([0], dtype=torch.float),
        torch.tensor([-500], dtype=torch.float),
        float(0),
        torch.tensor([500], dtype=torch.float)
    )
]


@pytest.fixture
def fixed_table_genetic_distance() -> Table:
    """
    Data is based on fixed_genomes
    Table is in the form of table[(genome_1_id, genome_2_id)]

    :return: The instance of table_genetic_distance of fixed_genomes
    :rtype: Table
    """
    table_genetic_distance = Table()

    # I optain the value by running maths_helper.get_genetic_distance(genome_1=genome_1, genome_2=genome_2, config=config)
    table_genetic_distance[(1, 2)] = 19 / 12

    return table_genetic_distance


@pytest.fixture
def fixed_table_output_distance() -> Table:
    """
    Data is based on fixed_genomes

    Genome data: 
    1. genome_1
        - Input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            - True value: [17]
            - Prediction: [71]
        - Input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            - True value: [47]
            - Prediction: [74]
        - Input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            - True value: [157]
            - Prediction: [751]
    2. genome_1
        - Input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            - True value: [17]
            - Prediction: [7]
        - Input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            - True value: [47]
            - Prediction: [3]
        - Input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            - True value: [157]
            - Prediction: [701]

    :return: The instance of table_output_distance of fixed_genomes
    :rtype: Table
    """
    table_output_distance = Table()

    # I optain the value by calculating (1 / n) * sum(|output_distance_1 - output_distance_2|)
    table_output_distance[(1, 2)] = 185 / 3

    return table_output_distance


@pytest.fixture
def fixed_table_fitness() -> Table:
    """
    Data is based on fixed_genomes
    fitness = loss + negative_penalty + weight_decay

    genome_1: 
    loss = (1 / 3) * ((71 - 17)^2 + (74 - 47)^2 + (751 - 157)^2) = 118827
    negative_penalty = -1 * (1 / 3) * ((71 - 39)^2 + (74 - 38.5)^2 + (751 - 726)^2) = -969.75
    weight_decay = 10^2 / (1 + 10^2) + 3^2 / (1 + 3^2) + 20^2 / (1 + 20^2) + 7^2 / (1 + 7^2) = 7832094 / 2025050
    fitness = (471429 * 1012525 + 7832094 * 2) / 4050100

    genome_2: 
    loss = (1 / 3) * ((7 - 17)^2 + (3 - 47)^2 + (701 - 157)^2) = 99324
    negative_penalty = -1 * (1 / 3) * ((7 - 39)^2 + (3 - 38.5)^2 + (701 - 726)^2) = -969.75
    weight_decay = 10^2 / (1 + 10^2) + 2^2 / (1 + 2^2) + 1^2 / (1 + 1^2) + 3^2 / (1 + 3^2) + 20^2 / (1 + 20^2) + 5^2 / (1 + 5^2) = 54221822 / 10530260
    fitness = (393417 * 2632565 + 54221822) / 10530260

    :return: The instance of table_fitness of fixed_genomes
    :rtype: Table
    """
    table_fitness = Table()

    table_fitness[(1, 1)] = (471429 * 1012525 + 7832094 * 2) / 4050100
    table_fitness[(2, 2)] = (393417 * 2632565 + 54221822) / 10530260

    return table_fitness


@pytest.fixture
def get_population_diversity_data(config: neat.Config, fixed_genomes_factory: any, fixed_table_genetic_distance: Table, fixed_table_output_distance: Table, fixed_table_fitness: Table) -> tuple:
    """
    Return the testing data for testing get_population_diversity(...) in the form of (dummy_subpopulation_key, [(genome_id, genome)], table_genetic_distance, table_output_distance, table_fitness, expected_population_diversity)

    :param config: The configuration instance
    :type config: neat.Config
    :param fixed_genomes_factory: The function that returns a list of fixed genome instances
    :type fixed_genomes_factory: any
    :return: (dummy_subpopulation_key, [(genome_id, genome)], table_genetic_distance, table_output_distance, table_fitness, expected_population_diversity)
    :rtype: tuple
    """
    """
    Genome data: 
    1. fixed_genomes[0]
        - Input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            - True value: [17]
            - Prediction: [71]
        - Input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            - True value: [47]
            - Prediction: [74]
        - Input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            - True value: [157]
            - Prediction: [751]
    2. fixed_genomes[0]
        - Input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            - True value: [17]
            - Prediction: [7]
        - Input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            - True value: [47]
            - Prediction: [3]
        - Input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            - True value: [157]
            - Prediction: [701]
    """
    # It is meant to represent number of nodes in this subpopulation but we are not interested in this so we just set it to a random value
    # TODO: Maybe make sure fixed genomes genome_1 and genome_2 actually have the same number of nodes (for now, we expect the calculation to be correct)
    dummy_subpopulation_key = 1

    genomes = fixed_genomes_factory(config=config)

    # In this special case, fitness_1 + fitness_2 = 2 * fitness_average
    expected_subpopulation_diversity_1_2 = 19 / 12 * 185 / 3
    # k = 2 so SD = SD_{1,2}
    expected_subpopulation_diversity = expected_subpopulation_diversity_1_2

    # We only have 1 subpopulation
    expected_population_diversity = expected_subpopulation_diversity

    return (dummy_subpopulation_key, genomes, fixed_table_genetic_distance, fixed_table_output_distance, fixed_table_fitness, expected_population_diversity)


@pytest.mark.parametrize("y_predictions, expected_diversity_coefficient", test_diversity_coefficient_data)
def test_diversity_coefficient(y_predictions: torch.Tensor, expected_diversity_coefficient: torch.Tensor):
    actual_diversity_coefficient = maths_helper.diversity_coefficient(
        y_predictions=y_predictions)

    # NOTE: We do not use torch.equal() here because there might be floating-point precision issue with torch.mean()
    assert torch.allclose(expected_diversity_coefficient,
                          actual_diversity_coefficient), f"Expected {expected_diversity_coefficient} but received: {actual_diversity_coefficient}"


@pytest.mark.parametrize("min_correlation_penalty_coefficient, max_correlation_penalty_coefficient, current_population_diversity, max_population_diversity, expected_correlation_penalty_coefficient", test_get_dynamic_correlation_penalty_coefficient)
def test_get_dynamic_correlation_penalty_coefficient(min_correlation_penalty_coefficient: float, max_correlation_penalty_coefficient: float, current_population_diversity: float, max_population_diversity: float, expected_correlation_penalty_coefficient: float):
    """
    Test the correctness of the formula of dynamic_correlation_penalty_coefficient()
    """
    actual_correlation_penalty_coefficient = maths_helper.get_dynamic_correlation_penalty_coefficient(
        min_correlation_penalty_coefficient=min_correlation_penalty_coefficient,
        max_correlation_penalty_coefficient=max_correlation_penalty_coefficient,
        current_population_diversity=current_population_diversity,
        max_population_diversity=max_population_diversity
    )

    assert math.isclose(actual_correlation_penalty_coefficient,
                        expected_correlation_penalty_coefficient), f"Expected {expected_correlation_penalty_coefficient} but received: {actual_correlation_penalty_coefficient}"


@pytest.mark.parametrize("fixed_genome_weight_decays", test_get_weight_decay_data)
def test_get_weight_decay(config: neat.Config, fixed_genomes_factory: any, fixed_genome_weight_decays: tuple):
    fixed_genomes = fixed_genomes_factory(config=config)

    assert len(fixed_genomes) == len(
        fixed_genome_weight_decays), f"Invalid testing data (expect fixed_genomes and fixed_genomes_weight_decays have the same length): {len(fixed_genomes)} != {len(fixed_genome_weight_decays)}"

    for i in range(len(fixed_genomes)):
        genome_id = fixed_genomes[i][0]
        genome = fixed_genomes[i][1]
        expected_weight_decay = fixed_genome_weight_decays[i]

        actual_weight_decay = maths_helper.get_weight_decay(genome=genome)

        assert math.isclose(
            actual_weight_decay, expected_weight_decay), f"Expected {expected_weight_decay} for Genome {genome_id} but received: {actual_weight_decay}"


@pytest.mark.skip(reason="I trust neat-python's implementation")
def test_get_genetic_distance():
    """
    Test the correctness of genome fitness calculation
    """
    pass


@pytest.mark.parametrize("genome_1_id, genome_2_id", test_get_genetic_distance_commutativity_genome_ids)
def test_get_genetic_distance_commutativity(config: neat.Config, genome_factory: any, genome_1_id: NEATNCLGenome, genome_2_id: NEATNCLGenome):
    genome_1 = genome_factory(config=config, genome_id=genome_1_id)
    genome_2 = genome_factory(config=config, genome_id=genome_2_id)

    assert maths_helper.get_genetic_distance(genome_1=genome_1, genome_2=genome_2, config=config) == maths_helper.get_genetic_distance(
        genome_1=genome_2, genome_2=genome_1, config=config), f"Expected {maths_helper.get_genetic_distance(genome_1=genome_1, genome_2=genome_2, config=config)} == {maths_helper.get_genetic_distance(genome_1=genome_2, genome_2=genome_1, config=config)} but: {maths_helper.get_genetic_distance(genome_1=genome_1, genome_2=genome_2, config=config)} != {maths_helper.get_genetic_distance(genome_1=genome_2, genome_2=genome_1, config=config)}"


def test_get_average_genome_fitness(config: neat.Config, fixed_genomes_factory: any, fixed_table_fitness: Table):
    fixed_genomes = fixed_genomes_factory(config=config)

    """
    genome_1: 
    fitness = (471429 * 1012525 + 7832094 * 2) / 4050100

    genome_2: 
    fitness = (393417 * 2632565 + 54221822) / 10530260
    """
    expected_average_genome_fitness = (
        (471429 * 1012525 + 7832094 * 2) / 4050100 + (393417 * 2632565 + 54221822) / 10530260) / 2

    actual_average_genome_fitness = maths_helper.get_average_genome_fitness(
        genomes=fixed_genomes, table_fitness=fixed_table_fitness)

    assert actual_average_genome_fitness == expected_average_genome_fitness, f"Expected {expected_average_genome_fitness} but received: {actual_average_genome_fitness}"


def test_get_population_diversity(get_population_diversity_data: tuple):
    test_data = get_population_diversity_data

    # In the form of {total_node_num: (subpopulation_genomes, subpopulation_1_table_genetic_distance, subpopulation_1_table_output_distance, subpopulation_1_table_fitness)}
    subpopulation_datasets = {}
    # Dummy subpopulation key (representing total number of nodes in the genome)
    subpopulation_key = test_data[0]
    expected_population_diversity = test_data[-1]
    subpopulation_datasets[subpopulation_key] = tuple(test_data[1:-1])

    actual_population_diversity = maths_helper.get_population_diversity(
        subpopulation_datasets=subpopulation_datasets)

    assert math.isclose(actual_population_diversity,
                        expected_population_diversity), f"Expected {expected_population_diversity} but received: {actual_population_diversity}"


@pytest.mark.parametrize("genome_loss, penalty, weight_decay, expected_genome_fitness", test_get_genome_fitness_data)
def test_get_genome_fitness(genome_loss: torch.Tensor, penalty: torch.Tensor, weight_decay: float, expected_genome_fitness: torch.Tensor):
    actual_genome_fitness = maths_helper.get_genome_fitness(
        genome_loss=genome_loss,
        penalty=penalty,
        weight_decay=weight_decay
    )

    assert torch.allclose(expected_genome_fitness,
                          actual_genome_fitness), f"Expected {expected_genome_fitness} but received: {actual_genome_fitness}"


if __name__ == "__main__":
    # Below run all test cases on all test files
    # pytest.main()

    # Below run all test cases on this test file only
    pytest.main([__file__])
