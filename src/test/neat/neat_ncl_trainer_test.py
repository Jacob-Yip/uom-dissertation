import logging
import neat
import os
import pytest
from src.model.neat_ncl_base_learner import NEATNCLBaseLearner
from src.neat import neat_ncl_trainer
from src.neat.innovation_tracker import InnovationTracker
from src.neat.neat_ncl_genome import NEATNCLGenome
import torch
from torch import nn

logger = logging.getLogger(__name__)

"""
Run: python -m pytest -s --log-cli-level=INFO src/test/neat/neat_ncl_trainer_test.py
"""
# Constant
# Absolute path to repository storing NEAT NCL base learners' genomes
MODEL_REPOSITORY_PATH = os.path.join(os.path.dirname(
    __file__), os.path.join("../../../data/test/model", "neat"))


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


test_tables_genetic_distance_output_distance_fitness_genome_ids = [
    (
        [i for i in range(150)],
        torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], dtype=torch.float)
    )
]

# In the form of (genome_ids (length must be equal to base_learner_num), train_epoch, base_learner_num, data_train (in the form of (data_train_X, data_train_y)), loss_function, correlation_penalty_coefficient, learning_rate)
test_ncl_update_data = [
    (
        [0, 1, 2, 3],
        3,
        4,
        # NOTE: Must not be set to 0 -> a genome with no connections gives 0 as a prediction and our test case fails
        (torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
         dtype=torch.float), torch.tensor([[100]], dtype=torch.float)),
        nn.MSELoss(),
        0.5,
        # Make the learning rate very big so that change in weights are visible -> this will fail the test case though because the model becomes inaccurate after training but it does show you backpropagation is working
        0.1
    )
]


@pytest.mark.parametrize("genome_ids, train_epoch, base_learner_num, data_train, loss_function, correlation_penalty_function, learning_rate", test_ncl_update_data)
def test_ncl_update(config: neat.Config, genome_factory: any, genome_ids: list, train_epoch: int, base_learner_num: int, data_train: tuple, loss_function: any, correlation_penalty_function: float, learning_rate: float):
    """
    Test that ncl_update does update the weights of base learners

    NOTE: I logged it and weights of base learners do change
        - It's just the changes of weights are super small and they are not reflected by any loss change due to precision problem
    """
    genomes = []  # In the form of [(genome_id, genome_instance)]
    last_losses = {}  # In the form of {genome_id: last_loss}

    for genome_id in genome_ids:
        # In the form of [(genome_id, genome_instance)]
        genomes.append((genome_id, genome_factory(
            config=config, genome_id=genome_id)))

    # Compute loss of each base learner
    for genome_id, genome in genomes:
        model = NEATNCLBaseLearner(
            genome=genome,
            config=config,
            genome_id=genome_id,
            model_repository_path=MODEL_REPOSITORY_PATH
        )
        model.train()

        y_prediction = model(data_train[0])

        loss = loss_function(y_prediction, data_train[1]).item()

        last_losses[genome_id] = loss

    logger.info(f"Original loss: {last_losses}")

    for epoch in range(train_epoch):
        logger.info(f"[Epoch {epoch + 1}]")

        neat_ncl_trainer.ncl_update(
            genomes=genomes,
            config=config,
            model_repository_path=MODEL_REPOSITORY_PATH,
            data_train=data_train,
            loss_function=loss_function,
            correlation_penalty_coefficient=correlation_penalty_function,
            learning_rate=learning_rate
        )

        # Load genomes
        genomes = neat_ncl_trainer.load_genomes(
            original_genomes=genomes, model_repository_path=MODEL_REPOSITORY_PATH)

        # Compute loss of each base learner
        for genome_id, genome in genomes:
            model = NEATNCLBaseLearner(
                genome=genome,
                config=config,
                genome_id=genome_id,
                model_repository_path=MODEL_REPOSITORY_PATH
            )
            model.train()

            y_prediction = model(data_train[0])

            logger.info(
                f"[Epoch {epoch + 1}] y_prediction from genome {genome_id}: {y_prediction}")

            current_loss = loss_function(y_prediction, data_train[1]).item()

            assert current_loss <= last_losses[
                genome_id], f"Expect current_loss <= last_loss for genome_{genome_id}: {current_loss} > {last_losses[genome_id]}"

        # Log
        # NOTE: Can be removed if you think it's too much logging
        logger.info(f"[Epoch {epoch + 1}] Loss: {last_losses}")
        logger.info(f"[Epoch {epoch + 1}]")


@pytest.mark.skip(reason="Have not implemented yet")
def test_load_genomes():
    """
    Test the correctness of loading genome
    """
    # TODO: Update
    pass


@pytest.mark.skip(reason="Have not implemented yet")
def test_dynamic_correlation_penalty_coefficient_computation():
    """
    Test the whole process of calculating the dynamic correlation penalty coefficient
    Input is list of genomes
    """
    # TODO: Update
    pass


@pytest.mark.parametrize("genome_ids, data_X", test_tables_genetic_distance_output_distance_fitness_genome_ids)
def test_tables_genetic_distance_output_distance_fitness(config: neat.Config, genome_factory: any, genome_ids: list, data_X: torch.Tensor):
    genomes = []  # In the form of [(genome_id, genome_instance)]

    for genome_id in genome_ids:
        # In the form of [(genome_id, genome_instance)]
        genomes.append((genome_id, genome_factory(
            config=config, genome_id=genome_id)))

    table_genetic_distance, table_output_distance, table_fitness = neat_ncl_trainer.get_tables_genetic_distance_output_distance_fitness(
        genomes=genomes,
        config=config,
        data_X=data_X
    )

    for genome_1_id, genome_1 in genomes:
        for genome_2_id, genome_2 in genomes:
            if genome_1_id == genome_2_id:
                continue

            assert table_genetic_distance.has_key(
                (genome_1_id, genome_2_id)), f"Expect table_genetic_distance to have key ({genome_1_id}, {genome_2_id}): {table_genetic_distance.has_key((genome_1_id, genome_2_id))}"
            assert table_genetic_distance.has_key(
                (genome_2_id, genome_1_id)), f"Expect table_genetic_distance to have key ({genome_2_id}, {genome_1_id}): {table_genetic_distance.has_key((genome_2_id, genome_1_id))}"

            assert table_output_distance.has_key(
                (genome_1_id, genome_2_id)), f"Expect table_output_distance to have key ({genome_1_id}, {genome_2_id}): {table_output_distance.has_key((genome_1_id, genome_2_id))}"
            assert table_output_distance.has_key(
                (genome_2_id, genome_1_id)), f"Expect table_output_distance to have key ({genome_2_id}, {genome_1_id}): {table_output_distance.has_key((genome_2_id, genome_1_id))}"

        assert table_fitness.has_key(
            (genome_1_id, genome_1_id)), f"Expect table_fitness to have key ({genome_1_id}, {genome_1_id}): {table_fitness.has_key((genome_1_id, genome_1_id))}"

# TODO: Make sure every method has at least 1 test case


if __name__ == "__main__":
    # Below run all test cases on all test files
    # pytest.main()

    # Below run all test cases on this test file only
    pytest.main([__file__])
