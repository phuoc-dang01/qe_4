import argparse
from typing import Tuple


def create_evolution_args(
    exp_name: str = 'test_run',
    env_name: str = 'Walker-v0',
    pop_size: int = 20,
    structure_shape: Tuple[int, int] = (5, 5),
    max_evaluations: int = 100,
    num_cores: int = 4
) -> argparse.Namespace:
    """
    Create a default argument namespace for evolution.

    Parameters:
    - exp_name: Experiment name
    - env_name: EvoGym environment name
    - pop_size: Population size for NEAT
    - structure_shape: Shape of the robot structure
    - max_evaluations: Maximum number of evaluations
    - num_cores: Number of CPU cores to use for parallel evaluation

    Returns:
    - argparse.Namespace: Arguments object
    """
    return argparse.Namespace(
        exp_name=exp_name,
        env_name=env_name,
        pop_size=pop_size,
        structure_shape=structure_shape,
        max_evaluations=max_evaluations,
        num_cores=num_cores
    )


def add_evolution_args(parser: argparse.ArgumentParser) -> None:
    """
    Add evolution arguments to an existing argument parser.
    """
    evolution_parser = parser.add_argument_group('evolution arguments')

    evolution_parser.add_argument('--exp-name', type=str, default='test_run',
                                help='Experiment name')
    evolution_parser.add_argument('--env-name', type=str, default='Walker-v0',
                                help='EvoGym environment name')
    evolution_parser.add_argument('--pop-size', type=int, default=20,
                                help='Population size')
    evolution_parser.add_argument('--structure-shape', nargs=2, type=int, default=[5, 5],
                                help='Robot structure shape (height width)')
    evolution_parser.add_argument('--max-evaluations', type=int, default=100,
                                help='Maximum number of evaluations')
    evolution_parser.add_argument('--num-cores', type=int, default=4,
                                help='Number of CPU cores for parallel evaluation')
