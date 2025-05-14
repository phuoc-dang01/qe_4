# t_args.py
import argparse
import os
from typing import List, Tuple


def add_ppo_args(parser: argparse.ArgumentParser) -> None:
    """
    Add PPO arguments with parameters adjusted for use with parallel evaluation
    """
    ppo_parser = parser.add_argument_group('ppo arguments')

    # Reduce resource usage for parallel runs
    ppo_parser.add_argument(
        '--verbose-ppo', default=1, type=int,
        help='Verbosity level for PPO: 0 for no output, 1 for info, 2 for debug'
    )
    ppo_parser.add_argument(
        '--learning-rate', default=3e-4, type=float,
        help='Learning rate for PPO'
    )
    # Make n_steps and batch_size compatible
    ppo_parser.add_argument(
        '--n-steps', default=32, type=int,
        help='Rollout buffer size (n_steps * n_envs)'
    )
    ppo_parser.add_argument(
        '--batch-size', default=32, type=int,
        help='Mini-batch size for PPO - should be a factor of n_steps*n_envs'
    )
    ppo_parser.add_argument(
        '--n-epochs', default=5, type=int,
        help='Number of epochs when optimizing the PPO objective'
    )
    ppo_parser.add_argument(
        '--gamma', default=0.99, type=float,
        help='Discount factor'
    )
    ppo_parser.add_argument(
        '--gae-lambda', default=0.95, type=float,
        help='GAE lambda parameter'
    )
    ppo_parser.add_argument(
        '--vf-coef', default=0.5, type=float,
        help='Value function coefficient'
    )
    ppo_parser.add_argument(
        '--max-grad-norm', default=0.5, type=float,
        help='Maximum gradient norm'
    )
    ppo_parser.add_argument(
        '--ent-coef', default=0.01, type=float,
        help='Entropy coefficient'
    )
    ppo_parser.add_argument(
        '--clip-range', default=0.2, type=float,
        help='Clipping parameter'
    )
    # Reduce the total timesteps for faster evaluation
    ppo_parser.add_argument(
        '--total-timesteps', default=2000, type=int,
        help='Total number of timesteps for PPO'
    )
    ppo_parser.add_argument(
        '--log-interval', default=1, type=int,
        help='Log interval'
    )
    ppo_parser.add_argument(
        '--n-envs', default=1, type=int,
        help='Number of parallel environments'
    )
    ppo_parser.add_argument(
        '--n-eval-envs', default=1, type=int,
        help='Number of parallel environments for evaluation'
    )
    ppo_parser.add_argument(
        '--n-evals', default=1, type=int,
        help='Number of evaluation runs'
    )
    ppo_parser.add_argument(
        '--eval-interval', default=500, type=int,
        help='Steps between evaluations'
    )

def create_t_args(
    exp_name: str = 'test_run',
    env_name: str = 'Walker-v0',
    pop_size: int = 3,
    structure_shape: Tuple[int, int] = (5, 5),
    max_evaluations: int = 9,
    num_cores: int = 2
) -> argparse.Namespace:
    """
    Create a default argument namespace for testing purposes

    Parameters:
    - exp_name: Experiment name
    - env_name: EvoGym environment name (must be a valid name registered in evogym)
    - pop_size: Population size for NEAT
    - structure_shape: Shape of the robot structure
    - max_evaluations: Maximum number of evaluations
    - num_cores: Number of CPU cores to use for parallel evaluation

    Returns:
    - argparse.Namespace: Arguments object
    """
    parser = argparse.ArgumentParser()
    add_ppo_args(parser)

    # Add our specific args
    parser.add_argument('--exp-name', type=str, default=exp_name)
    parser.add_argument('--env-name', type=str, default=env_name)
    parser.add_argument('--pop-size', type=int, default=pop_size)
    parser.add_argument('--structure-shape', type=tuple, default=structure_shape)
    parser.add_argument('--max-evaluations', type=int, default=max_evaluations)
    parser.add_argument('--num-cores', type=int, default=num_cores)

    args = parser.parse_args([])  # Empty list to avoid reading command line args
    return args
