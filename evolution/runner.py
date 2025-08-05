# evolution/runner.py
import argparse
import os
import pdb
import shutil
import sys
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Define current directory
curr_dir = os.path.dirname(os.path.abspath(__file__))

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add evogym paths if it exists as a subdirectory
evogym_path = PROJECT_ROOT / 'evogym'
if evogym_path.exists():
    sys.path.insert(0, str(evogym_path / 'examples'))
    sys.path.insert(0, str(evogym_path / 'examples' / 'externals' / 'PyTorch-NEAT'))

import neat
import numpy as np
import torch

# Import from project root
from ppo_evaluation_args import create_ppo_eval_args
from pytorch_neat.cppn import create_cppn

from evogym import get_full_connectivity, has_actuator, hashable, is_connected
from evogym.envs import *
import evogym.envs

# Import from local evolution module
from .parallel import ParallelEvaluator
from .population import Population
from .ppo_eval import run_ppo


class CPPNRobotGenerator:
    @staticmethod
    def get_cppn_input(
        structure_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = torch.meshgrid(
            torch.arange(structure_shape[0]),
            torch.arange(structure_shape[1]),
            indexing="ij"
        )
        x, y = x.flatten(), y.flatten()
        center = (np.array(structure_shape) - 1) / 2
        d = ((x - center[0]) ** 2 + (y - center[1]) ** 2).sqrt()
        return x, y, d

    @staticmethod
    def get_robot_from_genome(
        genome: neat.DefaultGenome,
        config: neat.Config
    ) -> np.ndarray:
        nodes = create_cppn(
            genome,
            config,
            leaf_names=["x", "y", "d"],
            node_names=["empty", "rigid", "soft", "hori", "vert"],
        )
        structure_shape = config.extra_info["structure_shape"]
        x, y, d = CPPNRobotGenerator.get_cppn_input(structure_shape)
        material = []
        for node in nodes:
            material.append(node(x=x, y=y, d=d).numpy())
        material = np.vstack(material).argmax(axis=0)
        return material.reshape(structure_shape)



# Function to create fresh PPO args
def get_fresh_ppo_args():
    return create_ppo_eval_args()

def get_robot_from_genome(genome, config):
    """Helper function for backward compatibility."""
    return CPPNRobotGenerator.get_robot_from_genome(genome, config)


def eval_genome_constraint(genome, config, genome_id, generation):
    """Check if genome produces valid and unique robot structure."""
    try:
        # Generate robot from genome
        robot = CPPNRobotGenerator.get_robot_from_genome(genome, config)

        # Check basic validity
        if not is_connected(robot):
            print(f"   [INVALID] Robot not connected (genome {genome_id})")
            return False

        if not has_actuator(robot):
            print(f"   [INVALID] Robot has no actuators (genome {genome_id})")
            return False

        # Check uniqueness
        robot_hash = hashable(robot)
        if robot_hash in config.extra_info["structure_hashes"]:
            print(f"   [DUPLICATE] Structure already exists (genome {genome_id})")
            return False

        # Mark this structure as seen
        config.extra_info["structure_hashes"][robot_hash] = True
        return True

    except Exception as e:
        print(f"   [ERROR] Constraint check failed for genome {genome_id}: {str(e)}")
        return False


def eval_fitness(genome, config, genome_id, generation):
    """Evaluate genome fitness using PPO. Assumes genome passed constraint check."""

    # Ensure clean process environment
    import os
    if mp.current_process().name != 'MainProcess':
        # We're in a worker process
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'

        import torch
        torch.set_num_threads(1)

    process_id = os.getpid()
    print(f"Evaluating genome {genome_id} on process {process_id} for generation {generation}")

    try:
        # Generate robot from genome
        robot = CPPNRobotGenerator.get_robot_from_genome(genome, config)
        print(f"Robot shape: {robot.shape}")

        # Get configuration - create fresh args for each process
        if 'get_ppo_args' in config.extra_info:
            args = config.extra_info['get_ppo_args']()
        else:
            # Fallback for compatibility
            args = config.extra_info['args']

        env_name = config.extra_info["env_name"]
        connectivity = get_full_connectivity(robot)

        # Set up save paths
        save_path_generation = os.path.join(config.extra_info['save_path'], f'generation_{generation}')
        save_path_structure = os.path.join(save_path_generation, 'structure', f'{genome_id}')
        save_path_controller = os.path.join(save_path_generation, 'controller')

        # Ensure directories exist
        os.makedirs(os.path.dirname(save_path_structure), exist_ok=True)
        os.makedirs(save_path_controller, exist_ok=True)

        # Save structure
        np.savez(save_path_structure, robot, connectivity)

        # Evaluate fitness with PPO
        fitness = run_ppo(
            args,
            robot,
            env_name,
            save_path_controller,
            str(genome_id),
            connectivity
        )

        # Ensure fitness is a valid number
        if not isinstance(fitness, (int, float)) or not np.isfinite(fitness):
            print(f"   [WARNING] Invalid fitness returned: {fitness}, using -1.0")
            fitness = -1.0

        print(f"Genome {genome_id} evaluation completed with fitness: {fitness}")
        return fitness

    except Exception as e:
        print(f"   [ERROR] Exception in eval_fitness for genome {genome_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return -1.0


class SaveResultReporter(neat.BaseReporter):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.generation = None

    def start_generation(self, generation):
        self.generation = generation
        save_path_structure = os.path.join(self.save_path, f'generation_{generation}', 'structure')
        save_path_controller = os.path.join(self.save_path, f'generation_{generation}', 'controller')
        os.makedirs(save_path_structure, exist_ok=True)
        os.makedirs(save_path_controller, exist_ok=True)

    def post_evaluate(self, config, population, species, best_genome):
        save_path_ranking = os.path.join(self.save_path, f'generation_{self.generation}', 'output.txt')
        genome_id_list, genome_list = np.arange(len(population)), np.array(list(population.values()))
        sorted_idx = sorted(genome_id_list, key=lambda i: genome_list[i].fitness, reverse=True)
        genome_id_list, genome_list = list(genome_id_list[sorted_idx]), list(genome_list[sorted_idx])
        with open(save_path_ranking, 'w') as f:
            out = ''
            for genome_id, genome in zip(genome_id_list, genome_list):
                out += f'{genome_id}\t\t{genome.fitness}\n'
            f.write(out)


def run_evolution(
    args: argparse.Namespace,
    use_adaptive: bool = True,
    transfer_policy: Optional[str] = None,
    selection_strategy: str = 'epsilon_greedy',
    exploration_rate: float = 0.3
):
    """
    Run NEAT evolution with optional adaptive mutation strategies.

    Args:
        args: Evolution configuration arguments
        use_adaptive: Whether to use adaptive mutation strategies
        transfer_policy: Path to saved mutation policy for transfer learning
        selection_strategy: Mutation selection strategy ('epsilon_greedy', 'softmax', 'ucb')
        exploration_rate: Initial exploration rate for mutations

    Returns:
        Tuple of (best_robot, best_fitness, final_population)
    """
    exp_name, env_name, pop_size, structure_shape, max_evaluations, num_cores = (
        args.exp_name,
        args.env_name,
        args.pop_size,
        args.structure_shape,
        args.max_evaluations,
        args.num_cores,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join("saved_data", f"{args.exp_name}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    save_path_metadata = os.path.join(save_path, 'metadata.txt')
    with open(save_path_metadata, 'w') as f:
        f.write(f'POP_SIZE: {pop_size}\n'
                f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n'
                f'MAX_EVALUATIONS: {max_evaluations}\n'
                f'USE_ADAPTIVE: {use_adaptive}\n'
                f'SELECTION_STRATEGY: {selection_strategy}\n'
                f'EXPLORATION_RATE: {exploration_rate}\n')
        if transfer_policy:
            f.write(f'TRANSFER_POLICY: {transfer_policy}\n')

    structure_hashes = {}

    config_path = os.path.join(curr_dir, 'neat.cfg')

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
        extra_info={
            'structure_shape': structure_shape,
            'save_path': save_path,
            'structure_hashes': structure_hashes,
            'args': create_ppo_eval_args(),  # args for run_ppo
            'get_ppo_args': get_fresh_ppo_args,  # Function to create fresh args
            'env_name': env_name,
        },
        custom_config=[
            ('NEAT', 'pop_size', pop_size),
        ],
    )

    # Create population with adaptive mutations
    pop = Population(config, initial_policy=transfer_policy, use_adaptive=use_adaptive)

    # Configure adaptive reproduction if enabled
    if use_adaptive and hasattr(pop.reproduction, 'selection_strategy'):
        pop.reproduction.selection_strategy = selection_strategy
        pop.reproduction.exploration_rate = exploration_rate

    # Add reporters
    reporters = [
        neat.StatisticsReporter(),
        neat.StdOutReporter(True),
        SaveResultReporter(save_path),
    ]
    for reporter in reporters:
        pop.add_reporter(reporter)

    # Create parallel evaluator
    evaluator = ParallelEvaluator(
        num_workers=num_cores,
        fitness_function=eval_fitness,
        constraint_function=eval_genome_constraint,
        timeout=600
    )

    # Run evolution
    pop.run(
        evaluator.evaluate_fitness,
        evaluator.evaluate_constraint,
        n=int(np.ceil(max_evaluations / pop_size))
    )

    # Get best robot
    best_robot = get_robot_from_genome(pop.best_genome, config)
    best_fitness = pop.best_genome.fitness

    # Save final mutation policy if using adaptive
    if use_adaptive:
        policy_path = os.path.join(save_path, 'final_mutation_policy.pkl')
        pop.save_mutation_policy(policy_path)
        print(f"\nFinal mutation policy saved to: {policy_path}")

    return best_robot, best_fitness, pop
