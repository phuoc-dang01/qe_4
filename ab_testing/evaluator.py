import os
import pdb
import sys
import time
from pathlib import Path

import neat
import numpy as np
from ab_testing.robot_gen import CPPNRobotGenerator

from evogym import get_full_connectivity, has_actuator, hashable, is_connected

# Get the project root (3 levels up from current file)
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent.parent

# Add paths to sys.path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'evogym' / 'examples' / 'ppo'))
sys.path.insert(0, str(PROJECT_ROOT / 'evogym' / 'examples' / 'externals' / 'PyTorch-NEAT'))

import multiprocessing as mp
from functools import partial

from ppo.run import run_ppo


class GenomeEvaluator:
    _rl_model = None
    _rl_model_path = None


    @staticmethod
    def eval_fitness(genome, config, genome_id, generation):
        robot = CPPNRobotGenerator.get_robot_from_genome(genome, config)
        print(robot)

        # Check for duplicates and cache
        robot_hash = hashable(robot)
        if robot_hash in config.extra_info["structure_hashes"]:
            # Re-use the original fitness
            prev_fitness = config.extra_info["structure_hashes"][robot_hash]
            print(f"   [SKIP] Duplicate structure (genome {genome_id}), reusing fitness {prev_fitness:.5f}")
            genome.fitness = prev_fitness
            return prev_fitness


        config.extra_info["structure_hashes"][robot_hash] = True
        args = config.extra_info["args"]

        connectivity = get_full_connectivity(robot)
        save_path_generation = os.path.join(config.extra_info['save_path'], f'generation_{generation}')
        save_path_structure = os.path.join(save_path_generation, 'structure', f'{genome_id}')
        save_path_controller = os.path.join(save_path_generation, 'controller')
        np.savez(save_path_structure, robot, connectivity)

        # Evaluate fitness with PPO
        fitness = run_ppo(
            args,
            robot,
            config.extra_info["env_name"],
            save_path_controller,
            str(genome_id),
            connectivity
        )
        genome.fitness = fitness
        config.extra_info["structure_hashes"][robot_hash] = fitness
        return fitness

    @staticmethod
    def batch_eval_fitness(genome_list, config, generation):
        """
        For each (genome_id, genome) in genome_list, call eval_fitness
        and assign genome.fitness accordingly.
        """
        for genome_id, genome in genome_list:
            # call the single‐genome evaluator
            f = GenomeEvaluator.eval_fitness(genome, config, genome_id, generation)
            genome.fitness = f

    @staticmethod
    def eval_constraint(genome, config, genome_id, generation):
        robot = CPPNRobotGenerator.get_robot_from_genome(genome, config)
        return is_connected(robot) and has_actuator(robot)

    @staticmethod
    def batch_eval_constraint(genome_list, config, generation):
        return [
            GenomeEvaluator.eval_constraint(genome, config, genome_id, generation)
            for genome_id, genome in genome_list
        ]

    @staticmethod
    def _eval_fitness_worker(data, config, generation):
        """Worker function for parallel evaluation"""
        genome_id, genome = data
        return GenomeEvaluator.eval_fitness(genome, config, genome_id, generation)
