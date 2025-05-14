# ab_testing/ab_reproduction.py
import logging
import os
import pdb
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import neat
import numpy as np
import torch
from env.neat_env import NeatMutationEnv
from neat.math_util import mean
from option_critic.algorithm import OptionCriticPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_option_critic import (create_option_critic_model, init_train_args,
                                 make_vec_env, setup_neat_config)


def setup_logger(name):
    """Set up logger with proper formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class TTestingReproduction(neat.DefaultReproduction):
    """
    A reproduction class that implements A/B testing between RL-guided
    mutations and standard NEAT mutations.
    """

    def __init__(self, config, reporters, stagnation, rl_policy_path=None):
        """Initialize the A/B testing reproduction module."""
        super().__init__(config, reporters, stagnation)
        self.rl_policy_path = rl_policy_path
        self.logger = setup_logger('ab_testing')
        # Load the RL policy
        self.rl_model, self.eval_env = self.load_rl_policy(self.rl_policy_path)
        self.rl_policy = self.rl_model.policy
        # Setup the activation functions
        self.activation_functions = ["sigmoid", "tanh", "abs", "gauss", "identity", "sin", "relu"]

        if rl_policy_path:
            self.logger.info(f"Using RL policy from: {rl_policy_path}")

        # Add generation counter
        self.generation = 0

    def load_rl_policy(self, policy_path):
        from test_policy import load_option_critic_model
        args = init_train_args()
        return load_option_critic_model(policy_path, args)


    def reproduce(self, config, species, pop_size, generation):
        """
        Reproduce using either RL-guided or standard mutations based on A/B testing.

        Args:
            config: NEAT configuration
            species: Current species
            pop_size: Population size
            generation: Current generation

        Returns:
            New population
        """
        # Call stagnation to get species fitness info
        self.logger.info(f"Starting reproduction for generation {generation}")
        all_fitnesses = []
        remaining_species = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)

        if not remaining_species:
            species.species = {}
            return {}

        # Calculate adjusted fitness
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)

        # MAGIC CLIP
        fitness_range = max(1.0, max_fitness - min_fitness)

        for afs in remaining_species:
            # Compute adjusted fitness.
            msf = mean([m.fitness for m in afs.members.values()])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = sum(adjusted_fitnesses) / len(adjusted_fitnesses)
        self.logger.info(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")

        # Calculate spawn amounts for each species
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = self.reproduction_config.min_species_size

        min_species_size = max(min_species_size, self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes, pop_size, min_species_size)

        # Create the new population
        new_population = {}
        species.species = {}
        # Handle reproduction for each species
        for spawn, s in zip(spawn_amounts, remaining_species):
            # Ensure at least elitism members survive
            spawn = max(spawn, self.reproduction_config.elitism)

            if spawn <= 0:
                continue

            # The species survives
            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s

            # Sort members by fitness (descending)
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation
            if self.reproduction_config.elitism > 0:
                for i, m in old_members[:self.reproduction_config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # Get parents for reproduction based on survival threshold
            repro_cutoff = int(len(old_members) * self.reproduction_config.survival_threshold)
            repro_cutoff = max(repro_cutoff, 2)  # At least 2 parents
            old_members = old_members[:repro_cutoff]

            # Create offspring
            while spawn > 0:
                spawn -= 1

                # Select parents
                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Create a new genome
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)

                child = self._apply_rl_mutation(child)
                # Record group assignment and ancestry
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        self.logger.info(f"Reproduction complete. New population size: {len(new_population)}")
        return new_population


    def _apply_rl_mutation(self, genome):
        print(f"[RL_MUTATE] Genome: {genome.key}")
        # Reset environment
        obs = self.eval_env.reset()
        inner_vec = getattr(self.eval_env, 'venv', self.eval_env)
        wrapped = inner_vec.envs[0]
        base_env = wrapped.unwrapped

        # Inject genome into environment
        base_env.genome = genome

        best_reward = float('-inf')
        best_genome = self._copy_genome(genome)

        # Perform RL-guided mutations
        for step in range(3):
            actions, _ = self.rl_model.predict(obs, deterministic=True)
            action = actions[0]
            print(f"[RL_MUTATE] Step {step+1}: action={actions[0]}")
            obs, rewards, dones, infos = self.eval_env.step(actions)

            # Check if this mutation result is better than previous best
            current_reward = rewards[0]
            print(f"    reward={current_reward:.4f}, done={dones[0]}")

            # If this is the best result so far, save it
            if current_reward > best_reward:
                best_reward = current_reward
                # Create a deep copy of the current genome
                best_genome = self._copy_genome(base_env.genome)

            if dones[0]:
                break

        # Return only the genome, not the tuple
        return best_genome

    def _copy_genome(self, genome):
        """Create a deep copy of a genome."""
        # This is a simplified implementation - you might need to adjust based on your genome class
        import copy
        return copy.deepcopy(genome)

