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


class ABTestingReproduction(neat.DefaultReproduction):
    """
    A reproduction class that implements A/B testing between RL-guided
    mutations and standard NEAT mutations.
    """

    def __init__(self, config, reporters, stagnation, rl_policy_path=None,
                 ab_ratio=0.5, tracking_enabled=True, parent_population=None):
        """Initialize the A/B testing reproduction module."""
        super().__init__(config, reporters, stagnation)
        self.parent_population = parent_population
        self.rl_policy_path = rl_policy_path
        self.ab_ratio = ab_ratio
        self.tracking_enabled = tracking_enabled
        self.logger = setup_logger('ab_testing')

        # Load the RL policy
        self.rl_model, self.eval_env = self.load_rl_policy(self.rl_policy_path)
        self.rl_policy = self.rl_model.policy

        # Setup tracking dictionaries
        self.genome_group = {}  # Maps genome_id -> group ('rl' or 'standard')
        self.stats = {
            'rl': {
                'fitness': [],
                'complexity': [],
                'mutation_stats': defaultdict(int),
                'survival_rate': [],
            },
            'standard': {
                'fitness': [],
                'complexity': [],
                'mutation_stats': defaultdict(int),
                'survival_rate': [],
            }
        }

        # Setup the activation functions
        self.activation_functions = ["sigmoid", "tanh", "abs", "gauss", "identity", "sin", "relu"]

        self.logger.info(f"Initialized A/B Testing with ratio: {ab_ratio}")
        if rl_policy_path:
            self.logger.info(f"Using RL policy from: {rl_policy_path}")

        # Add generation counter
        self.generation = 0

    def load_rl_policy(self, policy_path):
        from test_policy import load_option_critic_model
        args = init_train_args()
        return load_option_critic_model(policy_path, args)

    def create_new(self, genome_type, genome_config, pop_size):
        """When NEAT wants a brand-new population (e.g., after extinction/constraints)."""
        new_pop = super().create_new(genome_type, genome_config, pop_size)
        # Tag every genome at first sight
        for gid in new_pop:
            self.genome_group[gid] = self._assign_random_group()
        return new_pop

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
        species_fitness = []
        remaining_species = []

        for stag_sid, stag_s, stagnant in self.stagnation.update(species, generation):
            if stagnant:
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                species_fitness.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)

        # ─── Gen-0 baseline ───
        if generation == 0:
            # overall average fitness across all individuals
            avg_fit = float(np.mean(species_fitness))
            baseline = (avg_fit, avg_fit, avg_fit)
            # record the same starting point for both arms
            # self.stats['rl']['fitness'].append(baseline)
            # self.stats['standard']['fitness'].append(baseline)

        if not remaining_species:
            species.species = {}
            return {}

        # Calculate adjusted fitness
        min_fitness = min(species_fitness)
        max_fitness = max(species_fitness)

        fitness_diff = max_fitness - min_fitness
        eps = 1e-8
        if not np.isfinite(fitness_diff) or fitness_diff < eps:
            fitness_range = eps
        else:
            fitness_range = fitness_diff

        fitness_range = max(1.0, max_fitness - min_fitness)

        for s in remaining_species:
            mean_fit = sum(m.fitness for m in s.members.values()) / len(s.members)
            s.adjusted_fitness = (mean_fit - min_fitness) / fitness_range

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
                    self.genome_group.setdefault(i,self._assign_random_group())

                    spawn -= 1

                    # Keep track of the group assignment
                    if i in self.genome_group:
                        group = self.genome_group[i]
                        self.logger.info(f"Elite {i} from group {group} survived")

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

                # Determine which group this child belongs to
                p1g = self.genome_group.get(parent1_id, self._assign_random_group())
                p2g = self.genome_group.get(parent2_id, self._assign_random_group())
                if p1g == p2g:
                    child_group = p1g
                else:
                    child_group = self._assign_random_group()
                self.genome_group[gid] = child_group

                # Apply mutations based on group
                if child_group == 'rl':
                    child = self._apply_rl_mutation(child, config.genome_config)
                else:
                    child.mutate(config.genome_config)

                # Record group assignment and ancestry
                self.ancestors[gid] = (parent1_id, parent2_id)

                # Add to new population
                new_population[gid] = child

                if self.tracking_enabled:
                    mutation_type = 'rl' if child_group == 'rl' else 'standard'
                    self.stats[child_group]['mutation_stats'][mutation_type] += 1

        self.logger.info(f"Reproduction complete. New population size: {len(new_population)}")

        # # Only collect group‐specific stats from gen 1 onward
        # if generation > 0 and self.tracking_enabled and self.parent_population is not None:
        #     self.parent_population._update_group_stats(generation, new_population)
        return new_population

    def _assign_random_group(self):
        """Randomly assign a genome to either 'rl' or 'standard' group based on ab_ratio."""
        return 'rl' if random.random() < self.ab_ratio else 'standard'

    def _apply_rl_mutation(self, genome, _genome_config):
        print(f"[RL_MUTATE] Genome: {genome.key}")

        # 1) Reset all the Gym wrappers (this also calls NeatMutationEnv.reset → _reset_bookkeeping)
        obs = self.eval_env.reset()

        # 2) Drill down to the raw NeatMutationEnv
        inner_vec = getattr(self.eval_env, 'venv', self.eval_env)
        wrapped = inner_vec.envs[0]
        base_env = wrapped.unwrapped

        # 3) Inject your genome and compute its initial fitness
        base_env.genome = genome
        base_env.prev_fitness = base_env._evaluate_genome()

        # 4) Now do N RL steps
        for step in range(5):
            actions, _ = self.rl_model.predict(obs, deterministic=True)
            print(f"[RL_MUTATE] Step {step+1}: action={actions[0]}")
            obs, rewards, dones, infos = self.eval_env.step(actions)
            print(f"    reward={rewards[0]:.4f}, done={dones[0]}")
            if dones[0]:
                break

        # 5) Return the mutated genome
        return base_env.genome

    def _get_observation(self, genome):
        """Extract features from the genome to feed into RL policy."""
        # Same feature extraction as used in the training environment
        env = NeatMutationEnv(self.parent_population.config)
        env._reset_bookkeeping()
        env.genome = genome
        return env._get_observation()

