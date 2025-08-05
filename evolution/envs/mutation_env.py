# mutation_env.py

import random
from typing import List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from fitness.reward_const import RewardConst
from neat.genome import DefaultGenome

class NeatMutationEnv(gym.Env):
    """
    Gym environment for NEAT mutation RL (option-critic compatible).
    Action space: MultiDiscrete([n_mutations, max_secondary])
      - primary: mutation type
      - secondary: parameter (contextual per type)
    """
    ADD_NODE, DELETE_NODE, ADD_CONNECTION, DELETE_CONNECTION, MODIFY_WEIGHT, MODIFY_BIAS = range(6)

    def __init__(self, config, evaluator_type: str = "abc", reward_function: str = "improvement"):
        self.config = config
        self.evaluator_type = evaluator_type
        self.reward_function_type = reward_function
        self._setup_action_spaces()
        self._setup_observation_space()
        self._initialize_evaluator()
        self._setup_reward_function()
        self._reset_bookkeeping()

    def _initialize_evaluator(self):
        """Initialize evaluator based on type."""
        if self.evaluator_type == "dummy":
            from fitness.base_evaluator import DummyEvaluator
            self.evaluator = DummyEvaluator(self.config)
        elif self.evaluator_type == "proxy":
            from fitness.base_evaluator import ProxyEvaluator
            self.evaluator = ProxyEvaluator(self.config)
        else:
            from fitness.evaluator import FitnessEvaluator
            self.evaluator = FitnessEvaluator(self.config)

    def _setup_reward_function(self):
        from fitness.reward_functions import ImprovementReward, NoveltyReward

        if self.reward_function_type == "improvement":
            self.reward_func = ImprovementReward()
        elif self.reward_function_type == "novelty":
            self.reward_func = NoveltyReward()
        else:
            self.reward_func = ImprovementReward()

    def calculate_reward(self, current_fitness, prev_fitness):
        info = {
            'num_nodes': len(self.genome.nodes),
            'num_connections': len(self.genome.connections),
            'step': self.steps
        }
        return self.reward_func.calculate_reward(current_fitness, prev_fitness, info)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_bookkeeping()
        self.prev_fitness = self._evaluate_genome()
        return self._get_observation(), {}

    def set_evaluator_type(self, evaluator_type: str):
        self.evaluator_type = evaluator_type
        self._initialize_evaluator()

    def _evaluate_genome(self):
        try:
            # Centralized fitness evaluation (standalone)
            return self.evaluator.evaluate(self.genome)
        except Exception as e:
            print(f"Warning: Evaluation failed: {e}")
            return RewardConst.INVALID_ROBOT

    def _setup_action_spaces(self):
        # Number of mutation types
        self.n_mutations = 6
        # These are set per type (see usage in step)
        self.activation_functions: List = list(getattr(self.config.genome_config, 'activation_options', ['sigmoid', 'tanh', 'abs', 'gauss', 'identity', 'sin', 'relu']))
        self.action_space_activation   = spaces.Discrete(len(self.activation_functions))
        self.action_space_node_select  = spaces.Discrete(5)
        self.action_space_conn_select  = spaces.Discrete(5)
        self.action_space_weight_change= spaces.Discrete(5)

        self.action_spaces_consequences = [
            self.action_space_activation,    # add_node
            self.action_space_node_select,   # delete_node
            self.action_space_conn_select,   # add_connection
            self.action_space_conn_select,   # delete_connection
            self.action_space_weight_change, # modify_weight
            self.action_space_weight_change  # modify_bias
        ]
        max_secondary = max(s.n for s in self.action_spaces_consequences)
        self.action_space = spaces.MultiDiscrete([self.n_mutations, max_secondary])

    @property
    def secondary_action_dims(self) -> List[int]:
        """How many buckets for each option."""
        return [s.n for s in self.action_spaces_consequences]

    def _setup_observation_space(self):
        # Example: [num_nodes, num_conns, enabled_frac, avg_weight, 4x dummy hist]
        obs_dim = 2 + 2 + 4
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)

    def _reset_bookkeeping(self):
        self.fitness_history = []
        self.prev_fitness = 0.0
        self.steps = 0
        self.max_steps = 50
        self.genome = DefaultGenome(0)
        self.genome.configure_new(self.config.genome_config)

    def reset_for_new_genome(self, genome):
        print(f"[NEAT_ENV] Resetting for new genome {getattr(genome, 'key', 'unknown')}")
        self.genome = genome
        self.steps = 0
        self.prev_fitness = 0.0
        self.fitness_history = []
        return self._get_observation()

    def step(self, action):
        primary, secondary = int(action[0]), int(action[1])
        print(f"[NEAT_ENV] Mutation: primary={primary}, secondary={secondary}")

        cfg = self.config.genome_config
        inputs, outputs = list(getattr(cfg, 'input_keys', [])), list(getattr(cfg, 'output_keys', []))
        pre_fitness = self.prev_fitness
        pre_nodes = len(self.genome.nodes)
        pre_conns = len(self.genome.connections)
        mutation_successful = False

        # -- Main mutation logic (context-sensitive secondary) --
        if primary == self.ADD_NODE:
            act_fn = self.activation_functions[secondary] if secondary < len(self.activation_functions) else self.activation_functions[0]
            old_node_count = len(self.genome.nodes)
            try:
                self.genome.mutate_add_node(cfg)
                newest = max(self.genome.nodes) if self.genome.nodes else None
                if newest and newest not in inputs + outputs:
                    self.genome.nodes[newest].activation = act_fn
                mutation_successful = len(self.genome.nodes) > old_node_count
            except Exception as e:
                print(f"[NEAT_ENV] Add node mutation failed: {e}")
                mutation_successful = False

        elif primary == self.DELETE_NODE:
            old_node_count = len(self.genome.nodes)
            try:
                self.genome.mutate_delete_node(cfg)
                mutation_successful = len(self.genome.nodes) < old_node_count
            except Exception as e:
                print(f"[NEAT_ENV] Delete node mutation failed: {e}")
                mutation_successful = False

        elif primary == self.ADD_CONNECTION:
            old_conn_count = len(self.genome.connections)
            try:
                self.genome.mutate_add_connection(cfg)
                mutation_successful = len(self.genome.connections) > old_conn_count
            except Exception as e:
                print(f"[NEAT_ENV] Add connection failed: {e}")
                mutation_successful = False

        elif primary == self.DELETE_CONNECTION:
            old_conn_count = len(self.genome.connections)
            try:
                self.genome.mutate_delete_connection()
                mutation_successful = len(self.genome.connections) < old_conn_count
            except Exception as e:
                print(f"[NEAT_ENV] Delete connection mutation failed: {e}")
                mutation_successful = False

        elif primary == self.MODIFY_WEIGHT:
            conns = list(self.genome.connections.values())
            if conns:
                num_conns = len(conns)
                conn_index = secondary % num_conns
                cg = conns[conn_index]
                old_weight = cg.weight
                if random.random() < getattr(cfg, 'weight_mutate_rate', 1.0):
                    if random.random() < getattr(cfg, 'weight_replace_rate', 0.5):
                        cg.weight = random.gauss(getattr(cfg, 'weight_init_mean', 0.0), getattr(cfg, 'weight_init_stdev', 1.0))
                    else:
                        cg.weight += random.gauss(0.0, 1.0) * getattr(cfg, 'weight_mutate_power', 0.5)
                    cg.weight = max(getattr(cfg, 'weight_min_value', -5.0), min(cg.weight, getattr(cfg, 'weight_max_value', 5.0)))
                    mutation_successful = abs(cg.weight - old_weight) > 1e-6

        elif primary == self.MODIFY_BIAS:
            nodes = list(self.genome.nodes.values())
            if nodes:
                num_nodes = len(nodes)
                node_index = secondary % num_nodes
                ng = nodes[node_index]
                old_bias = ng.bias
                if random.random() < getattr(cfg, 'bias_mutate_rate', 1.0):
                    if random.random() < getattr(cfg, 'bias_replace_rate', 0.5):
                        ng.bias = random.gauss(getattr(cfg, 'bias_init_mean', 0.0), getattr(cfg, 'bias_init_stdev', 1.0))
                    else:
                        ng.bias += random.gauss(0.0, 1.0) * getattr(cfg, 'bias_mutate_power', 0.5)
                    ng.bias = max(getattr(cfg, 'bias_min_value', -5.0), min(ng.bias, getattr(cfg, 'bias_max_value', 5.0)))
                    mutation_successful = abs(ng.bias - old_bias) > 1e-6

        # -- Evaluate --
        current_fitness = self._evaluate_genome()
        print(f"[NEAT_ENV] Pre-mutation fitness: {pre_fitness:.4f}")
        print(f"[NEAT_ENV] Post-mutation fitness: {current_fitness:.4f}")
        print(f"[NEAT_ENV] Mutation successful: {mutation_successful}")

        reward, terminated = self.calculate_reward(current_fitness, pre_fitness)
        if mutation_successful and reward > 0:
            reward += 0.1
        elif mutation_successful and reward <= 0:
            reward -= 0.05
        elif not mutation_successful:
            reward -= 0.1

        print(f"[NEAT_ENV] Final reward: {reward:.4f}, terminated: {terminated}")

        self.prev_fitness = current_fitness
        self.fitness_history.append(current_fitness)
        self.steps += 1
        truncated = (self.steps >= self.max_steps)

        info = {
            'mutation_type': primary,
            'parameter_bucket': secondary,
            'num_nodes': len(self.genome.nodes),
            'num_connections': len(self.genome.connections),
            'fitness': current_fitness,
            'fitness_improvement': current_fitness - pre_fitness if current_fitness > RewardConst.INVALID_ROBOT else 0,
            'mutation_successful': mutation_successful,
            'reward': reward,
            'pre_fitness': pre_fitness,
            'post_fitness': current_fitness,
            'step': self.steps
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        features = []
        num_nodes = len(self.genome.nodes)
        num_connections = len(self.genome.connections)
        features.extend([num_nodes, num_connections])

        if num_connections > 0:
            enabled_connections = sum(1 for conn in self.genome.connections.values() if getattr(conn, 'enabled', True))
            avg_weight = sum(conn.weight for conn in self.genome.connections.values()) / num_connections
            features.extend([enabled_connections / num_connections, avg_weight])
        else:
            features.extend([0, 0])

        features.extend([0, 0, 0, 0])  # Placeholder for fitness history

        return np.array(features, dtype=np.float32)
