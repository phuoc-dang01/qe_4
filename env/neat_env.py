import pdb
import random
from typing import List

import gymnasium as gym
import numpy as np
from fitness.reward_const import *
from gymnasium import spaces
from neat.genome import DefaultGenome


class NeatMutationEnv(gym.Env):
    ADD_NODE, DELETE_NODE, ADD_CONNECTION, DELETE_CONNECTION, MODIFY_WEIGHT, MODIFY_BIAS = range(6)

    def __init__(self, config):
        self.config = config
        from fitness.evaluator import FitnessEvaluator
        self.evaluator = FitnessEvaluator(self.config)
        self._setup_action_spaces()
        self._setup_observation_space()

    def _setup_action_spaces(self):
        # Primary choices
        self.n_mutations = 6
        # Consequence subspaces
        self.activation_functions: List = list(self.config.genome_config.activation_options)
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

        # Flatten to a fixed 2-vector: [primary, secondary]
        max_secondary = max(s.n for s in self.action_spaces_consequences)
        self.action_space = spaces.MultiDiscrete([self.n_mutations, max_secondary])

    @property
    def secondary_action_dims(self) -> List[int]:
        """
        Informs the policy how many buckets each option has,
        so it never samples an invalid index.
        """
        return [s.n for s in self.action_spaces_consequences]

    def _setup_observation_space(self):
        obs_dim = 2 + 2 + 4
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self._reset_bookkeeping()

    def _reset_bookkeeping(self):
        self.fitness_history = []
        self.prev_fitness = 0.0
        self.steps = 0
        self.max_steps = 50
        self.genome = DefaultGenome(0)
        self.genome.configure_new(self.config.genome_config)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create and configure genome
        self.fitness_history = []
        self.steps = 0

        self._reset_bookkeeping()
        self.prev_fitness = self._evaluate_genome()

        return self._get_observation(), {}

    def step(self, action):
        primary, secondary = int(action[0]), int(action[1])
        # print(f"Primary aciton: {primary}, secondary: {secondary}" )

        cfg = self.config.genome_config
        inputs, outputs = list(cfg.input_keys), list(cfg.output_keys)

        # Dispatch by primary choice
        if primary == self.ADD_NODE:
            if secondary < len(self.activation_functions):
                act_fn = self.activation_functions[secondary]
            else:
                # Default to a safe activation function if out of bounds
                act_fn = self.activation_functions[0]
            self.genome.mutate_add_node(cfg)
            # Assign activation to the new node
            newest = max(self.genome.nodes)
            if newest not in inputs + outputs:
                self.genome.nodes[newest].activation = act_fn

        elif primary == self.DELETE_NODE:
            self.genome.mutate_delete_node(cfg)

        elif primary == self.ADD_CONNECTION:
            try:
                self.genome.mutate_add_connection(cfg)
            except Exception as e:
                print(f"Activation function index error: {e}")

        elif primary == self.DELETE_CONNECTION:
            self.genome.mutate_delete_connection()

        elif primary == self.MODIFY_WEIGHT:
            conns = list(self.genome.connections.values())
            if conns:
                num_conns = len(conns)
                # Map to valid range with modulo
                conn_index = secondary % num_conns
                cg = conns[conn_index]

                # Apply mutation based on NEAT parameters
                if random.random() < self.config.genome_config.weight_mutate_rate:
                    if random.random() < self.config.genome_config.weight_replace_rate:
                        # Complete replacement
                        cg.weight = random.gauss(
                            self.config.genome_config.weight_init_mean,
                            self.config.genome_config.weight_init_stdev
                        )
                    else:
                        # Perturbation
                        cg.weight += random.gauss(0.0, 1.0) * self.config.genome_config.weight_mutate_power


                    # Ensure weight is within bounds
                    cg.weight = max(
                        self.config.genome_config.weight_min_value,
                        min(cg.weight, self.config.genome_config.weight_max_value)
                    )

        elif primary == self.MODIFY_BIAS:
            nodes = list(self.genome.nodes.values())
            if nodes:
                num_nodes = len(nodes)
                # Map to valid range with modulo
                node_index = secondary % num_nodes
                ng = nodes[node_index]

                # Apply bias mutation with similar logic
                if random.random() < self.config.genome_config.bias_mutate_rate:
                    if random.random() < self.config.genome_config.bias_replace_rate:
                        # Complete replacement
                        ng.bias = random.gauss(
                            self.config.genome_config.bias_init_mean,
                            self.config.genome_config.bias_init_stdev
                        )
                    else:
                        # Perturbation
                        ng.bias += random.gauss(0.0, 1.0) * self.config.genome_config.bias_mutate_power

                    # Ensure bias is within bounds
                    ng.bias = max(
                        self.config.genome_config.bias_min_value,
                        min(ng.bias, self.config.genome_config.bias_max_value)
                    )

        # Compute reward + termination
        current = self._evaluate_genome()
        reward, terminated = self.calculate_reward(current, self.prev_fitness)

        if len(self.fitness_history) >= 2:
            if current > self.fitness_history[-1] and self.fitness_history[-1] > self.fitness_history[-2]:
                reward += 0.25


        self.prev_fitness = current
        self.fitness_history.append(current)
        self.steps += 1

        terminated = terminated or (self.steps >= self.max_steps)

        if current <= RewardConst.INVALID_ROBOT:
            # enforce the hard penalty exactly, no further shaping
            reward = RewardConst.INVALID_ROBOT
            terminated = True
        else:
            reward = float(current - self.prev_fitness)
            terminated = False

        self.prev_fitness = current
        self.fitness_history.append(current)

        # counter
        self.steps += 1
        truncated  = (self.steps >= self.max_steps)

        info = {
            'mutation_type': primary,
            'parameter_bucket': secondary,
            'num_nodes': len(self.genome.nodes),
            'num_connections': len(self.genome.connections),
            'fitness': current,
            'fitness_improvement': reward,
            'enabled_ratio': sum(1 for c in self.genome.connections.values() if c.enabled) / max(1, len(self.genome.connections)),
            'genome_size': len(self.genome.nodes) + len(self.genome.connections),
            'step': self.steps
        }

        return self._get_observation(), reward, terminated, truncated, info

    def calculate_reward(self, current_fitness, prev_fitness):
        """Calculate reward based on fitness improvement with immediate bonuses."""
        # Handle invalid genomes
        if current_fitness <= RewardConst.INVALID_ROBOT:
            return RewardConst.INVALID_ROBOT, True

        # Calculate improvement
        improvement = current_fitness - prev_fitness

        if improvement > 0:
            return RewardConst.POS_REWARD, False
        else:
            return RewardConst.NEG_REWARD, False

    def _add_node_with_activation(self, activation_function):
        """Add a node with the specified activation function."""
        if not self.genome.connections:
            return False

        # Choose a random connection to split
        conn_to_split = random.choice(list(self.genome.connections.values()))
        new_node_id = self.config.genome_config.get_new_node_key(self.genome.nodes)

        # Create new node with selected activation
        ng = self.config.genome_type.create_node(self.config.genome_config, new_node_id)
        ng.activation = activation_function
        self.genome.nodes[new_node_id] = ng

        # Disable original connection and add two new ones
        conn_to_split.enabled = False
        i, o = conn_to_split.key
        self.genome.add_connection(self.config.genome_config, i, new_node_id, 1.0, True)
        self.genome.add_connection(self.config.genome_config, new_node_id, o, conn_to_split.weight, True)

        return True

    def _delete_node(self):
        """Delete a random non-input, non-output node."""
        if len(self.genome.nodes) <= self.config.genome_config.num_inputs + self.config.genome_config.num_outputs:
            return

        # Get non-input, non-output nodes
        input_keys = list(range(self.config.genome_config.num_inputs))
        output_keys = list(range(self.config.genome_config.num_outputs))
        hidden_keys = [k for k in self.genome.nodes.keys()
                      if k not in input_keys and k not in output_keys]

        if hidden_keys:
            # Choose random hidden node to remove
            node_key = random.choice(hidden_keys)
            self.genome.remove_node(node_key)

    def _evaluate_genome(self):
        return self.evaluator.evaluate(self.genome)

    def _get_observation(self):
        """Extract features from the genome."""
        # Example features (customize based on your needs):
        features = []

        # 1. Network structure features
        num_nodes = len(self.genome.nodes)
        num_connections = len(self.genome.connections)
        features.extend([num_nodes, num_connections])

        # 2. Connectivity features
        if num_connections > 0:
            enabled_connections = sum(1 for conn in self.genome.connections.values() if conn.enabled)
            avg_weight = sum(conn.weight for conn in self.genome.connections.values()) / num_connections
            features.extend([enabled_connections / num_connections, avg_weight])
        else:
            features.extend([0, 0])

        # 3. Recent fitness history (last 4 values)
        fitness_history = self.fitness_history[-4:] if self.fitness_history else []
        fitness_history = [0] * (4 - len(fitness_history)) + fitness_history
        features.extend(fitness_history)

        for i, feature in enumerate(features):
            if np.isnan(feature) or np.isinf(feature):
                print(f"Warning: Feature {i} is NaN or Inf, setting to 0")
                features[i] = 0.0

        return np.array(features, dtype=np.float32)
