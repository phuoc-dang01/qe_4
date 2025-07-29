import pdb
import random
from typing import List, Optional, Union

import gymnasium as gym
import numpy as np
from fitness.reward_const import *
from gymnasium import spaces
from neat.genome import DefaultGenome

class NeatMutationEnv(gym.Env):
    ADD_NODE, DELETE_NODE, ADD_CONNECTION, DELETE_CONNECTION, MODIFY_WEIGHT, MODIFY_BIAS = range(6)

    def __init__(self, config, evaluator_type: str = "full"):
        """
        Initialize NEAT mutation environment.

        Args:
            config: NEAT configuration
            evaluator_type: Type of evaluator to use
                - "dummy": Fast dummy evaluator for RL training
                - "proxy": ML-based fitness predictor
                - "full": Complete robot simulation (requires all dependencies)
        """
        self.config = config
        self._setup_action_spaces()
        self._setup_observation_space()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create and configure genome
        self.fitness_history = []
        self.steps = 0

        self._reset_bookkeeping()
        self.prev_fitness = self._evaluate_genome()

        return self._get_observation(), {}  # Always return (obs, info)

    def set_evaluator_type(self, evaluator_type: str):
        """Change evaluator type during runtime."""
        self.evaluator_type = evaluator_type
        self._initialize_evaluator()

    def _evaluate_genome(self):
        """Evaluate genome using the centralized eval_fitness method."""
        try:
            # Import the centralized evaluation function
            from testing.t_runner import eval_fitness

            # Use the same evaluation as the main evolution
            fitness = eval_fitness(
                self.genome,
                self.config,
                self.genome.key if hasattr(self.genome, 'key') else 0,
                0  # generation number, not critical for single evaluation
            )
            return fitness
        except Exception as e:
            print(f"Warning: Evaluation failed: {e}")
            return RewardConst.INVALID_ROBOT

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

    def reset_for_new_genome(self, genome):
        """Reset environment state for a new genome without full reset."""
        print(f"[NEAT_ENV] Resetting for new genome {getattr(genome, 'key', 'unknown')}")

        # Set the new genome
        self.genome = genome

        # Reset fitness tracking
        self.fitness_history = []
        self.prev_fitness = 0.0
        self.steps = 0

        # Get initial fitness
        try:
            self.prev_fitness = self._evaluate_genome()
            print(f"[NEAT_ENV] Initial fitness for genome: {self.prev_fitness:.4f}")
        except Exception as e:
            print(f"[NEAT_ENV] Failed to evaluate initial fitness: {e}")
            self.prev_fitness = RewardConst.INVALID_ROBOT

        return self._get_observation()

    def step(self, action):
        primary, secondary = int(action[0]), int(action[1])
        print(f"[NEAT_ENV] Mutation: primary={primary}, secondary={secondary}")

        cfg = self.config.genome_config
        inputs, outputs = list(cfg.input_keys), list(cfg.output_keys)

        # Store pre-mutation state for comparison
        pre_fitness = self.prev_fitness
        pre_nodes = len(self.genome.nodes)
        pre_conns = len(self.genome.connections)

        # Apply mutation based on primary action
        mutation_successful = False

        if primary == self.ADD_NODE:
            if secondary < len(self.activation_functions):
                act_fn = self.activation_functions[secondary]
            else:
                act_fn = self.activation_functions[0]

            old_node_count = len(self.genome.nodes)
            try:
                self.genome.mutate_add_node(cfg)
                newest = max(self.genome.nodes) if self.genome.nodes else None
                if newest and newest not in inputs + outputs:
                    self.genome.nodes[newest].activation = act_fn
                mutation_successful = len(self.genome.nodes) > old_node_count
            except (AssertionError, KeyError, ValueError) as e:
                print(f"[NEAT_ENV] Add node mutation failed: {e}")
                mutation_successful = False

        elif primary == self.DELETE_NODE:
            old_node_count = len(self.genome.nodes)
            try:
                self.genome.mutate_delete_node(cfg)
                mutation_successful = len(self.genome.nodes) < old_node_count
            except (AssertionError, KeyError, ValueError) as e:
                print(f"[NEAT_ENV] Delete node mutation failed: {e}")
                mutation_successful = False

        elif primary == self.ADD_CONNECTION:
            old_conn_count = len(self.genome.connections)
            try:
                self.genome.mutate_add_connection(cfg)
                mutation_successful = len(self.genome.connections) > old_conn_count
            except Exception as e:
                print(f"Add connection failed: {e}")
                mutation_successful = False

        elif primary == self.DELETE_CONNECTION:
            old_conn_count = len(self.genome.connections)
            try:
                self.genome.mutate_delete_connection()
                mutation_successful = len(self.genome.connections) < old_conn_count
            except (AssertionError, KeyError, ValueError) as e:
                print(f"[NEAT_ENV] Delete connection mutation failed: {e}")
                mutation_successful = False

        elif primary == self.MODIFY_WEIGHT:
            conns = list(self.genome.connections.values())
            if conns:
                num_conns = len(conns)
                conn_index = secondary % num_conns
                cg = conns[conn_index]

                old_weight = cg.weight
                if random.random() < cfg.weight_mutate_rate:
                    if random.random() < cfg.weight_replace_rate:
                        cg.weight = random.gauss(cfg.weight_init_mean, cfg.weight_init_stdev)
                    else:
                        cg.weight += random.gauss(0.0, 1.0) * cfg.weight_mutate_power

                    cg.weight = max(cfg.weight_min_value, min(cg.weight, cfg.weight_max_value))
                    mutation_successful = abs(cg.weight - old_weight) > 1e-6

        elif primary == self.MODIFY_BIAS:
            nodes = list(self.genome.nodes.values())
            if nodes:
                num_nodes = len(nodes)
                node_index = secondary % num_nodes
                ng = nodes[node_index]

                old_bias = ng.bias
                if random.random() < cfg.bias_mutate_rate:
                    if random.random() < cfg.bias_replace_rate:
                        ng.bias = random.gauss(cfg.bias_init_mean, cfg.bias_init_stdev)
                    else:
                        ng.bias += random.gauss(0.0, 1.0) * cfg.bias_mutate_power

                    ng.bias = max(cfg.bias_min_value, min(ng.bias, cfg.bias_max_value))
                    mutation_successful = abs(ng.bias - old_bias) > 1e-6

        # Evaluate the mutated genome
        current_fitness = self._evaluate_genome()
        print(f"[NEAT_ENV] Pre-mutation fitness: {pre_fitness:.4f}")
        print(f"[NEAT_ENV] Post-mutation fitness: {current_fitness:.4f}")
        print(f"[NEAT_ENV] Mutation successful: {mutation_successful}")

        # Use the calculate_reward method for consistency
        reward, terminated = self.calculate_reward(current_fitness, pre_fitness)

        # Add mutation-specific bonuses/penalties
        if mutation_successful and reward > 0:
            reward += 0.1  # Bonus for successful improving mutations
        elif mutation_successful and reward <= 0:
            reward -= 0.05  # Small penalty for successful but non-improving mutations
        elif not mutation_successful:
            reward -= 0.1  # Penalty for failed mutations

        print(f"[NEAT_ENV] Final reward: {reward:.4f}, terminated: {terminated}")

        # Update state
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

    def calculate_reward(self, current_fitness, prev_fitness):
        """Calculate reward based on fitness improvement with immediate bonuses."""
        # Handle invalid genomes
        if current_fitness <= RewardConst.INVALID_ROBOT:
            return RewardConst.INVALID_ROBOT, True

        # Calculate improvement with scaling
        scaled_prev = prev_fitness * RewardConst.SCALE_PREVIOUS
        improvement = current_fitness - scaled_prev

        # Base reward on improvement
        if improvement > 0:
            # Positive improvement gets positive reward
            base_reward = min(improvement * 10, RewardConst.POS_REWARD)  # Scale and clip
            return base_reward, False
        else:
            # No improvement or negative gets negative reward
            base_reward = max(improvement * 10, RewardConst.NEG_REWARD)  # Scale and clip
            return base_reward, False

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
