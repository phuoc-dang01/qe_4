import logging
import pdb
import random
import sys
from pathlib import Path

import neat
from fitness.reward_const import RewardConst
from neat.math_util import mean
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
        self.config = config  # Store config for later use
        self.rl_policy_path = rl_policy_path
        self.logger = setup_logger('t_testing')

        self.rl_model = None
        self.eval_env = None
        self.rl_policy = None

        # Load the RL policy if path provided
        # if self.rl_policy_path:
        if True:
            self.rl_model, self.eval_env = self.load_rl_policy(self.rl_policy_path)
            self.rl_policy = self.rl_model.policy if self.rl_model else None
            self.logger.info(f"Using RL policy from: {rl_policy_path}")

        # Setup the activation functions
        self.activation_functions = ["sigmoid", "tanh", "abs", "gauss", "identity", "sin", "relu"]

        # Add generation counter
        self.generation = 0

    def load_rl_policy(self, policy_path):
        """Load the RL policy model."""
        try:
            print(f"[DEBUG] load_rl_policy called with policy_path: {policy_path}")
            from test_policy import load_option_critic_model
            from train_option_critic import init_train_args
            args = init_train_args()
            result = load_option_critic_model(policy_path, args)
            print(f"[DEBUG] load_option_critic_model returned successfully")
            return result
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to load RL policy: {e}")
            traceback.print_exc()
            return None, None

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

                print(f"[REPRODUCTION] Creating offspring {gid} from parents {parent1_id}, {parent2_id}")
                print(f"[REPRODUCTION] Child before RL mutation - nodes: {len(child.nodes)}, connections: {len(child.connections)}")

                # Apply RL-guided mutation if available, otherwise use standard mutation
                # if self.rl_model and self.eval_env:
                if True:
                    mutated_child = self._apply_rl_mutation(child, config)
                else:
                    mutated_child = self._apply_standard_mutation(child, config)

                print(f"[REPRODUCTION] Child after mutation - nodes: {len(mutated_child.nodes)}, connections: {len(mutated_child.connections)}")
                print(f"[REPRODUCTION] Child fitness: {getattr(mutated_child, 'fitness', 'None')}")

                # Record in new population
                new_population[gid] = mutated_child
                self.ancestors[gid] = (parent1_id, parent2_id)

        print(f"[REPRODUCTION] New population summary:")
        for gid, genome in new_population.items():
            print(f"[REPRODUCTION]   Genome {gid}: nodes={len(genome.nodes)}, connections={len(genome.connections)}, fitness={getattr(genome, 'fitness', 'None')}")

        self.logger.info(f"Reproduction complete. New population size: {len(new_population)}")
        return new_population

    def _apply_rl_mutation(self, genome, config):
        """Apply single RL-guided mutation to a genome."""
        print(f"[RL_MUTATION] Starting RL-guided mutation for genome {genome.key}")

        try:
            # Create a fresh environment if needed
            if self.eval_env is None:
                from rl_mutation.env.env_neat import NeatMutationEnv
                from stable_baselines3.common.monitor import Monitor
                from stable_baselines3.common.vec_env import DummyVecEnv

                # Create base environment
                base_env = NeatMutationEnv(config)
                # Wrap it properly for compatibility
                monitored_env = Monitor(base_env)
                self.eval_env = DummyVecEnv([lambda: monitored_env])

            # Reset env
            obs = self.eval_env.reset()

            base_env = self.eval_env.envs[0]
            if hasattr(base_env, 'env'):
                base_env = base_env.env

            # Inject the genome into the environment WITHOUT evaluating fitness
            base_env.genome = self._copy_genome(genome)
            base_env.steps = 0

            # Get observation without fitness evaluation
            obs = np.array([base_env._get_observation()])

            print(f"[RL_MUTATION] Original genome: {len(genome.nodes)} nodes, {len(genome.connections)} connections")

            # Get action from RL policy - SINGLE STEP ONLY
            actions, _ = self.rl_model.predict(obs, deterministic=True)
            print(f"[RL_MUTATION] Applying single mutation: action={actions[0]}")

            # Apply mutation without fitness evaluation
            mutated_genome = self._apply_mutation_only(base_env, actions[0], config)

            print(f"[RL_MUTATION] Mutation complete")
            print(f"[RL_MUTATION] Final genome: {len(mutated_genome.nodes)} nodes, {len(mutated_genome.connections)} connections")

            return mutated_genome

        except Exception as e:
            self.logger.error(f"RL mutation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to standard mutation
            return self._apply_standard_mutation(genome, config)

    def _apply_mutation_only(self, env, action, config):
        """Apply mutation without fitness evaluation"""
        primary, secondary = int(action[0]), int(action[1])

        # Copy genome before mutation
        genome = self._copy_genome(env.genome)

        # Apply the mutation based on primary action
        cfg = config.genome_config

        if primary == 0:  # ADD_NODE
            genome.mutate_add_node(cfg)
        elif primary == 1:  # DELETE_NODE
            genome.mutate_delete_node(cfg)
        elif primary == 2:  # ADD_CONNECTION
            genome.mutate_add_connection(cfg)
        elif primary == 3:  # DELETE_CONNECTION
            genome.mutate_delete_connection()
        elif primary == 4:  # MODIFY_WEIGHT
            # Apply weight mutation logic
            conns = list(genome.connections.values())
            if conns:
                conn_index = secondary % len(conns)
                cg = conns[conn_index]
                cg.weight = random.gauss(cfg.weight_init_mean, cfg.weight_init_stdev)
        elif primary == 5:  # MODIFY_BIAS
            # Apply bias mutation logic
            nodes = list(genome.nodes.values())
            if nodes:
                node_index = secondary % len(nodes)
                ng = nodes[node_index]
                ng.bias = random.gauss(cfg.bias_init_mean, cfg.bias_init_stdev)

        return genome


    def _apply_standard_mutation(self, genome, config):
        """Apply standard NEAT mutations as fallback."""
        print(f"[] Applying standard NEAT mutations to genome {genome.key}")

        # Create a copy to mutate
        mutated = self._copy_genome(genome)

        # Apply standard NEAT mutations
        mutated.mutate(config.genome_config)

        return mutated

    def _copy_genome(self, genome):
        """Create a deep copy of a genome."""
        import copy
        try:
            new_genome = copy.deepcopy(genome)
            # Preserve key and fitness
            new_genome.key = genome.key
            if hasattr(genome, 'fitness'):
                new_genome.fitness = genome.fitness
            return new_genome
        except Exception as e:
            self.logger.error(f"Failed to copy genome: {e}")
            # Fallback: return original
            return genome
