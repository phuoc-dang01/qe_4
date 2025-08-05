import logging
import random
import copy
import numpy as np
from collections import defaultdict
from neat.reproduction import DefaultReproduction
from neat.math_util import mean


class AdaptiveReproduction(DefaultReproduction):
    """
    Online adaptive mutation strategy that learns during evolution.

    This class tracks the success of different mutation types and adapts
    the mutation selection strategy based on historical performance.
    """

    def __init__(self, config, reporters, stagnation):
        super().__init__(config, reporters, stagnation)

        # Mutation types to track
        self.mutation_types = [
            'add_node',
            'delete_node',
            'add_connection',
            'delete_connection',
            'modify_weight',
            'modify_bias'
        ]

        # Initialize mutation tracking
        self.mutation_stats = defaultdict(lambda: {
            'attempts': 0,
            'successes': 0,
            'total_improvement': 0.0,
            'recent_improvements': [],  # Rolling window
            'avg_improvement': 0.0
        })

        # Strategy parameters
        self.exploration_rate = 0.3  # Initial exploration rate
        self.exploration_decay = 0.995  # Decay per generation
        self.min_exploration = 0.1  # Minimum exploration rate
        self.window_size = 20  # Size of rolling window
        self.min_attempts = 5  # Min attempts before considering stats

        # Selection strategy: 'epsilon_greedy', 'softmax', 'ucb'
        self.selection_strategy = 'softmax'
        self.temperature = 1.0  # For softmax
        self.ucb_c = 2.0  # For UCB

        # Tracking
        self.pending_mutations = {}  # genome_id -> mutation info
        self.generation = 0

        # Logging
        self.logger = logging.getLogger('adaptive_reproduction')
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def create_new(self, genome_type, genome_config, num_genomes):
        """Create new population - called at start or after extinction."""
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()
        return new_genomes

    def reproduce(self, config, species, pop_size, generation):
        """Create new generation with adaptive mutations."""
        self.generation = generation

        # Decay exploration rate
        self.exploration_rate = max(
            self.min_exploration,
            self.exploration_rate * self.exploration_decay
        )

        # Update fitness results from previous generation
        all_fitnesses = []
        for s in species.species.values():
            for g in s.members.values():
                all_fitnesses.append(g.fitness)
                # Update mutation stats if this genome was tracked
                if g.key in self.pending_mutations:
                    self._update_mutation_stats(g.key, g.fitness)

        # Log generation statistics
        if all_fitnesses:
            self.logger.info(f"Generation {generation}: "
                           f"Avg fitness={mean(all_fitnesses):.4f}, "
                           f"Max fitness={max(all_fitnesses):.4f}, "
                           f"Exploration rate={self.exploration_rate:.3f}")

        # Standard NEAT reproduction process
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
            return self.create_new(config.genome_type, config.genome_config, pop_size)

        # Calculate adjusted fitness
        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)
        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:
            mean_fitness = mean([m.fitness for m in afs.members.values()])
            afs.adjusted_fitness = (mean_fitness - min_fitness) / fitness_range

        adjusted_fitnesses = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness = mean(adjusted_fitnesses)
        self.logger.debug(f"Average adjusted fitness: {avg_adjusted_fitness:.3f}")

        # Compute spawn amounts
        previous_sizes = [len(s.members) for s in remaining_species]
        min_species_size = max(self.reproduction_config.min_species_size,
                               self.reproduction_config.elitism)
        spawn_amounts = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                         pop_size, min_species_size)

        # Create new population
        new_population = {}
        species.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            spawn = max(spawn, self.reproduction_config.elitism)
            if spawn <= 0:
                continue

            old_members = list(s.members.items())
            s.members = {}
            species.species[s.key] = s

            # Sort by fitness
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites
            if self.reproduction_config.elitism > 0:
                for i, (gid, g) in enumerate(old_members[:self.reproduction_config.elitism]):
                    if spawn > 0:
                        new_population[gid] = g
                        spawn -= 1

            # Only use species members for reproduction if they meet threshold
            repro_cutoff = max(2, int(len(old_members) * self.reproduction_config.survival_threshold))
            old_members = old_members[:repro_cutoff]

            # Create offspring with adaptive mutations
            while spawn > 0:
                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Create offspring
                gid = next(self.genome_indexer)
                child = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)

                # Apply adaptive mutation
                mutation_type = self._select_mutation_type()
                self._apply_mutation(child, mutation_type, config)

                # Track mutation for later analysis
                parent_fitness = max(parent1.fitness, parent2.fitness)
                self.pending_mutations[gid] = {
                    'mutation_type': mutation_type,
                    'parent_fitness': parent_fitness,
                    'generation': generation
                }

                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)
                spawn -= 1

        return new_population

    def _select_mutation_type(self):
        """Select mutation type based on adaptive strategy."""
        if self.selection_strategy == 'epsilon_greedy':
            if random.random() < self.exploration_rate:
                # Explore: random mutation
                return random.choice(self.mutation_types)
            else:
                # Exploit: best performing mutation
                scores = {}
                for mutation in self.mutation_types:
                    stats = self.mutation_stats[mutation]
                    if stats['attempts'] < self.min_attempts:
                        scores[mutation] = 0.0
                    else:
                        success_rate = stats['successes'] / stats['attempts']
                        avg_improvement = stats['avg_improvement']
                        scores[mutation] = success_rate * (1 + max(0, avg_improvement))

                if not scores or max(scores.values()) == 0:
                    return random.choice(self.mutation_types)

                return max(scores, key=scores.get)

        elif self.selection_strategy == 'softmax':
            scores = []
            mutations = []
            for mutation in self.mutation_types:
                stats = self.mutation_stats[mutation]
                if stats['attempts'] < self.min_attempts:
                    score = 0.5
                else:
                    success_rate = stats['successes'] / stats['attempts']
                    score = success_rate * (1 + max(0, stats['avg_improvement']))
                scores.append(score / self.temperature)
                mutations.append(mutation)

            # Softmax probabilities
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            return np.random.choice(mutations, p=probs)

        elif self.selection_strategy == 'ucb':
            scores = {}
            total_attempts = sum(self.mutation_stats[m]['attempts'] for m in self.mutation_types)

            for mutation in self.mutation_types:
                stats = self.mutation_stats[mutation]
                if stats['attempts'] == 0:
                    scores[mutation] = float('inf')
                else:
                    exploitation = stats['successes'] / stats['attempts']
                    exploration = np.sqrt(self.ucb_c * np.log(total_attempts) / stats['attempts'])
                    scores[mutation] = exploitation + exploration

            return max(scores, key=scores.get)

    def _apply_mutation(self, genome, mutation_type, config, secondary=None):
        cfg = config.genome_config

        if mutation_type == 'add_node':
            genome.mutate_add_node(cfg)
        elif mutation_type == 'delete_node':
            genome.mutate_delete_node(cfg)
        elif mutation_type == 'add_connection':
            genome.mutate_add_connection(cfg)
        elif mutation_type == 'delete_connection':
            genome.mutate_delete_connection()
        elif mutation_type == 'modify_weight':
            # Use secondary to select which connection/bin to mutate
            conns = list(genome.connections.values())
            if conns:
                idx = (secondary or 0) % len(conns)
                conn = conns[idx]
                mean = getattr(cfg, 'weight_init_mean', 0.0)
                std = getattr(cfg, 'weight_init_stdev', 1.0)
                conn.weight = random.gauss(mean, std)
        elif mutation_type == 'modify_bias':
            nodes = list(genome.nodes.values())
            if nodes:
                idx = (secondary or 0) % len(nodes)
                node = nodes[idx]
                mean = getattr(cfg, 'bias_init_mean', 0.0)
                std = getattr(cfg, 'bias_init_stdev', 1.0)
                node.bias = random.gauss(mean, std)
        else:
            self.logger.warning(f"Unsupported mutation type: {mutation_type}")

    def _update_mutation_stats(self, genome_id, fitness):
        """Update mutation statistics after fitness evaluation."""
        if genome_id not in self.pending_mutations:
            return

        info = self.pending_mutations[genome_id]
        mutation_type = info['mutation_type']
        parent_fitness = info['parent_fitness']

        # Calculate improvement
        improvement = fitness - parent_fitness

        # Update statistics
        stats = self.mutation_stats[mutation_type]
        stats['attempts'] += 1

        if improvement > 0:
            stats['successes'] += 1
            stats['total_improvement'] += improvement

        # Update rolling window
        stats['recent_improvements'].append(improvement)
        if len(stats['recent_improvements']) > self.window_size:
            stats['recent_improvements'].pop(0)

        # Update average improvement
        if stats['recent_improvements']:
            stats['avg_improvement'] = mean(stats['recent_improvements'])

        # Log significant improvements
        if improvement > 0.1:
            self.logger.info(f"Mutation {mutation_type}: improvement={improvement:.4f}")

        # Cleanup
        del self.pending_mutations[genome_id]

    def get_mutation_report(self):
        """Generate report of mutation effectiveness."""
        report = []
        report.append("\n=== Mutation Effectiveness Report ===")
        report.append(f"Generation: {self.generation}")
        report.append(f"Exploration rate: {self.exploration_rate:.3f}")
        report.append("\nMutation Statistics:")

        for mutation in self.mutation_types:
            stats = self.mutation_stats[mutation]
            if stats['attempts'] > 0:
                success_rate = stats['successes'] / stats['attempts']
                report.append(f"\n{mutation}:")
                report.append(f"  Attempts: {stats['attempts']}")
                report.append(f"  Success rate: {success_rate:.3f}")
                report.append(f"  Avg improvement: {stats['avg_improvement']:.4f}")

        return "\n".join(report)

    def save_policy(self, filepath):
        """Save learned mutation policy for transfer learning."""
        import pickle
        policy_data = {
            'mutation_stats': dict(self.mutation_stats),
            'exploration_rate': self.exploration_rate,
            'generation': self.generation,
            'selection_strategy': self.selection_strategy
        }
        with open(filepath, 'wb') as f:
            pickle.dump(policy_data, f)
        self.logger.info(f"Saved mutation policy to {filepath}")

    def load_policy(self, filepath):
        """Load mutation policy from another environment/run."""
        import pickle
        with open(filepath, 'rb') as f:
            policy_data = pickle.load(f)

        # Update stats while preserving some exploration
        for mutation, stats in policy_data['mutation_stats'].items():
            if mutation in self.mutation_stats:
                self.mutation_stats[mutation].update(stats)

        # Adjust exploration rate for transfer learning
        self.exploration_rate = max(self.min_exploration,
                                   policy_data['exploration_rate'] * 1.5)

        self.logger.info(f"Loaded mutation policy from {filepath}")
        self.logger.info(f"Reset exploration rate to {self.exploration_rate:.3f} for transfer learning")
