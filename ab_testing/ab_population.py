# ab_testing/ab_population.py
import pdb

import neat
import numpy as np

from .ab_reporter import ABTestingReporter, SaveResultReporter
from .ab_reproduction import ABTestingReproduction


class ABTestingPopulation(neat.Population):
    """Population class that supports A/B testing of mutation strategies."""

    def __init__(self, config, rl_policy_path=None, ab_ratio=0.5, tracking_enabled=True):
        """Initialize population with A/B testing capabilities."""
        # Initialize as a standard population
        super().__init__(config)
        # Replace the reproduction method with our A/B testing version
        self.reproduction = ABTestingReproduction(
            config.reproduction_config,
            self.reporters,
            config.stagnation_type(config.stagnation_config, self.reporters),
            rl_policy_path=rl_policy_path,
            ab_ratio=ab_ratio,
            tracking_enabled=tracking_enabled,
            parent_population=self,
        )

        # Add AB testing reporter
        self.ab_reporter = ABTestingReporter(self.reproduction, config.extra_info['save_path'], report_interval=1)
        self.add_reporter(self.ab_reporter)
        self.save_reporter = SaveResultReporter(config.extra_info['save_path'])
        self.add_reporter(self.save_reporter)

        # Initialize tracking - assign groups to initial population
        for genome_id, genome in self.population.items():
            self.reproduction.genome_group[genome_id] = self.reproduction._assign_random_group()

    def _update_group_stats(self, generation, population):
        """Update statistics for each group (rl vs standard), computing (min, mean, max) for fitness and complexity."""
        # split into two lists
        rl_group = [g for g in population.values() if self.reproduction.genome_group[g.key] == 'rl']
        std_group = [g for g in population.values() if self.reproduction.genome_group[g.key] == 'standard']

        def compute_stats(values):
            if not values:
                return (0.0, 0.0, 0.0)
            arr = np.array(values, dtype=float)
            return (float(arr.min()), float(arr.mean()), float(arr.max()))

        # fitness stats
        rl_f_stats = compute_stats([g.fitness for g in rl_group])
        std_f_stats = compute_stats([g.fitness for g in std_group])

        # complexity stats: here we define complexity as total genes (nodes + connections);
        # replace with whatever measure you track
        rl_c_stats = compute_stats([
            len(getattr(g, 'nodes', [])) + len(getattr(g, 'connections', []))
            for g in rl_group
        ])
        std_c_stats = compute_stats([
            len(getattr(g, 'nodes', [])) + len(getattr(g, 'connections', []))
            for g in std_group
        ])

        # push through to reporter in the expected shape
        self.ab_reporter.update_stats(generation, {
            'rl': {
                'fitness':    rl_f_stats,
                'complexity': rl_c_stats,
            },
            'standard': {
                'fitness':    std_f_stats,
                'complexity': std_c_stats,
            }
        })

    def run(self, fitness_function, constraint_function=None, n=None):
        """
        Runs NEAT's genetic algorithm for at most n generations.  If n
        is None, run until solution is found or extinction occurs.

        The user-provided fitness_function must take only two arguments:
            1. The population as a list of (genome id, genome) tuples.
            2. The current configuration object.

        The return value of the fitness function is ignored, but it must assign
        a Python float to the `fitness` member of each genome.

        The fitness function is free to maintain external state, perform
        evaluations in parallel, etc.

        It is assumed that fitness_function does not modify the list of genomes,
        the genomes themselves (apart from updating the fitness member),
        or the configuration object.
        """
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        k = 0
        while n is None or k < n:
            k += 1
            print("===Start Generation: {}".format(self.generation))
            self.reporters.start_generation(self.generation)

            # Evaluate all genomes using the user-provided constraint function.
            # If some genomes violate the constraint, generate new genomes and replace them, until all genomes satisfy the constraint.
            if constraint_function is not None:
                genomes = list(self.population.items())
                validity = constraint_function(genomes, self.config, self.generation)
                if not all(validity):
                    valid_idx = np.where(validity)[0]
                    valid_genomes = np.array(genomes)[valid_idx]
                    while len(valid_genomes) < self.config.pop_size:
                        new_population = self.reproduction.create_new(self.config.genome_type,
                                                                    self.config.genome_config,
                                                                    self.config.pop_size)
                        new_genomes = list(new_population.items())
                        validity = constraint_function(new_genomes, self.config, self.generation)
                        valid_idx = np.where(validity)[0]
                        valid_genomes = np.vstack([valid_genomes, np.array(new_genomes)[valid_idx]])
                    valid_genomes = valid_genomes[:self.config.pop_size]
                    self.population = dict(valid_genomes)
                    self.species.speciate(self.config, self.population, self.generation)

            # Evaluate all genomes using the user-provided function.
            fitness_function(list(self.population.items()), self.config, self.generation)

            # Gather and report statistics.
            best = None
            for g in self.population.values():
                if g.fitness is None:
                    raise RuntimeError("Fitness not assigned to genome {}".format(g.key))

                if best is None or g.fitness > best.fitness:
                    best = g
            self.reporters.post_evaluate(self.config, self.population, self.species, best)

            # Track the best genome ever seen.
            if self.best_genome is None or best.fitness > self.best_genome.fitness:
                self.best_genome = best

            if not self.config.no_fitness_termination:
                # End if the fitness threshold is reached.
                fv = self.fitness_criterion(g.fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    self.reporters.found_solution(self.config, self.generation, best)
                    break

            pdb.set_trace()
            # Create the next generation from the current generation.
            self.population = self.reproduction.reproduce(self.config, self.species,
                                                          self.config.pop_size, self.generation)

            # Check for complete extinction.
            if not self.species.species:
                self.reporters.complete_extinction()

                # If requested by the user, create a completely new population,
                # otherwise raise an exception.
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(self.config.genome_type,
                                                                   self.config.genome_config,
                                                                   self.config.pop_size)
                else:
                    raise CompleteExtinctionException()

            # Divide the new population into species.
            self.species.speciate(self.config, self.population, self.generation)

            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        if self.config.no_fitness_termination:
            self.reporters.found_solution(self.config, self.generation, self.best_genome)

        return self.best_genome

