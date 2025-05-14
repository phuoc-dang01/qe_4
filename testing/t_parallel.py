import concurrent.futures
import logging
import multiprocessing as mp
import os
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from evogym.envs import *


class ParallelEvaluator:
    def __init__(self, num_workers, fitness_function, constraint_function=None, timeout=None):
        """
        Parameters:
        - num_workers: Number of parallel workers
        - fitness_function: Function to evaluate genome fitness
        - constraint_function: Function to check genome constraints
        - timeout: Timeout for each genome evaluation (seconds)
        """
        self.num_workers = num_workers
        self.fitness_function = fitness_function
        self.constraint_function = constraint_function
        self.timeout = timeout

    def evaluate_fitness(self, genomes, config, generation):
        """
        Evaluate the fitness of each genome in parallel.

        Parameters:
        - genomes: List of (genome_id, genome) tuples
        - config: Configuration object
        - generation: Current generation number
        """
        logging.info(f"Evaluating {len(genomes)} genomes with {self.num_workers} workers (gen {generation})")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all jobs
            future_to_genome = {}
            for i, (gid, genome) in enumerate(genomes):
                future = executor.submit(self._evaluate_genome, genome, config, i, generation)
                future_to_genome[future] = (gid, genome)

            # Collect results
            for future in concurrent.futures.as_completed(future_to_genome):
                gid, genome = future_to_genome[future]
                try:
                    fitness = future.result(timeout=self.timeout)
                    genome.fitness = fitness
                    logging.info(f"Genome {gid} fitness: {fitness}")
                except concurrent.futures.TimeoutError:
                    logging.error(f"Genome {gid} evaluation timed out")
                    genome.fitness = -1  # or some default value
                except Exception as e:
                    logging.error(f"Genome {gid} evaluation failed: {str(e)}")
                    genome.fitness = -1  # or some default value

    def _evaluate_genome(self, genome, config, idx, generation):
        """Wrapper for fitness function to handle exceptions"""
        try:
            return self.fitness_function(genome, config, idx, generation)
        except Exception as e:
            logging.error(f"Error in fitness function for genome {idx}: {str(e)}")
            raise

    def evaluate_constraint(self, genomes, config, generation):
        """
        Evaluate the constraints of each genome in parallel.

        Parameters:
        - genomes: List of (genome_id, genome) tuples
        - config: Configuration object
        - generation: Current generation number

        Returns:
        - List of validity values, one for each genome
        """
        if not self.constraint_function:
            return [True] * len(genomes)

        logging.info(f"Evaluating constraints for {len(genomes)} genomes (gen {generation})")

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all jobs
            future_to_idx = {}
            for i, (gid, genome) in enumerate(genomes):
                future = executor.submit(self.constraint_function, genome, config, i, generation)
                future_to_idx[future] = (i, gid, genome)

            # Collect results in order
            validity_all = [None] * len(genomes)
            for future in concurrent.futures.as_completed(future_to_idx):
                idx, gid, genome = future_to_idx[future]
                try:
                    validity = future.result(timeout=self.timeout)
                    validity_all[idx] = validity
                    logging.info(f"Genome {gid} constraint: {validity}")
                except Exception as e:
                    logging.error(f"Constraint evaluation failed for genome {idx}: {str(e)}")
                    validity_all[idx] = False

        return validity_all


# Example usage:

# Define your fitness function
# def eval_genome_fitness(genome, config, idx, generation):
#     # Print process info to verify parallelism
#     process_id = os.getpid()
#     print(f"Evaluating genome {idx} in process {process_id} for generation {generation}")

#     # Simulate some computation
#     start_time = time.time()
#     time.sleep(0.5)  # Simulate work

#     # Your actual fitness calculation here
#     fitness = idx + generation  # Just a dummy calculation

#     end_time = time.time()
#     print(f"Genome {idx} evaluation completed in {end_time - start_time:.2f}s (process {process_id})")

#     return fitness

# # Define your constraint function (if needed)
# def eval_genome_constraint(genome, config, idx, generation):
#     # Some constraint check
#     return True  # All genomes are valid in this example

# # Create dummy objects for testing
# class DummyGenome:
#     def __init__(self, id):
#         self.id = id
#         self.fitness = None

# class DummyConfig:
#     pass

# # Setup parameters
# num_cores = mp.cpu_count()  # Use all available cores
# pop_size = 20
# max_evaluations = 100

# # Create dummy population
# genomes = [(i, DummyGenome(i)) for i in range(pop_size)]
# config = DummyConfig()

# # Create the evaluator
# evaluator = ParallelEvaluator(num_cores, eval_genome_fitness, eval_genome_constraint)

# # Simulate a few generations
# for generation in range(int(np.ceil(max_evaluations / pop_size))):
#     print(f"\n--- Generation {generation} ---")

#     # Evaluate fitness
#     evaluator.evaluate_fitness(genomes, config, generation)

#     # Evaluate constraints
#     validity = evaluator.evaluate_constraint(genomes, config, generation)

#     # Print results
#     for (gid, genome), valid in zip(genomes, validity):
#         print(f"Genome {gid}: Fitness={genome.fitness}, Valid={valid}")
