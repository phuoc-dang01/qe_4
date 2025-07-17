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
        Evaluate the fitness of each genome serially to avoid nested parallelism issues.
        """
        logging.info(f"Evaluating {len(genomes)} genomes serially (gen {generation})")

        for i, (gid, genome) in enumerate(genomes):
            try:
                logging.info(f"Starting evaluation for genome {gid} ({i+1}/{len(genomes)})")
                fitness = self.fitness_function(genome, config, gid, generation)  # Use gid instead of i
                genome.fitness = fitness
                logging.info(f"Genome {gid} completed with fitness: {fitness}")
            except Exception as e:
                logging.error(f"Genome {gid} evaluation failed: {e}")
                genome.fitness = -1.0  # Default negative fitness for failed evaluations

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
