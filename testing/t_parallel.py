"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
import multiprocessing as mp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(mp.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NonDaemonPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc


class ParallelEvaluator(object):
    def __init__(self, num_workers, fitness_function, constraint_function=None, timeout=None):
        """
        fitness_function should take four arguments:
        (genome object, config object, genome_id, generation)
        and return a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.fitness_function = fitness_function
        self.constraint_function = constraint_function
        self.timeout = timeout
        self.pool = NonDaemonPool(num_workers) if num_workers > 1 else None

    def __del__(self):
        if self.pool:
            self.pool.close()
            self.pool.join()

    def evaluate_fitness(self, genomes, config, generation):
        logging.info(f"Evaluating {len(genomes)} genomes in parallel (gen {generation})")

        if self.num_workers == 1:
            # Serial execution
            for i, (gid, genome) in enumerate(genomes):
                genome.fitness = self.fitness_function(genome, config, gid, generation)
        else:
            # Parallel execution
            jobs = []
            for gid, genome in genomes:
                jobs.append(self.pool.apply_async(self.fitness_function,
                                                 (genome, config, gid, generation)))

            # assign the fitness back to each genome
            for job, (gid, genome) in zip(jobs, genomes):
                try:
                    genome.fitness = job.get(timeout=self.timeout)
                except Exception as e:
                    logging.error(f"Fitness evaluation failed for genome {gid}: {str(e)}")
                    genome.fitness = -1.0

    def evaluate_constraint(self, genomes, config, generation):
        logging.info(f"Evaluating constraints for {len(genomes)} genomes (gen {generation})")

        validity_all = []
        for i, (gid, genome) in enumerate(genomes):
            validity = self.constraint_function(genome, config, gid, generation)
            validity_all.append(validity)
            logging.info(f"Genome {gid} constraint: {validity}")

        return validity_all
