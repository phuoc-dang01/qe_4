import copy
import multiprocessing as mp
from experiments.config import ExperimentConfig  # Fix import path

class ParallelTrainingManager:
    """Manages parallel training with different configurations."""

    def __init__(self, base_config: ExperimentConfig):
        self.base_config = base_config
        self.experiments = []

    def add_experiment(self, name: str, config_overrides: dict):
        """Add an experiment with specific config overrides."""
        config = copy.deepcopy(self.base_config)
        for key, value in config_overrides.items():
            setattr(config, key, value)
        self.experiments.append((name, config))

    def run_experiments(self, n_parallel=1):
        """Run experiments with n_parallel at a time."""
        # Use spawn method for clean process isolation
        ctx = mp.get_context('spawn')

        with ctx.Pool(n_parallel) as pool:
            results = pool.map(self._run_single_experiment, self.experiments)
        return results

    def _run_single_experiment(self, exp_tuple):
        name, config = exp_tuple
        # Import here to avoid pickling issues
        from train_option_critic import train_with_config
        return train_with_config(name, config)
