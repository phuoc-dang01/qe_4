import os
import multiprocessing as mp
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from .env_neat import NeatMutationEnv

class NEATEnvFactory:
    """Factory for creating NEAT mutation environments with proper isolation."""

    def __init__(self, neat_config, evaluator_type="dummy"):
        self.neat_config = neat_config
        self.evaluator_type = evaluator_type

    def make_env(self, rank=0, seed=0):
        """Create a single environment instance."""
        def _init():
            import os
            # Ensure process isolation
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['OMP_NUM_THREADS'] = '1'

            env = NeatMutationEnv(self.neat_config, evaluator_type=self.evaluator_type)
            env.seed(seed + rank)
            return Monitor(env)
        return _init

    def make_vec_env(self, n_envs, start_index=0):
        """Create vectorized environments."""
        return SubprocVecEnv([self.make_env(i, start_index) for i in range(n_envs)])
