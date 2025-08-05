from dataclasses import dataclass, asdict

@dataclass
class ExperimentConfig:
    """Configuration for RL-guided NEAT experiments."""

    # Environment settings
    env_name: str = "Walker-v0"
    structure_shape: tuple = (5, 5)
    evaluator_type: str = "dummy"  # Always use dummy for testing

    # RL training settings - MINIMAL for testing
    n_envs: int = 1  # Reduced from 4
    total_timesteps: int = 100  # Reduced from 10000
    n_steps: int = 16  # Reduced from 128
    batch_size: int = 16  # Reduced from 256
    n_epochs: int = 2  # Reduced from 10

    # Reward function
    reward_function: str = "improvement"

    # Option-Critic settings - SIMPLIFIED
    num_options: int = 4  # Reduced from 6
    termination_reg: float = 0.02
    entropy_reg: float = 0.02

    def to_dict(self):
        return asdict(self)
