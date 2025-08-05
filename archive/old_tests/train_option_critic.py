import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import neat
import numpy as np
import torch
from env.env_callbacks import NEATMetricsCallback, OptionMetricsCallback
from env.env_neat import NeatMutationEnv
from option_critic.algorithm import OptionCriticPPO
from option_critic.policy import OptionCriticPolicy
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from wandb.integration.sb3 import WandbCallback
import wandb

def create_env_fn(neat_config, evaluator_type="dummy"):
    """Create environment factory function."""
    def _init():
        # Process isolation
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['OMP_NUM_THREADS'] = '1'

        env = NeatMutationEnv(neat_config, evaluator_type=evaluator_type)
        return Monitor(env)
    return _init

def setup_neat_config(env_name="Walker-v0", structure_shape=(5, 5)):
    """Create NEAT configuration."""
    config_path = os.path.join(os.path.dirname(__file__), 'neat.cfg')

    # Import the PPO evaluation args
    import rl_mutation.evolution.ppo_evaluation_args as ppo_evaluation_args
    ppo_args = ppo_evaluation_args.create_ppo_eval_args()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
        extra_info={
            'structure_shape': structure_shape,
            'env_name': env_name,
            'args': ppo_args,  # Add this line - it's required by eval_fitness
            'save_path': './saved_data',  # Add default save path
            'structure_hashes': {},  # Add structure hashes dict
        }
    )
    return config

def train_option_critic_quick():
    """Quick training for testing."""
    print("=" * 60)
    print("QUICK OPTION-CRITIC TRAINING")
    print("=" * 60)

    # Configuration - MINIMAL for testing
    n_envs = 1
    total_timesteps = 100
    save_dir = f'./models/quick_test_{datetime.now().strftime("%Y%m%d_%H%M")}'
    num_options = 4

    print(f"Total timesteps: {total_timesteps}")
    print(f"Save directory: {save_dir}")
    print(f"Environments: {n_envs}")
    print(f"Number of options: {num_options}")

    # Initialize W&B
    wandb.init(
        project="QuickTest-OptionCritic",
        name=f"quick_test_{datetime.now().strftime('%H%M')}",
        config={
            "n_envs": n_envs,
            "total_timesteps": total_timesteps,
            "num_options": num_options,
        }
    )

    # Setup NEAT config
    neat_config = setup_neat_config()

    # Get action dimensions from environment
    temp_env = NeatMutationEnv(neat_config, evaluator_type="dummy")

    # Debug: Let's see what we're getting
    print(f"Environment action space: {temp_env.action_space}")
    print(f"Secondary action dims: {temp_env.secondary_action_dims}")

    # For a MultiDiscrete action space, we need to handle it differently
    # The environment expects actions of shape [primary, secondary]
    # For Option-Critic, each option should be able to output both

    # Get the maximum secondary action dimension
    max_secondary = max(temp_env.secondary_action_dims)

    # Create a list where each option has the same max secondary dimension
    # This ensures all options can perform any mutation
    secondary_dims_for_options = [max_secondary for _ in range(num_options)]

    print(f"Secondary dims for options: {secondary_dims_for_options}")

    temp_env.close()

    # Create vectorized environment
    env_fns = [create_env_fn(neat_config, "dummy") for _ in range(n_envs)]
    train_env = DummyVecEnv(env_fns)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Create model with minimal parameters
    model = OptionCriticPPO(
        policy=OptionCriticPolicy,
        env=train_env,
        policy_kwargs={
            'secondary_action_dims': secondary_dims_for_options,
            'num_options': num_options,
            'entropy_reg': 0.02,
        },
        learning_rate=3e-4,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        gamma=0.9,
        gae_lambda=0.8,
        clip_range=0.1,
        verbose=1,
        device='cpu',
        num_options=num_options,
        termination_reg=0.02,
    )
    # Setup logging
    os.makedirs(save_dir, exist_ok=True)
    logger = configure(os.path.join(save_dir, 'logs'), ['stdout'])
    model.set_logger(logger)

    # Minimal callbacks
    callbacks = CallbackList([
        NEATMetricsCallback(),
        OptionMetricsCallback(),
        WandbCallback(verbose=0),
    ])

    print("\nStarting training...")

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    print("\nTraining completed! Saving model...")

    # Save model
    model_path = os.path.join(save_dir, "final_model")
    model.save(model_path)
    train_env.save(os.path.join(save_dir, "vecnormalize.pkl"))

    print(f"\n✓ Model saved to: {model_path}.zip")
    print(f"✓ Directory: {save_dir}")

    # Cleanup
    wandb.finish()
    train_env.close()

    return f"{model_path}.zip", save_dir

def train_with_config(exp_name: str, config):
    """Train with experiment configuration."""
    from experiments.config import ExperimentConfig
    from env.env_factory import NEATEnvFactory

    print(f"\nTraining experiment: {exp_name}")
    print(f"Configuration: {config.to_dict()}")

    # Process isolation
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'

    # Initialize W&B
    wandb.init(
        project="RL-NEAT-Experiments",
        name=exp_name,
        config=config.to_dict(),
        reinit=True
    )

    # Setup NEAT config
    neat_config = setup_neat_config(config.env_name, config.structure_shape)

    # Create environment factory
    env_factory = NEATEnvFactory(neat_config, config.evaluator_type)

    # Create training environments
    train_env = env_factory.make_vec_env(config.n_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Get action dimensions
    dummy_env = env_factory.make_env(0, 0)()
    secondary_dims = dummy_env.secondary_action_dims
    dummy_env.close()

    # Fix: Use the same dims for each option
    secondary_dims_per_option = secondary_dims

    # Create model
    model = OptionCriticPPO(
        policy=OptionCriticPolicy,
        env=train_env,
        policy_kwargs={
            'secondary_action_dims': secondary_dims_per_option,  # Fixed format
            'num_options': config.num_options,
            'entropy_reg': config.entropy_reg,
        },
        learning_rate=3e-4,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        seed=42,
        verbose=1,
        device='cpu',
        num_options=config.num_options,
        termination_reg=config.termination_reg,
    )

    # Minimal callbacks
    callbacks = CallbackList([
        NEATMetricsCallback(),
        OptionMetricsCallback(),
        WandbCallback(verbose=0),
    ])

    # Train
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    # Save
    save_path = f"./models/{exp_name}"
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, "final_model")
    model.save(model_path)
    train_env.save(os.path.join(save_path, "vecnormalize.pkl"))

    wandb.finish()
    train_env.close()

    return exp_name, f"{model_path}.zip"

if __name__ == '__main__':
    # Quick test by default
    model_path, save_dir = train_option_critic_quick()
    print(f"\nTo use this model in test_t_run.py:")
    print(f"rl_path = '{model_path}'")
