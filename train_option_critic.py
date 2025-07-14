import argparse
import multiprocessing
import os
import pdb
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'evogym' / 'examples' / 'ppo'))
sys.path.insert(0, str(PROJECT_ROOT / 'evogym' / 'examples' / 'externals' / 'PyTorch-NEAT'))


from datetime import datetime

import neat
import numpy as np
import torch
from env.callbacks import NEATMetricsCallback, OptionMetricsCallback
from env.neat_env import NeatMutationEnv
from fitness import evaluator
from fitness.reward_const import RewardConst
from option_critic.algorithm import OptionCriticPPO
from option_critic.policy import OptionCriticPolicy
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecNormalize)
from wandb.integration.sb3 import WandbCallback

import wandb


def make_vec_env(env_fn, n_envs: int = 1):
    """Create a vectorized environment with n_envs subprocesses or dummy."""
    env_fns = [env_fn for _ in range(n_envs)]
    return SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns)

def setup_neat_config(args):
    """Create and configure the NEAT configuration."""
    config_path = os.path.join(os.path.dirname(__file__), 'neat.cfg')
    # import ppo_evaluation_args
    # ppo_args = ppo_evaluation_args.create_ppo_eval_args()
    ppo_args = None
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
        extra_info={
            'structure_shape': args.structure_shape,
            'save_path': args.save_dir,
            'structure_hashes': {},
            'args': ppo_args,
            'env_name': args.env_name,
        }
    )
    return config

def create_option_critic_model(env, secondary_dims, args):
    """Instantiate the OptionCriticPPO model with given args and env."""
    model = OptionCriticPPO(
        policy=OptionCriticPolicy,
        env=env,
        policy_kwargs={
            'secondary_action_dims': secondary_dims,
            'num_options': args.num_options,
            'entropy_reg': args.entropy_reg,
        },
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        seed=42,
        verbose=1,
        device='cpu',
        num_options=args.num_options,
        termination_reg=args.termination_reg,
    )
    return model

def setup_callbacks(args, eval_env):
    """Create a list of callbacks: checkpoint, eval, NEAT metrics, Wandb, and option metrics."""
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    best_dir = os.path.join(args.save_dir, 'best_model')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    # Checkpoint every save_interval steps
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_interval,
        save_path=ckpt_dir,
        name_prefix='option_critic',
        save_replay_buffer=False
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(args.save_dir, 'eval_logs'),
        eval_freq=args.eval_interval,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        verbose=1
    )
    neat_metrics = NEATMetricsCallback()
    option_metrics = OptionMetricsCallback()

    wandb_cb = WandbCallback(
        gradient_save_freq=0,
        model_save_path=os.path.join(args.save_dir, 'wandb_models'),
        verbose=1,
        log='parameters',
    )
    # return CallbackList([checkpoint_callback, eval_callback, neat_metrics, option_metrics, wandb_cb])
    return CallbackList([checkpoint_callback, eval_callback, wandb_cb, neat_metrics])

def train_option_critic_quick():
    """Streamlined training for quick testing."""
    args = get_quick_test_args()

    print("=" * 60)
    print("QUICK OPTION-CRITIC TRAINING")
    print("=" * 60)
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Save directory: {args.save_dir}")
    print(f"Environments: {args.n_envs}")

    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        name=f"quick_test_{datetime.now().strftime('%H%M')}",
        config=vars(args),
        sync_tensorboard=True
    )

    # NEAT config
    neat_config = setup_neat_config(args)

    # Create training environment (simple setup)
    raw_env = NeatMutationEnv(neat_config)
    secondary_dims = [space.n for space in raw_env.action_spaces_consequences]
    raw_env.close()

    # Training environment
    env_fn = lambda: Monitor(NeatMutationEnv(neat_config))
    train_env = make_vec_env(env_fn, args.n_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Create model
    model = create_option_critic_model(train_env, secondary_dims, args)

    # Simple logging
    logger = configure(os.path.join(args.save_dir, 'logs'), ['stdout', 'tensorboard'])
    model.set_logger(logger)

    # Minimal callbacks - just W&B
    wandb_cb = WandbCallback(
        model_save_path=os.path.join(args.save_dir, 'wandb_models'),
        verbose=1,
    )

    print("\nStarting training...")

    # Train
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=wandb_cb,
        progress_bar=True
    )

    print("\nTraining completed! Saving model...")

    # Save using our clean method
    model_path = save_model_properly(model, args.save_dir)

    print(f"\n✓ All done!")
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ Directory: {args.save_dir}")

    # Cleanup
    wandb.finish()
    train_env.close()

    return model_path, args.save_dir


def save_model_properly(model, save_dir, model_name="final_model"):
    """Save model and VecNormalize separately using SB3's methods."""
    os.makedirs(save_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)  # This creates final_model.zip
    print(f"✓ Model saved to: {model_path}.zip")

    # Save VecNormalize if you're using it
    if hasattr(model, 'env') and hasattr(model.env, 'save'):
        vecnorm_path = os.path.join(save_dir, 'vecnormalize.pkl')
        model.env.save(vecnorm_path)
        print(f"✓ VecNormalize saved to: {vecnorm_path}")

    return f"{model_path}.zip"

def init_train_args():
    """Initialize the argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_envs', type=int, default=18)
    parser.add_argument('--total_timesteps', type=int, default=1000)

    parser.add_argument('--n_steps', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64) # n_steps * n_epochs
    parser.add_argument('--n_epochs', type=int, default=8)

    parser.add_argument('--eval_interval', type=int, default=256)
    parser.add_argument('--n_eval_episodes', type=int, default=2)

    parser.add_argument('--save_interval', type=int, default=500)

    parser.add_argument('--num_options', type=int, default=6)
    parser.add_argument('--termination_reg', type=float, default=0.02)
    parser.add_argument('--entropy_reg', type=float, default=0.02)
    parser.add_argument('--learning_rate', type=float, default=3e-4)

    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--gae_lambda', type=float, default=0.8)
    parser.add_argument('--clip_range', type=float, default=0.1)

    parser.add_argument('--wandb_project', type=str, default='OptionCriticMutation')
    # parser.add_argument('--wandb_grad_save_freq', type=int, default=0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    parser.add_argument('--save_dir', type=str, default=f'/home/pd468/qe/rl_mutation/models/{timestamp}')
    parser.add_argument('--structure_shape', type=tuple, default=(5,5))
    parser.add_argument('--env_name', type=str, default='Walker-v0')
    args = parser.parse_args()
    return args


def get_quick_test_args():
    """Get minimal args for quick testing."""
    parser = argparse.ArgumentParser()

    # Minimal training parameters
    parser.add_argument('--n_envs', type=int, default=2)  # Small for testing
    parser.add_argument('--total_timesteps', type=int, default=500)  # Very short

    parser.add_argument('--n_steps', type=int, default=16)  # Small buffer
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=2)  # Quick epochs

    parser.add_argument('--eval_interval', type=int, default=100)  # Frequent eval
    parser.add_argument('--n_eval_episodes', type=int, default=1)  # Minimal eval
    parser.add_argument('--save_interval', type=int, default=200)  # Frequent saves

    # Standard RL parameters
    parser.add_argument('--num_options', type=int, default=6)
    parser.add_argument('--termination_reg', type=float, default=0.02)
    parser.add_argument('--entropy_reg', type=float, default=0.02)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--gae_lambda', type=float, default=0.8)
    parser.add_argument('--clip_range', type=float, default=0.1)

    # Paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    parser.add_argument('--save_dir', type=str, default=f'./models/quick_test_{timestamp}')
    parser.add_argument('--structure_shape', type=tuple, default=(5,5))
    parser.add_argument('--env_name', type=str, default='Walker-v0')

    # W&B
    parser.add_argument('--wandb_project', type=str, default='QuickTest-OptionCritic')

    args = parser.parse_args([])  # Empty args for defaults
    return args


if __name__ == '__main__':
    model_path, save_dir = train_option_critic_quick()

    print(f"\nNext steps:")
    print(f"1. Update test_policy.py with model path: {model_path}")
    print(f"2. Run integration test")
