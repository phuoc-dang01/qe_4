# t_run_ppo.py
import argparse
import sys
from pathlib import Path
from typing import Optional

import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import evogym

current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parent.parent

# Add paths to sys.path
sys.path.insert(0, str(PROJECT_ROOT))

from evogym.envs import *

# Import the simplified callback
from .t_callback import TEvalCallback


def run_ppo(
    args: argparse.Namespace,
    body: np.ndarray,
    env_name: str,
    model_save_dir: str,
    model_save_name: str,
    connections: Optional[np.ndarray] = None,
    seed: int = 42,
) -> float:
    """
    Run ppo with a single environment to avoid nested parallelism issues.
    """
    print(f"Starting run_ppo for genome {model_save_name}, environment {env_name}")

    try:
        # Create a single environment
        env_kwargs = {
            'body': body,
            'connections': connections,
        }

        print(f"Creating environment {env_name}...")
        base_env = gym.make(env_name, **env_kwargs)
        monitored_env = Monitor(base_env)

        # Wrap it in DummyVecEnv for compatibility with PPO
        print("Creating vector environment wrapper...")
        vec_env = DummyVecEnv([lambda: monitored_env])

        # Create our simplified callback
        print("Creating callback...")
        callback = TEvalCallback(
            body=body,
            connections=connections,
            env_name=env_name,
            eval_every=args.eval_interval,
            n_evals=args.n_evals,
            model_save_dir=model_save_dir,
            model_save_name=model_save_name,
            verbose=args.verbose_ppo,
        )

        # Train with the same parameters
        print("Creating PPO model...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,  # Reduce verbosity to minimize output
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            ent_coef=args.ent_coef,
            clip_range=args.clip_range,
            seed=seed
        )

        print(f"Starting PPO training for {args.total_timesteps} steps...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=args.log_interval
        )

        print(f"PPO training complete, best reward: {callback.best_reward}")

        # Clean up
        vec_env.close()

        return float(callback.best_reward) if np.isfinite(callback.best_reward) else -1.0

    except Exception as e:
        print(f"Error in run_ppo: {str(e)}")
        import traceback
        traceback.print_exc()
        return -1.0  # Return a default negative reward on error
