# t_run_ppo.py
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym  # instead of gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import evogym.envs
import wandb
from evogym.envs import *

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
    Run PPO in serial mode with progress bar and wandb logging.
    """
    print(f"Starting run_ppo for genome {model_save_name}, environment {env_name}")

    try:
        env_kwargs = {
            'body': body,
            'connections': connections,
        }

        print(f"Creating environment {env_name}...")
        base_env = gym.make(env_name, **env_kwargs)
        monitored_env = Monitor(base_env)
        vec_env = DummyVecEnv([lambda: monitored_env])

        # Create log directory
        log_dir = os.path.join(model_save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Initialize W&B
        wandb.init(
            project="NEAT-PPO-EvoGym",
            name=model_save_name,
            config=vars(args),
            sync_tensorboard=True,
            reinit=True,
        )

        # Configure SB3 logger to log to TensorBoard and stdout
        sb3_logger = configure(log_dir, ["stdout", "tensorboard"])

        # Setup model
        print("Creating PPO model...")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=args.verbose_ppo,
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
        model.set_logger(sb3_logger)

        # Setup callback
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

        # Train
        print(f"Starting PPO training for {args.total_timesteps} steps...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=args.log_interval,
            progress_bar=True
        )

        print(f"PPO training complete, best reward: {callback.best_reward}")

        vec_env.close()
        wandb.finish()

        return float(callback.best_reward) if np.isfinite(callback.best_reward) else -1.0

    except Exception as e:
        print(f"Error in run_ppo: {str(e)}")
        import traceback
        traceback.print_exc()
        return -1.0
