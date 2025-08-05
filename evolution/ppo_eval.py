# evolution/ppo_eval.py
import argparse
import os
import sys
from pathlib import Path
from typing import Optional
import torch

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import evogym.envs
import wandb
from evogym.envs import *

from .callbacks import TEvalCallback  # Changed from .t_callback


def run_ppo(args, body, env_name, model_save_dir, model_save_name, connections=None, seed=42):
    print(f"[DEBUG PPO] Process {os.getpid()} starting run_ppo for {model_save_name}")

    try:
        # The key fix: EvoGym expects 'body' parameter, not 'robot'
        env_kwargs = {
            'body': body,
        }

        # Add connections only if provided
        if connections is not None:
            env_kwargs['connections'] = connections

        print(f"[PPO] Creating environment {env_name}...")
        base_env = gym.make(env_name, **env_kwargs)
        monitored_env = Monitor(base_env)
        vec_env = DummyVecEnv([lambda: monitored_env])

        # Create log directory
        log_dir = os.path.join(model_save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Configure SB3 logger (no W&B)
        print(f"[DEBUG PPO] Configuring logger...")
        sb3_logger = configure(log_dir, ["stdout"])

        print(f"[DEBUG PPO] Creating PPO model on process {os.getpid()}...")
        print(f"[DEBUG PPO] Using device: {torch.cuda.is_available()}")

        # Setup model
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
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
            seed=seed,
            device='cpu'  # Force CPU to avoid GPU conflicts
        )
        print(f"[DEBUG PPO] PPO model created successfully")
        model.set_logger(sb3_logger)

        # Setup callback - also fix the callback to use correct parameters
        callback = TEvalCallback(
            body=body,
            connections=connections,
            env_name=env_name,
            eval_every=args.eval_interval,
            n_evals=args.n_evals,
            model_save_dir=model_save_dir,
            model_save_name=model_save_name,
            verbose=0,  # Reduce callback verbosity
        )

        # Train
        print(f"[PPO] Starting PPO training for {args.total_timesteps} steps...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=args.log_interval,
            progress_bar=False
        )

        print(f"[PPO] PPO training complete, best reward: {callback.best_reward}")

        vec_env.close()

        return float(callback.best_reward) if np.isfinite(callback.best_reward) else -1.0

    except Exception as e:
        print(f"[PPO] Error in run_ppo: {str(e)}")
        import traceback
        traceback.print_exc()
        return -1.0
