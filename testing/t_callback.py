# t_callback.py
import os
from typing import List, Optional

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class TEvalCallback(BaseCallback):
    """
    A simplified evaluation callback that avoids nested parallelism.
    """
    def __init__(
        self,
        body: np.ndarray,
        env_name: str,
        eval_every: int,
        n_evals: int,
        model_save_dir: str,
        model_save_name: str,
        connections: Optional[np.ndarray] = None,
        verbose: int = 0
    ):
        super().__init__(verbose)

        self.body = body
        self.connections = connections
        self.env_name = env_name
        self.eval_every = eval_every
        self.n_evals = n_evals
        self.model_save_dir = model_save_dir
        self.model_save_name = model_save_name

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        self.best_reward = -float('inf')

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        if self.num_timesteps % self.eval_every == 0:
            self._validate_and_save()
        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self._validate_and_save()

    def _validate_and_save(self) -> None:
        """
        Simplified evaluation function that uses a single environment.
        """
        rewards = []
        for _ in range(self.n_evals):
            # Create a single environment for evaluation
            env_kwargs = {'body': self.body, 'connections': self.connections}
            env = gym.make(self.env_name, **env_kwargs)
            env = Monitor(env)

            # Reset environment
            obs = env.reset()
            done = False
            total_reward = 0

            # Run one episode
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                total_reward += reward

            rewards.append(total_reward)
            env.close()

        # Calculate statistics
        mean_reward = np.mean(rewards).item()
        out = f"[{self.model_save_name}] Mean: {mean_reward:.3f}, Std: {np.std(rewards):.3f}, Min: {np.min(rewards):.3f}, Max: {np.max(rewards):.3f}"

        # Save model if better than previous best
        if mean_reward > self.best_reward:
            out += f" NEW BEST ({mean_reward:.3f} > {self.best_reward:.3f})"
            self.best_reward = mean_reward
            self.model.save(os.path.join(self.model_save_dir, self.model_save_name))

        if self.verbose > 0:
            print(out)
