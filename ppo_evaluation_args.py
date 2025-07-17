from argparse import Namespace
from types import SimpleNamespace as Namespace


def create_ppo_eval_args():
    """Ultra-fast PPO eval settings for rapid genome fitness trials."""
    return Namespace(
        n_steps        = 32,      # Reduced from 128
        batch_size     = 32,      # Match n_steps for single update
        n_epochs       = 1,       # Minimal epochs
        total_timesteps= 100,     # Keep minimal
        eval_interval  = 200,     # Skip evaluation during training
        n_evals        = 1,
        n_eval_envs    = 1,
        log_interval   = 10,
        verbose_ppo    = 0,       # Silent mode
        learning_rate  = 3e-4,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        ent_coef       = 0.01,
        clip_range     = 0.2,
        n_envs         = 1,       # Single environment
    )

