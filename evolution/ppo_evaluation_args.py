from argparse import Namespace
from types import SimpleNamespace as Namespace


def create_ppo_eval_args():
    """Minimal PPO settings for quick pipeline testing."""
    return Namespace(
        # ——————————————————————————————————————————————————————
        # Rollout & update sizes - REDUCED for speed
        n_steps        = 32,      # Reduced from 256
        n_envs         = 1,       # Single env for simplicity
        batch_size     = 16,      # Reduced from 64
        n_epochs       = 2,       # Reduced from 8
        # ——————————————————————————————————————————————————————
        # Total interaction budget - MINIMAL
        total_timesteps= 100,     # Very short for testing (was 256)
        # ——————————————————————————————————————————————————————
        # Evaluation settings - REDUCED
        eval_interval  = 50,      # Evaluate once at 50 steps
        n_evals        = 1,       # Single evaluation run
        n_eval_envs    = 1,       # Single environment
        log_interval   = 10,      # Log every 10 steps
        verbose_ppo    = 1,       # Standard verbosity
        # ——————————————————————————————————————————————————————
        # Learning & regularization - keep defaults
        learning_rate  = 3e-4,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        vf_coef        = 0.5,
        max_grad_norm  = 0.5,
        ent_coef       = 0.01,
        clip_range     = 0.2,
    )
