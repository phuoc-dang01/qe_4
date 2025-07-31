from argparse import Namespace
from types import SimpleNamespace as Namespace


def create_ppo_eval_args():
    """Improved PPO eval settings for better robot fitness evaluation."""
    return Namespace(
        # ——————————————————————————————————————————————————————
        # Rollout & update sizes
        n_steps        = 256,     # Increased buffer size for more stable training
        n_envs         = 4,
        batch_size     = 64,      # Larger batches for more stable gradients
        n_epochs       = 8,       # More epochs to better use collected data
        # ——————————————————————————————————————————————————————
        # Total interaction budget per genome - INCREASED
        total_timesteps= 256,    # Much more training time (was 100)
        # ——————————————————————————————————————————————————————
        # Evaluation settings
        eval_interval  = 128,     # Evaluate every 500 steps
        n_evals        = 3,       # Multiple evaluation runs for stability
        n_eval_envs    = 1,       # Single environment for evaluation
        log_interval   = 10,      # Less frequent logging
        verbose_ppo    = 1,       # Standard verbosity
        # ——————————————————————————————————————————————————————
        # Learning & regularization
        learning_rate  = 3e-4,    # Standard PPO learning rate
        gamma          = 0.99,    # Standard discount factor
        gae_lambda     = 0.95,    # Standard GAE parameter
        vf_coef        = 0.5,     # Standard value loss weight
        max_grad_norm  = 0.5,     # Standard gradient clipping
        ent_coef       = 0.01,    # Standard entropy bonus
        clip_range     = 0.2,     # Standard PPO clipping
    )
