from argparse import Namespace
from types import SimpleNamespace as Namespace


def create_ppo_eval_args():
    """Ultra‐fast PPO eval settings for rapid genome fitness trials."""
    return Namespace(
        # ——————————————————————————————————————————————————————
        # Rollout & update sizes
        n_steps        = 128,     # small horizon ⇒ more frequent updates
        batch_size     = 4,     # tiny minibatches ⇒ more gradient steps per rollout
        n_epochs       = 4,      # few passes over each rollout to avoid stale‐data overfit
        # ——————————————————————————————————————————————————————
        # Total interaction budget per genome
        total_timesteps= 1000,    # only 100 steps—fitness is noisy but cheap
        # ——————————————————————————————————————————————————————
        # Quick self‐evaluation
        eval_interval  = 100,     # log every 25 steps (optional)
        n_evals        = 1,      # single eval rollout per eval_interval
        n_eval_envs    = 1,      # single environment for evaluation
        log_interval   = 5,      # print a line every 5 updates (optional)
        verbose_ppo    = 0,      # minimize console clutter
        # ——————————————————————————————————————————————————————
        # Learning & regularization
        learning_rate  = 5e-4,   # slightly higher LR to adapt fast on tiny data
        gamma          = 0.99,   # shallower discounting for short rollouts
        gae_lambda     = 0.95,    # more bias, less variance in GAE
        vf_coef        = 0.5,    # standard value loss weight
        max_grad_norm  = 0.5,    # keep gradients well‐behaved
        ent_coef       = 0.01,   # minimal entropy bonus
        clip_range     = 0.1,    # standard PPO clipping,

    )

