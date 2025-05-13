import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

import wandb


class NEATMetricsCallback(BaseCallback):
    """
    Aggregate NEAT‐specific info on every step into in‐memory buffers,
    then flush as SB3 logger records once/rollout.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # --- 1) set up an in‐memory buffer of lists for each metric key

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []

        # Process each environment's info
        for info in infos:
            # Skip empty info dicts
            if not info:
                continue

            # Log directly to wandb for immediate visibility
            if wandb.run is not None:
                wandb_logs = {}
                # Only log metrics that exist in the info dict, excluding fitness
                for k in ["num_nodes", "num_connections", "fitness", "fitness_improvement", "enabled_ratio", "genome_size"]:
                    if k in info:
                        wandb_logs[f"neat/{k}"] = info[k]

                # Also track mutation type specifically
                if "mutation_type" in info:
                    mutation_type = info["mutation_type"]
                    # Create a one-hot encoding for this mutation
                    for i in range(6):  # Assuming 6 mutation types
                        wandb_logs[f"mutations/option_{i}_freq"] = 1.0 if i == mutation_type else 0.0

                # Only log if we have data
                if wandb_logs:
                    wandb.log(wandb_logs)

        return True


class OptionMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Get info from most recent step
        if wandb.run is not None and hasattr(self.model, 'policy'):
            policy = self.model.policy
            # Log only termination probabilities
            if hasattr(policy, 'current_termination_logits'):
                term_logits = policy.current_termination_logits.detach().cpu().numpy()
                betas = 1.0 / (1.0 + np.exp(-term_logits))
                wandb.log({"options/termination_beta": float(np.mean(betas))})
        return True

    def _on_rollout_end(self):
        if wandb.run is not None and hasattr(self.model, 'rollout_buffer'):
            buf = self.model.rollout_buffer

            # Log option statistics from the buffer
            if hasattr(buf, 'options'):
                opts_flat = np.array(buf.options.detach().cpu(), dtype=int).ravel()
                counts = np.bincount(opts_flat, minlength=self.model.num_options)
                freqs = counts / counts.sum()

                wandb_logs = {}
                for i, f in enumerate(freqs):
                    wandb_logs[f"options/freq_option_{i}"] = float(f)

                # Log Q-values if available
                if hasattr(buf, "q_values") and buf.q_values:
                    q_arr = np.vstack(buf.q_values)
                    q_means = q_arr.mean(axis=0)
                    for i, qm in enumerate(q_means):
                        wandb_logs[f"options/mean_q_value_{i}"] = float(qm)
                    # Clear for next rollout
                    buf.q_values = []

                wandb.log(wandb_logs)
