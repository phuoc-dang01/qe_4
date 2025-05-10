from typing import List, Tuple

import torch
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import Tensor

# Device
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .network import OptionCriticNetwork


class OptionCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        secondary_action_dims: List[int],
        num_options: int = 6,
        entropy_reg: float = 0.01,
        **kwargs
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # Custom network
        self.network = OptionCriticNetwork(
            observation_space,
            action_space,
            num_options=num_options,
            secondary_action_dims=secondary_action_dims,
        )
        # Replace SB3 feature extractor
        self.extract_features = self.network.get_features
        # if hasattr(self, "mlp_extractor"): del self.mlp_extractor
        # if hasattr(self, "policy_net"): del self.policy_net

        # Hyperparameters
        self.num_options = num_options
        self.secondary_action_dims = secondary_action_dims
        self.entropy_reg = entropy_reg

        # Buffers per environment
        n_envs = getattr(self, 'n_envs', 1)
        self.current_option = torch.zeros(n_envs, dtype=torch.long, device=torch_device)
        self.current_bucket = torch.zeros(n_envs, dtype=torch.long, device=torch_device)
        # Option-Critic extras
        self.current_termination_logits = torch.zeros(n_envs, device=torch_device)
        self.current_survive_mask = torch.zeros(n_envs, dtype=torch.bool, device=torch_device)

        # Add eval_options attribute
        self.eval_options = None

    def _sample_action(self, probs: Tensor, max_dims: Tensor) -> Tensor:
        batch = probs.size(0)
        actions = torch.zeros(batch, dtype=torch.long, device=probs.device)
        for i in range(batch):
            valid = probs[i, : max_dims[i]]
            valid = valid / (valid.sum() + 1e-8)
            actions[i] = torch.multinomial(valid, 1)
        return actions

    def _compute_entropy(self, probs: Tensor, max_dims: Tensor) -> Tensor:
        logp = torch.log(probs + 1e-8)
        mask = torch.arange(probs.size(1), device=probs.device)[None, :] < max_dims[:, None]
        return -(probs * logp * mask).sum(dim=1)

    def get_action_probs(
        self,
        features: Tensor,
        options: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Get action probabilities for given features and options.

        Args:
            features: Extracted state features [batch, feature_dim]
            options: Option indices [batch]

        Returns:
            p1: Primary action probabilities [batch, prim_dim]
            p2: Secondary action probabilities [batch, max_sec_dim]
        """
        batch, device = features.size(0), features.device
        max_sec = max(self.secondary_action_dims)

        # Primary action probabilities for each option
        prim_probs = []
        for head in self.network.option_policies_primary:
            logits = head(features)
            # if it comes out [batch,1,D], drop that extra dim
            if logits.dim() == 3 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            prim_probs.append(F.softmax(logits, dim=-1))

        prim_stack = torch.stack(prim_probs, dim=1)  # [batch, num_options, prim_dim]

        # Secondary action probabilities for each option
        sec_list = []
        for head_idx, (head, dim) in enumerate(zip(self.network.option_policies_secondary, self.secondary_action_dims)):
            logits = head(features)
            if logits.dim() > 2:
                logits = logits.view(batch, -1)

            # Make sure logits has the correct size
            curr_dim = logits.size(1)
            if curr_dim != dim:
                print(f"Warning: Secondary head {head_idx} output dim {curr_dim} doesn't match expected {dim}")
                # Trim or pad as needed
                if curr_dim > dim:
                    logits = logits[:, :dim]
                else:
                    pad = logits.new_full((batch, dim - curr_dim), float('-inf'))
                    logits = torch.cat([logits, pad], dim=1)

            # Always pad to max_sec for consistent stacking
            if dim < max_sec:
                pad = logits.new_full((batch, max_sec - dim), float('-inf'))
                logits = torch.cat([logits, pad], dim=1)

            # Make sure we have the correct shape now
            assert logits.size(1) == max_sec, f"Expected padded size {max_sec}, got {logits.size(1)}"

            # Apply softmax only within the valid dim range
            probs = torch.zeros_like(logits)
            valid_logits = logits[:, :dim]
            valid_probs = F.softmax(valid_logits, dim=-1)
            probs[:, :dim] = valid_probs

            sec_list.append(probs)

        # Now all tensors in sec_list should have the same shape [batch, max_sec]
        sec_stack = torch.stack(sec_list, dim=1)  # [batch, num_options, max_sec]

        # Select probabilities for the given options
        batch_idx = torch.arange(batch, device=device)
        p1 = prim_stack[batch_idx, options]  # [batch, prim_dim]
        p2 = sec_stack[batch_idx, options]   # [batch, max_sec]

        p1 = p1.reshape(batch, -1)
        p2 = p2.reshape(batch, -1)

        return p1, p2

    def forward(
        self,
        obs: Tensor,
        deterministic: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Extract features & Q
        feats = self.network.get_features(obs)
        q_logits = self.network.get_q_values(feats)

        batch_idx = torch.arange(feats.size(0), device=obs.device)
        # Termination logits & survive mask
        term_logits = torch.stack([
            head(feats).squeeze(-1)
            for head in self.network.termination_heads
        ], dim=1)
        cur_opts = self.current_option
        cur_term = term_logits[batch_idx, cur_opts]
        beta = torch.sigmoid(cur_term)
        survive = torch.rand_like(beta) > beta
        self.current_termination_logits = cur_term
        self.current_survive_mask = survive
        # Option selection
        opt_probs = F.softmax(q_logits / self.entropy_reg, dim=-1)
        if deterministic:
            cand_opts = opt_probs.argmax(dim=1)
        else:
            cand_opts = torch.multinomial(opt_probs, num_samples=1).squeeze(1)
        opts = torch.where(survive, cur_opts, cand_opts).reshape(-1)
        self.current_option = opts
        # Action-heads for chosen option
        p1, p2 = self.get_action_probs(feats, opts)
        # Sample or pick primary action
        if deterministic:
            a1 = p1.argmax(dim=1)
        else:
            a1 = torch.multinomial(p1, num_samples=1).squeeze(1)
        # Sample or pick secondary action
        max_dims = torch.tensor(self.secondary_action_dims, device=obs.device)[opts]
        if deterministic:
            mask = torch.arange(p2.size(1), device=obs.device)[None, :] < max_dims[:, None]
            p2m = p2.masked_fill(~mask, 0.0)
            a2 = p2m.argmax(dim=1)
        else:
            a2 = self._sample_action(p2, max_dims)
        self.current_bucket = a2
        # Pack
        actions = torch.stack([a1, a2], dim=1)
        # Value & log-prob
        value = q_logits[batch_idx, opts]
        logp = self._compute_log_probs(p1, p2, a1, a2, max_dims)
        return actions, value, logp

    def _compute_log_probs(
        self,
        p1: Tensor,
        p2: Tensor,
        a1: Tensor,
        a2: Tensor,
        max_dims: Tensor
    ) -> Tensor:
        """
        Compute log probabilities for actions, with safety checks.
        """
        # collapse any accidental extra dims:
        if p1.dim() == 3 and p1.size(1) == 1:
            p1 = p1.squeeze(1)    # now [B, A1]
        if p2.dim() == 3 and p2.size(1) == 1:
            p2 = p2.squeeze(1)    # now [B, A2_max]
        batch_size = p1.size(0)

        device = p1.device
        batch_idx = torch.arange(batch_size, device=device)

        # Ensure a1 and a2 are the right shape
        a1 = a1.reshape(-1)
        a2 = a2.reshape(-1)

        # Ensure same batch size
        if a1.size(0) != batch_size:
            raise ValueError(f"a1 size {a1.size(0)} doesn't match batch size {batch_size}")
        if a2.size(0) != batch_size:
            raise ValueError(f"a2 size {a2.size(0)} doesn't match batch size {batch_size}")

        # Safety check for primary action
        a1_valid = torch.clamp(a1, 0, p1.size(1) - 1)
        if not torch.equal(a1, a1_valid):
            print(f"Warning: Primary actions out of bounds: {a1} vs {a1_valid}")
            a1 = a1_valid

        # Safety check for secondary action
        a2_valid = torch.clamp(a2, 0, p2.size(1) - 1)
        if not torch.equal(a2, a2_valid):
            print(f"Warning: Secondary actions out of bounds: {a2} vs {a2_valid}")
            a2 = a2_valid

        # Compute log probabilities
        logp1 = torch.log(p1[batch_idx, a1] + 1e-8)

        # Get secondary log probabilities
        all_logp2 = torch.log(p2[batch_idx, a2] + 1e-8)

        # Only count valid secondary actions
        valid_mask = a2 < max_dims
        logp2 = torch.where(valid_mask, all_logp2, torch.zeros_like(all_logp2))

        # Sum the log probabilities
        return logp1 + logp2

    def evaluate_actions(
        self,
        obs: Tensor,
        actions: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate actions according to the current policy.

        Args:
            obs: Observations
            actions: Actions to evaluate, could be [batch, 2] or [batch, 1, 2]

        Returns:
            values: State-option values
            log_prob: Log probability of actions
            entropy: Entropy of action distribution
        """
        # Ensure actions have the correct shape [batch, 2]
        if len(actions.shape) == 3 and actions.shape[1] == 1:
            acts = actions.squeeze(1)  # Remove dimension 1 if [batch, 1, 2]
        else:
            acts = actions.reshape(actions.size(0), -1)

        # Extract primary and secondary actions
        a1 = acts[:, 0].long()  # Primary action
        a2 = acts[:, 1].long()  # Secondary action

        # Debug info
        # print(f"evaluate_actions: a1={a1[:5]}, a2={a2[:5]}")

        # Extract features and get Q-values
        feats = self.network.get_features(obs)
        q_logits = self.network.get_q_values(feats)

        # if it comes back as [B,1,num_options], squeeze to [B,num_options]
        if q_logits.dim() == 3 and q_logits.size(1) == 1:
            q_logits = q_logits.squeeze(1)

        batch_size = feats.size(0)

        # Get options from the buffer passed via the policy
        options = getattr(self, 'eval_options', None)

        # CRITICAL: If options is None, we need to handle this gracefully
        if options is None:
            print("Warning: eval_options is None in evaluate_actions")
            # Fall back to using first option for all
            options = torch.zeros(batch_size, dtype=torch.long, device=obs.device)

        # Ensure options has the right shape and type
        if not isinstance(options, torch.Tensor):
            options = torch.tensor(options, dtype=torch.long, device=obs.device)
        if options.dim() == 0:
            options = options.expand(batch_size)
        if len(options) != batch_size:
            # If mismatch, expand or slice as needed
            if len(options) < batch_size:
                options = options.expand(batch_size)
            else:
                options = options[:batch_size]

        # Get action probabilities using options
        p1, p2 = self.get_action_probs(feats, options)

        # Ensure a1 is in bounds
        if a1.max().item() >= p1.size(1):
            print(f"Warning: a1 max {a1.max().item()} is >= p1 cols {p1.size(1)}")
            a1 = torch.clamp(a1, 0, p1.size(1) - 1)

        # Get max dims for secondary actions
        max_dims = torch.tensor(self.secondary_action_dims, device=obs.device)[options]

        # Compute log probabilities, entropy, and value
        logp = self._compute_log_probs(p1, p2, a1, a2, max_dims)

        # Compute entropy
        batch_idx = torch.arange(batch_size, device=obs.device)
        ent1 = self._compute_entropy(p1, torch.full_like(a1, p1.size(1), device=obs.device))
        ent2 = self._compute_entropy(p2, max_dims)
        entropy = ent1 + ent2

        # Value is based on the selected option
        value = q_logits[batch_idx, options]

        return value, logp, entropy
