# option_critic_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import Tensor

# DEVICE
torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OptionCriticNetwork(nn.Module):
    def __init__(self, obs_space, action_space, num_options: int, secondary_action_dims):
        super().__init__()
        obs_dim = obs_space.shape[0]
        prim_dim = int(action_space.nvec[0])
        assert len(secondary_action_dims) == num_options

        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.q_head = nn.Linear(128, num_options)

        self.option_policies_primary = nn.ModuleList(
            [nn.Linear(128, prim_dim) for _ in range(num_options)]
        )
        self.option_policies_secondary = nn.ModuleList(
            [nn.Linear(128, dim) for dim in secondary_action_dims]
        )
        self.termination_heads = nn.ModuleList(
            [nn.Linear(128, 1) for _ in range(num_options)]
        )

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(obs)

    def get_q_values(self, feats: torch.Tensor) -> torch.Tensor:
        return self.q_head(feats)

class OptionCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        secondary_action_dims,
        num_options: int = 6,
        entropy_reg: float = 0.01,
        **kwargs
    ):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        self.mlp_extractor = None  # Disable SB3's default MLP extractor
        self.network = OptionCriticNetwork(
            observation_space, action_space,
            num_options=num_options,
            secondary_action_dims=secondary_action_dims
        )
        self.extract_features = self.network.get_features

        self.num_options = num_options
        self.secondary_action_dims = secondary_action_dims
        self.entropy_reg = entropy_reg

        # Buffers per environment
        n_envs = getattr(self, 'n_envs', 1)
        self.current_option = torch.zeros(n_envs, dtype=torch.long, device=torch_device)
        self.current_bucket = torch.zeros(n_envs, dtype=torch.long, device=torch_device)
        self.current_termination_logits = torch.zeros(n_envs, device=torch_device)
        self.current_survive_mask = torch.zeros(n_envs, dtype=torch.bool, device=torch_device)
        self.eval_options = None

    def _sample_action(self, probs: Tensor, max_dims: Tensor) -> Tensor:
        batch = probs.size(0)
        actions = torch.zeros(batch, dtype=torch.long, device=probs.device)
        for i in range(batch):
            valid = probs[i, : max_dims[i]]
            valid = valid / (valid.sum() + 1e-8)
            actions[i] = torch.multinomial(valid, 1)
        return actions

    def get_action_probs(self, features: Tensor, options: Tensor):
        batch, device = features.size(0), features.device
        max_sec = max(self.secondary_action_dims)

        # Primary action probabilities for each option
        prim_probs = []
        for head in self.network.option_policies_primary:
            logits = head(features)
            if logits.dim() == 3 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            prim_probs.append(F.softmax(logits, dim=-1))
        prim_stack = torch.stack(prim_probs, dim=1)

        # Secondary action probabilities
        sec_list = []
        for head_idx, (head, dim) in enumerate(zip(self.network.option_policies_secondary, self.secondary_action_dims)):
            logits = head(features)
            if logits.dim() > 2:
                logits = logits.view(batch, -1)
            curr_dim = logits.size(1)
            # Adjust size
            if curr_dim > dim:
                logits = logits[:, :dim]
            elif curr_dim < dim:
                pad = logits.new_full((batch, dim - curr_dim), float('-inf'))
                logits = torch.cat([logits, pad], dim=1)
            if dim < max_sec:
                pad = logits.new_full((batch, max_sec - dim), float('-inf'))
                logits = torch.cat([logits, pad], dim=1)
            probs = torch.zeros_like(logits)
            valid_logits = logits[:, :dim]
            valid_probs = F.softmax(valid_logits, dim=-1)
            probs[:, :dim] = valid_probs
            sec_list.append(probs)
        sec_stack = torch.stack(sec_list, dim=1)

        # Select probabilities for the given options
        batch_idx = torch.arange(batch, device=device)
        p1 = prim_stack[batch_idx, options]  # [batch, prim_dim]
        p2 = sec_stack[batch_idx, options]   # [batch, max_sec]

        p1 = p1.reshape(batch, -1)
        p2 = p2.reshape(batch, -1)
        return p1, p2

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the OptionCritic policy.
        This bypasses SB3's default mlp_extractor.
        """
        # OptionCritic logic: a1=primary, a2=secondary
        # Use your own heads, not SB3's!
        acts = actions.reshape(actions.size(0), -1)
        a1 = acts[:, 0].long()
        a2 = acts[:, 1].long()

        feats = self.network.get_features(obs)
        q_logits = self.network.get_q_values(feats)

        batch_size = feats.size(0)
        batch_idx = torch.arange(batch_size, device=obs.device)

        options = getattr(self, 'eval_options', None)
        if options is None:
            options = torch.zeros(batch_size, dtype=torch.long, device=obs.device)

        # Use your own action-head logic:
        p1, p2 = self.get_action_probs(feats, options)
        max_dims = torch.tensor(self.secondary_action_dims, device=obs.device)[options]

        # Compute log probs, entropy, value
        logp = self._compute_log_probs(p1, p2, a1, a2, max_dims)
        ent1 = self._compute_entropy(p1, torch.full_like(a1, p1.size(1), device=obs.device))
        ent2 = self._compute_entropy(p2, max_dims)
        entropy = ent1 + ent2
        value = q_logits[batch_idx, options]

        return value, logp, entropy

    def forward(self, obs: Tensor, deterministic: bool = False):
        feats = self.network.get_features(obs)
        q_logits = self.network.get_q_values(feats)
        batch_idx = torch.arange(feats.size(0), device=obs.device)

        # Termination
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

        # Primary action
        if deterministic:
            a1 = p1.argmax(dim=1)
        else:
            a1 = torch.multinomial(p1, num_samples=1).squeeze(1)
        # Secondary action
        max_dims = torch.tensor(self.secondary_action_dims, device=obs.device)[opts]
        if deterministic:
            mask = torch.arange(p2.size(1), device=obs.device)[None, :] < max_dims[:, None]
            p2m = p2.masked_fill(~mask, 0.0)
            a2 = p2m.argmax(dim=1)
        else:
            a2 = self._sample_action(p2, max_dims)
        self.current_bucket = a2
        actions = torch.stack([a1, a2], dim=1)
        # Value & log-prob
        value = q_logits[batch_idx, opts]
        logp = self._compute_log_probs(p1, p2, a1, a2, max_dims)
        return actions, value, logp

    def _compute_log_probs(self, p1, p2, a1, a2, max_dims):
        if p1.dim() == 3 and p1.size(1) == 1:
            p1 = p1.squeeze(1)
        if p2.dim() == 3 and p2.size(1) == 1:
            p2 = p2.squeeze(1)
        batch_size = p1.size(0)
        device = p1.device
        batch_idx = torch.arange(batch_size, device=device)
        a1 = a1.reshape(-1)
        a2 = a2.reshape(-1)
        a1_valid = torch.clamp(a1, 0, p1.size(1) - 1)
        a2_valid = torch.clamp(a2, 0, p2.size(1) - 1)
        logp1 = torch.log(p1[batch_idx, a1_valid] + 1e-8)
        all_logp2 = torch.log(p2[batch_idx, a2_valid] + 1e-8)
        valid_mask = a2 < max_dims
        logp2 = torch.where(valid_mask, all_logp2, torch.zeros_like(all_logp2))
        return logp1 + logp2

    def _compute_entropy(self, probs, max_dims):
        """
        Compute entropy of the (masked) categorical distributions.
        probs: [batch, n_actions]
        max_dims: [batch] (how many actions are valid in each row)
        Returns: [batch] entropy for each row
        """
        # Safety clamp for numerical stability
        logp = torch.log(probs + 1e-8)
        mask = torch.arange(probs.size(1), device=probs.device)[None, :] < max_dims[:, None]
        entropy = -(probs * logp * mask).sum(dim=1)
        return entropy

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Custom predict for OptionCriticPolicy.
        Returns actions in [primary, secondary] format as needed by the env.
        """
        obs_tensor = torch.as_tensor(observation, device=next(self.parameters()).device, dtype=torch.float32)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        actions, _, _ = self.forward(obs_tensor, deterministic=deterministic)
        actions = actions.cpu().numpy()
        return actions, state
