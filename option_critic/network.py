from typing import List

import torch
import torch.nn as nn


class OptionCriticNetwork(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_options: int,
        secondary_action_dims: List[int]
    ):
        super().__init__()
        obs_dim = obs_space.shape[0]
        prim_dim = int(action_space.nvec[0])
        assert len(secondary_action_dims) == num_options
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.get_features(obs)
        return self.q_head(feats).max(dim=1)[0]
