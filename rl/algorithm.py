# algorithm.py

from typing import Any, Dict, List, NamedTuple, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

# --- Rollout Buffer Samples ---
class OptionCriticRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    options: torch.Tensor
    buckets: torch.Tensor
    termination_logits: torch.Tensor
    survive_mask: torch.Tensor
    next_observations: Optional[torch.Tensor] = None

# --- Custom Rollout Buffer ---
class OptionCriticRolloutBuffer(RolloutBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = None
        self.buckets = None
        self.termination_logits = None
        self.survive_mask = None
        self.next_observations = None

    def reset(self) -> None:
        super().reset()
        B, E = self.buffer_size, self.n_envs
        obs_shape = self.observation_space.shape
        self.options = torch.zeros((B, E), dtype=torch.long, device=self.device)
        self.buckets = torch.zeros((B, E), dtype=torch.long, device=self.device)
        self.termination_logits = torch.zeros((B, E), device=self.device)
        self.survive_mask = torch.zeros((B, E), device=self.device)
        self.next_observations = torch.zeros((B, E, *obs_shape), device=self.device, dtype=torch.float32)
        self.q_values = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        infos: List[Dict[str, Any]],
    ) -> None:
        value_np = value.detach().cpu()
        logp_np = log_prob.detach().cpu()
        super().add(obs, action, reward, episode_start, value_np, logp_np)
        idx = self.pos - 1

        self.options[idx] = self.policy.current_option.clone()
        self.buckets[idx] = self.policy.current_bucket.clone()
        self.termination_logits[idx] = self.policy.current_termination_logits.clone()
        self.survive_mask[idx] = self.policy.current_survive_mask.clone()

        for env_i in range(self.n_envs):
            info = infos[env_i] if infos and env_i < len(infos) else {}
            raw_nxt = info.get('next_obs', None)
            if raw_nxt is not None:
                t = torch.as_tensor(raw_nxt, device=self.device, dtype=torch.float32)
                if t.dim() > len(self.observation_space.shape):
                    t = t.view(*self.observation_space.shape)
                self.next_observations[idx, env_i] = t

    def get(self, batch_size=None):
        assert self.full
        total = self.buffer_size * self.n_envs
        inds  = np.random.permutation(total)
        bs    = batch_size or total
        for start in range(0, total, bs):
            yield self._get_samples(inds[start : start+bs])

    def _get_samples(self, batch_inds: np.ndarray):
        B, E = self.buffer_size, self.n_envs
        obs_flat       = self.observations.reshape(B*E, *self.observation_space.shape)
        acts_flat      = self.actions.reshape(B*E, *self.action_space.shape)
        vals_flat      = self.values.reshape(B*E)
        logp_flat      = self.log_probs.reshape(B*E)
        adv_flat       = self.advantages.reshape(B*E)
        ret_flat       = self.returns.reshape(B*E)
        opt_flat       = self.options.reshape(B*E)
        buck_flat      = self.buckets.reshape(B*E)
        term_flat      = self.termination_logits.reshape(B*E)
        mask_flat      = self.survive_mask.reshape(B*E)
        next_obs_flat  = self.next_observations.reshape(B*E, *self.observation_space.shape)
        idx_tensor = torch.as_tensor(batch_inds, device=self.device)
        return OptionCriticRolloutBufferSamples(
            observations      = torch.as_tensor(obs_flat[idx_tensor],      device=self.device),
            actions           = torch.as_tensor(acts_flat[idx_tensor],     device=self.device),
            old_values        = torch.as_tensor(vals_flat[idx_tensor],     device=self.device),
            old_log_prob      = torch.as_tensor(logp_flat[idx_tensor],     device=self.device),
            advantages        = torch.as_tensor(adv_flat[idx_tensor],      device=self.device),
            returns           = torch.as_tensor(ret_flat[idx_tensor],      device=self.device),
            options           = opt_flat[idx_tensor],
            buckets           = buck_flat[idx_tensor],
            termination_logits= term_flat[idx_tensor],
            survive_mask      = mask_flat[idx_tensor],
            next_observations = torch.as_tensor(next_obs_flat[idx_tensor], device=self.device),
        )

# --- OptionCriticPPO agent ---
class OptionCriticPPO(PPO):
    buffer_class = OptionCriticRolloutBuffer

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env:    Union[GymEnv, VecEnv],
        num_options:     int = 6,
        termination_reg: float   = 0.01,
        entropy_reg:     float   = 0.001,
        policy_kwargs:   Optional[dict] = None,
        **kwargs
    ):
        self.num_options = num_options
        self.termination_reg = termination_reg
        self.entropy_reg = entropy_reg
        kwargs['_init_setup_model'] = False
        super().__init__(policy, env, policy_kwargs=policy_kwargs, **kwargs)
        self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        n = self.n_envs
        self.policy.current_option = torch.zeros(n, dtype=torch.long, device=self.device)
        self.policy.current_bucket = torch.zeros(n, dtype=torch.long, device=self.device)
        self.policy.current_termination_logits = torch.zeros(n, device=self.device)
        self.policy.current_survive_mask = torch.zeros(n, dtype=torch.bool, device=self.device)

        # optimizers
        self.critic_optimizer = self.policy.optimizer
        self.actor_optimizer = optim.Adam(
            list(self.policy.network.option_policies_primary.parameters()) +
            list(self.policy.network.option_policies_secondary.parameters()),
            lr=self.lr_schedule(1.0),
        )

        # init rollout buffer
        self.rollout_buffer = OptionCriticRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=n,
        )
        self.rollout_buffer.policy = self.policy
        self.policy.rollout_buffer = self.rollout_buffer

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int
    ) -> bool:
        assert self._last_obs is not None
        rollout_buffer.reset()
        callback.on_rollout_start()
        n = 0
        while n < n_rollout_steps:
            obs_t = obs_as_tensor(self._last_obs, self.device)
            actions, values, logp = self.policy(obs_t)
            acts = actions.cpu().numpy()

            step_result = env.step(acts)
            if isinstance(step_result, tuple) and len(step_result) == 5:
                new_obs, rewards, terminated, truncated, infos = step_result
                dones = np.logical_or(terminated, truncated)
            else:
                new_obs, rewards, dones, infos = step_result

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())

            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)

            for i in range(len(infos)):
                if infos[i] is None or not isinstance(infos[i], dict):
                    infos[i] = {}
                infos[i]['next_obs'] = new_obs[i]

            with torch.no_grad():
                feats = self.policy.network.get_features(obs_t)
                q_cur = self.policy.network.get_q_values(feats)
            rollout_buffer.q_values.append(q_cur.cpu().numpy())

            rollout_buffer.add(
                self._last_obs,
                acts,
                rewards,
                self._last_episode_starts,
                values,
                logp,
                infos,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones
            n += 1

        val_t = obs_as_tensor(self._last_obs, self.device)
        _, values, _ = self.policy(val_t)
        values = values.detach()
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        clip_range = self.clip_range(self._current_progress_remaining)
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        termination_loss = 0

        for rollout_data in self.rollout_buffer.get(self.batch_size):
            actions = rollout_data.actions
            options = rollout_data.options
            if len(actions.shape) == 3 and actions.shape[1] == 1:
                actions_reshaped = actions.squeeze(1)
            else:
                actions_reshaped = actions.reshape(actions.size(0), -1)
            self.policy.eval_options = options
            values, log_prob, entropy = self.policy.evaluate_actions(
                rollout_data.observations, actions_reshaped
            )
            del self.policy.eval_options

            advantages = rollout_data.advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            ratio = torch.exp(log_prob - rollout_data.old_log_prob)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss += -torch.min(policy_loss_1, policy_loss_2).mean()
            value_loss += F.mse_loss(rollout_data.returns, values)
            entropy_loss += -entropy.mean()

            mask = rollout_data.survive_mask
            if mask.sum() > 0:
                next_feats = self.policy.network.get_features(rollout_data.next_observations)
                q_next = self.policy.network.get_q_values(next_feats)
                v_next = q_next.max(dim=1)[0]
                batch_idx = torch.arange(q_next.size(0), device=self.device)
                q_option = q_next[batch_idx, options]
                termination_advantage = v_next - q_option
                beta = torch.sigmoid(rollout_data.termination_logits)
                termination_loss += (beta * termination_advantage * mask).mean() * self.termination_reg

        total_loss = policy_loss + 0.5 * value_loss + self.entropy_reg * entropy_loss + termination_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        self.actor_optimizer.step()
        self.logger.dump(self.num_timesteps)
        self._n_updates += 1

    def get_save_params(self) -> Dict[str, Any]:
        data = super().get_save_params()
        data.update({
            "num_options":     self.num_options,
            "termination_reg": self.termination_reg,
            "entropy_reg":     self.entropy_reg,
        })
        return data
