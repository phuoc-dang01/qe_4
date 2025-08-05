from abc import ABC, abstractmethod
from collections import deque
import numpy as np
from fitness.reward_const import RewardConst

class RewardFunction(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def calculate_reward(self, current_fitness, prev_fitness, info):
        pass

class ImprovementReward(RewardFunction):
    """Reward based on *normalized* fitness improvement, optional diversity/rarity bonus."""
    def __init__(self, norm_window=1000):
        self.improvement_buffer = deque(maxlen=norm_window)

    def calculate_reward(self, current_fitness, prev_fitness, info):
        if current_fitness <= RewardConst.INVALID_ROBOT:
            return RewardConst.INVALID_ROBOT, True

        improvement = current_fitness - prev_fitness
        self.improvement_buffer.append(improvement)

        # Compute mean/std from buffer
        arr = np.array(self.improvement_buffer)
        mean = arr.mean() if len(arr) > 1 else 0.0
        std = arr.std() if len(arr) > 1 else 1.0

        # Normalized and clipped
        norm_improvement = (improvement - mean) / (std + 1e-8)
        norm_improvement = np.clip(norm_improvement, -3, 3)

        # Scale for RL stability
        base_reward = float(norm_improvement)

        # --- Optional: add exploration/diversity bonuses ---
        # Diversity bonus
        diversity_bonus = 0.0
        if info and 'diversity' in info:
            diversity_bonus = 0.1 * info['diversity']  # weight as needed
            base_reward += diversity_bonus

        # Rarity bonus (e.g., for using rarely-chosen mutations)
        rarity_bonus = 0.0
        if info and 'rarity' in info:
            rarity_bonus = 0.05 * info['rarity']
            base_reward += rarity_bonus

        # Penalty for invalid/failed mutation selection
        if info and ('mutation_successful' in info) and (not info['mutation_successful']):
            base_reward -= 0.2  # tune as needed

        return base_reward, False

class NoveltyReward(RewardFunction):
    def calculate_reward(self, current_fitness, prev_fitness, info):
        # TODO: Implement structural/behavioral novelty
        pass
