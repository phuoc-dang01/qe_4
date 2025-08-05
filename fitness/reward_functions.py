from abc import ABC, abstractmethod

class RewardFunction(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def calculate_reward(self, current_fitness, prev_fitness, info):
        pass

class ImprovementReward(RewardFunction):
    """Reward based on fitness improvement."""

    def calculate_reward(self, current_fitness, prev_fitness, info):
        if current_fitness <= RewardConst.INVALID_ROBOT:
            return RewardConst.INVALID_ROBOT, True

        improvement = current_fitness - prev_fitness
        reward = min(improvement * 10, RewardConst.POS_REWARD)
        return reward, False

class NoveltyReward(RewardFunction):
    """Reward based on structural novelty."""

    def calculate_reward(self, current_fitness, prev_fitness, info):
        # Implement novelty-based reward
        pass
