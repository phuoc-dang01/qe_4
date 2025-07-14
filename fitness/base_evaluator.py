from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEvaluator(ABC):
    """Abstract base class for genome evaluation."""

    @abstractmethod
    def evaluate(self, genome) -> float:
        """Evaluate a genome and return fitness score."""
        pass

class DummyEvaluator(BaseEvaluator):
    """Fast dummy evaluator for RL training."""

    def __init__(self, config=None):
        self.config = config
        self.step_count = 0

    def evaluate(self, genome) -> float:
        """Return a dummy fitness based on genome complexity."""
        self.step_count += 1

        # Simple heuristic: reward complexity but penalize extremes
        n_nodes = len(genome.nodes)
        n_connections = len(genome.connections)

        # Base complexity score
        complexity_score = (n_nodes + n_connections) / 20.0

        # Add some noise to make it interesting for RL
        noise = np.random.normal(0, 0.1)

        # Penalize very simple or very complex genomes
        if n_nodes < 3 or n_connections < 2:
            penalty = -0.5
        elif n_nodes > 20 or n_connections > 50:
            penalty = -0.3
        else:
            penalty = 0

        fitness = complexity_score + noise + penalty

        # Simulate some "improvement" over time to help RL learn
        time_bonus = min(0.1, self.step_count * 0.001)

        return fitness + time_bonus

class ProxyEvaluator(BaseEvaluator):
    """Proxy evaluator that learns to predict real fitness."""

    def __init__(self, config=None):
        self.config = config
        # Could implement a neural network here to predict fitness

    def evaluate(self, genome) -> float:
        """Use learned model to predict fitness."""
        # Placeholder - could implement ML-based fitness prediction
        return DummyEvaluator().evaluate(genome)
