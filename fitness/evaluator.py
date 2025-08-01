# isort:skip_file

import os
import sys
from pathlib import Path
# sys.path.append('/home/pd468/qe/evogym/examples/ppo/')
# sys.path.append('/home/pd468/qe/evogym/examples/externals/PyTorch-NEAT/')


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # ↑ 3 levels: fitness → rl_mutation → qe
sys.path.insert(0, str(PROJECT_ROOT))
evogym_root = PROJECT_ROOT / 'evogym'
if evogym_root.exists():
    sys.path.insert(0, str(evogym_root))  # For 'import evogym'
    sys.path.insert(0, str(evogym_root / 'examples' / 'ppo'))
    sys.path.insert(0, str(evogym_root / 'examples' / 'externals' / 'PyTorch-NEAT'))
    print(f"✓ Added evogym paths: {evogym_root}")
else:
    print(f"✗ evogym directory not found at: {evogym_root}")


print("Python paths:")
for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
    print(f"  {i}: {path}")


import pdb
import neat
import numpy as np
import torch

# Test imports before using them
try:
    from pytorch_neat.cppn import create_cppn
    print("✓ pytorch_neat import successful")
except ImportError as e:
    print(f"✗ pytorch_neat import failed: {e}")
    # Provide fallback or exit gracefully

try:
    import evogym.envs
    from evogym import get_full_connectivity, has_actuator, is_connected
    print("✓ evogym imports successful")
except ImportError as e:
    print(f"✗ evogym imports failed: {e}")

import neat
import numpy as np
import torch
# Now import the function
from fitness.reward_const import *

from run import run_ppo
from evogym.utils import hashable
from evogym.world import EvoWorld


class FitnessEvaluator:
    """
    Encapsulates everything needed to go from genome → robot → sim → fitness.

    This class handles the full pipeline:
    1. Converting a NEAT genome to a robot design
    2. Validating the robot's structure
    3. Simulating the robot in EvoGym using PPO controllers
    4. Calculating a fitness score with complexity penalties

    Parameters
    ----------
    config : neat.Config
        NEAT configuration with extra_info containing simulation parameters
    """

    def __init__(self, config):
        self.config = config

        # we may want to cache structure_hashes here if we care about uniqueness
        self.hashes = {}

    def genome_to_robot(self, genome: neat.DefaultGenome) -> np.ndarray:
        # Build inputs for the CPPN
        return self.get_robot_from_genome(genome)

    @staticmethod
    def get_cppn_input(structure_shape):
        x, y = torch.meshgrid(torch.arange(structure_shape[0]), torch.arange(structure_shape[1]))
        x, y = x.flatten(), y.flatten()
        center = (np.array(structure_shape) - 1) / 2
        d = ((x - center[0]) ** 2 + (y - center[1]) ** 2).sqrt()
        return x, y, d

    def get_robot_from_genome(self, genome):
        nodes = create_cppn(genome, self.config, leaf_names=['x', 'y', 'd'],
                            node_names=['empty', 'rigid', 'soft', 'hori', 'vert'])
        structure_shape = self.config.extra_info['structure_shape']
        x, y, d = self.get_cppn_input(structure_shape)
        material = []
        for node in nodes:
            material.append(node(x=x, y=y, d=d).numpy())
        material = np.vstack(material).argmax(axis=0)
        robot = material.reshape(structure_shape)
        return robot

    def simulate(self, robot: np.ndarray) -> float:
        try:
            perf = run_ppo(
                self.config.extra_info["args"],
                robot,
                self.config.extra_info["env_name"],
                os.path.join(self.config.extra_info["save_path"], "eval_tmp"),
                "genome_eval",
                get_full_connectivity(robot)
            )
            return float(perf)
        except Exception as e:
            return RewardConst.NEG_REWARD

    def evaluate(self, genome):
        robot = self.genome_to_robot(genome)
        key   = hashable(robot)
        if key in self.hashes:
            raw = self.hashes[key]
        else:
            # quick validity check first
            if not (is_connected(robot) and has_actuator(robot)):
                return RewardConst.INVALID_ROBOT
            else:
                raw = self.simulate(robot)
            self.hashes[key] = raw

        # clamp infinities
        if not np.isfinite(raw):
            raw = raw

        # complexity penalty
        n_nodes, n_conns = genome.size()
        penalty = 1e-3 * (n_nodes + n_conns)
        return raw - penalty
