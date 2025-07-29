# train_ab_testing.py
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'evogym' / 'examples'))
sys.path.insert(0, str(PROJECT_ROOT / 'evogym' / 'examples' / 'externals' / 'PyTorch-NEAT'))

import pdb
import random

import numpy as np
import torch
from testing.t_args import create_t_args
from testing.t_runner import run_t

from evogym.envs import *

SEED = 42

if __name__ == "__main__":
    # Create default arguments
    args = create_t_args(
        exp_name="test_walker",
        env_name="Walker-v0",  # Make sure this is a valid environment
        pop_size=3,
        structure_shape=(5, 5),
        max_evaluations=9,
        num_cores=3  # Use a smaller number for testing
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # Run the evolutionary process
    best_robot, best_fitness = run_t(args)

    print(f"\nEvolution completed!")
    print(f"Best fitness: {best_fitness}")
    print(f"Best robot structure:\n{best_robot}")
