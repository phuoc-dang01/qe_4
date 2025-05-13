# train_ab_testing.py
import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

sys.path.append('/home/pd468/qe/evogym/examples/')
sys.path.append('/home/pd468/qe/evogym/examples/externals/PyTorch-NEAT/')

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ↑ 3 levels: → rl_mutation → qe
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'evogym' / 'examples' / 'ppo'))
sys.path.insert(0, str(PROJECT_ROOT / 'evogym' / 'examples' / 'externals' / 'PyTorch-NEAT'))

import matplotlib

matplotlib.use('Agg')
import argparse
import pdb
import random

import numpy as np
import torch
from ab_testing.runner import run_ab_testing_experiment
from ppo.args import add_ppo_args

from evogym.envs import *


def add_ab_testing_args(parser):
    """Add A/B testing specific arguments to parser."""
    group = parser.add_argument_group("A/B Testing")
    group.add_argument(
        "--ab_ratio", type=float, default=0.5,
        help="Proportion of population to use RL-guided mutations (0-1)"
    )
    group.add_argument(
        "--gen_report_interval", type=int, default=1,
        help="Generate comparison reports every N generations"
    )
    group.add_argument(
        "--track_ancestry", action="store_true",
        help="Enable detailed ancestry tracking (may slow down evolution)"
    )
    return parser


def create_parser():
    """Create CLI parser with A/B testing support."""
    parser = argparse.ArgumentParser(
        description="Evolve robot designs with CPPN-NEAT + RL-guided mutation with A/B testing"
    )

    # Standard experiment arguments
    parser.add_argument(
        "--exp_name", type=str, default="rl_ab_testing",
        help="Experiment name (used for saving data)"
    )
    parser.add_argument(
        "--config", type=str, default=os.path.join(os.path.dirname(__file__), "neat.cfg"),
        help="Path to NEAT config file"
    )
    parser.add_argument(
        "--env_name", type=str, default="Walker-v0",
        help="Evogym environment name (e.g. Walker-v0)"
    )
    parser.add_argument(
        "--pop_size", type=int, default=10,
        help="Population size for NEAT"
    )
    parser.add_argument(
        "--structure_shape", type=eval, default="(5,5)",  # pass as string: "(5,5)"
        help="Robot structure shape (height,width)"
    )
    parser.add_argument(
        "--max_evaluations", type=int, default=100,
        help="Maximum total genome evaluations"
    )
    parser.add_argument(
        "--num_cores", type=int, default=max(1, mp.cpu_count() - 1),
        help="Number of CPU cores to use"
    )
    parser.add_argument(
        "--rl_policy_path", type=str, default="/home/pd468/qe/rl_mutation/models/20250512_2236/final_model.zip",
        help="Path to trained RL policy model"
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed for reproducibility")

    # Include PPO and A/B testing args
    add_ppo_args(parser)
    add_ab_testing_args(parser)

    return parser

if __name__ == "__main__":
    args = create_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_ab_testing_experiment(args)
