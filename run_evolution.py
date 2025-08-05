#!/usr/bin/env python3
"""
Main entry point for running NEAT evolution with adaptive mutations.
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add evogym paths if needed
evogym_path = PROJECT_ROOT / 'evogym'
if evogym_path.exists():
    sys.path.insert(0, str(evogym_path / 'examples'))
    sys.path.insert(0, str(evogym_path / 'examples' / 'externals' / 'PyTorch-NEAT'))

from evolution.runner import run_evolution
from evolution.args import create_evolution_args



def main():
    parser = argparse.ArgumentParser(
        description="Run NEAT evolution with adaptive mutation strategies"
    )

    # Basic arguments
    parser.add_argument('--env-name', default='Walker-v0',
                       help='EvoGym environment name')
    parser.add_argument('--generations', type=int, default=50,
                       help='Number of generations to run')
    parser.add_argument('--pop-size', type=int, default=20,
                       help='Population size')
    parser.add_argument('--structure-shape', nargs=2, type=int, default=[5, 5],
                       help='Robot structure shape (height width)')
    parser.add_argument('--num-cores', type=int, default=8,
                       help='Number of CPU cores for parallel evaluation')

    # Adaptive mutation arguments
    parser.add_argument('--use-adaptive', action='store_true', default=True,
                       help='Use adaptive mutation strategies (default: True)')
    parser.add_argument('--no-adaptive', dest='use_adaptive', action='store_false',
                       help='Disable adaptive mutations')
    parser.add_argument('--selection-strategy', default='epsilon_greedy',
                       choices=['epsilon_greedy', 'softmax', 'ucb'],
                       help='Mutation selection strategy')
    parser.add_argument('--exploration-rate', type=float, default=0.3,
                       help='Initial exploration rate for mutations')

    # Transfer learning arguments
    parser.add_argument('--transfer-from', type=str,
                       help='Path to saved mutation policy for transfer learning')
    parser.add_argument('--save-policy', type=str,
                       help='Path to save learned mutation policy')

    # Experiment arguments
    parser.add_argument('--exp-name', type=str,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate experiment name if not provided
    if not args.exp_name:
        args.exp_name = f"{args.env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 60)
    print(f"NEAT Evolution with Adaptive Mutations")
    print("=" * 60)
    print(f"Environment: {args.env_name}")
    print(f"Generations: {args.generations}")
    print(f"Population size: {args.pop_size}")
    print(f"Structure shape: {args.structure_shape}")
    print(f"Adaptive mutations: {args.use_adaptive}")
    if args.transfer_from:
        print(f"Transfer learning from: {args.transfer_from}")
    print("=" * 60)

    # Create evolution arguments
    evolution_args = create_evolution_args(
        exp_name=args.exp_name,
        env_name=args.env_name,
        pop_size=args.pop_size,
        structure_shape=tuple(args.structure_shape),
        max_evaluations=args.generations * args.pop_size,
        num_cores=args.num_cores
    )

    # Run evolution
    best_robot, best_fitness, final_population = run_evolution(
        evolution_args,
        use_adaptive=args.use_adaptive,
        transfer_policy=args.transfer_from,
        selection_strategy=args.selection_strategy,
        exploration_rate=args.exploration_rate
    )

    print("\n" + "=" * 60)
    print("Evolution completed!")
    print(f"Best fitness: {best_fitness:.6f}")
    print(f"Best robot structure:")
    print(best_robot)

    # Save mutation policy if requested
    if args.save_policy and args.use_adaptive:
        os.makedirs(os.path.dirname(args.save_policy) or '.', exist_ok=True)
        final_population.save_mutation_policy(args.save_policy)
        print(f"\nMutation policy saved to: {args.save_policy}")

    # Print mutation report
    if args.use_adaptive:
        print("\n" + final_population.get_mutation_report())

    # Example transfer learning workflow
    if args.save_policy:
        print("\n" + "=" * 60)
        print("To use this policy for transfer learning:")
        print(f"python run_evolution.py --env-name Climber-v0 --transfer-from {args.save_policy}")
        print("=" * 60)


if __name__ == "__main__":
    main()
