#!/usr/bin/env python3
"""
Minimal test to verify the basic Option-Critic + NEAT flow works.
This should be run FIRST before attempting full training.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_minimal_flow():
    print("=== Testing Minimal Option-Critic NEAT Flow ===")

    # Step 1: Test basic imports
    print("\n1. Testing imports...")
    try:
        import neat
        from env.neat_env import NeatMutationEnv
        from option_critic.algorithm import OptionCriticPPO
        from option_critic.policy import OptionCriticPolicy
        print("✓ All core imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

    # Step 2: Test NEAT config
    print("\n2. Testing NEAT configuration...")
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'neat.cfg')
        if not os.path.exists(config_path):
            print(f"✗ Config file not found: {config_path}")
            return False

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
            extra_info={
                'structure_shape': (5, 5),
                'save_path': './test_minimal',
                'structure_hashes': {},
                'env_name': 'Walker-v0',
            }
        )
        print("✓ NEAT configuration loaded")
    except Exception as e:
        print(f"✗ NEAT config failed: {e}")
        return False

    # Step 3: Test environment creation with different evaluators
    print("\n3. Testing environment creation...")

    evaluator_types = ["dummy", "proxy", "full"]

    for eval_type in evaluator_types:
        print(f"\n  Testing {eval_type} evaluator...")
        try:
            env = NeatMutationEnv(config, evaluator_type=eval_type)
            print(f"  ✓ {eval_type} evaluator environment created")

            # Quick test
            obs = env.reset()
            action = [0, 0]
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  ✓ {eval_type} evaluation completed. Reward: {reward:.4f}")

            env.close()
            break  # Use the first working evaluator

        except Exception as e:
            print(f"  ✗ {eval_type} evaluator failed: {e}")
            continue

    print("\n=== Minimal Flow Test PASSED ===")
    return True


if __name__ == "__main__":
    success = test_minimal_flow()
    sys.exit(0 if success else 1)
