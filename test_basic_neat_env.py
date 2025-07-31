import os
import sys

import neat
import numpy as np
from rl_mutation.env.env_neat import NeatMutationEnv


# Simple test to verify NEAT environment works
def test_basic_neat_env():
    print("Testing basic NEAT environment...")

    # Setup basic config
    config_path = os.path.join(os.path.dirname(__file__), 'neat.cfg')

    # Create minimal config for testing
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
        extra_info={
            'structure_shape': (5, 5),
            'save_path': './test_output',
            'structure_hashes': {},
            'env_name': 'Walker-v0',
        }
    )

    # Test environment creation
    try:
        env = NeatMutationEnv(config)
        print("✓ Environment created successfully")

        # Test reset
        obs = env.reset()
        print(f"✓ Environment reset. Observation shape: {obs[0].shape}")

        # Test a few random steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: action={action}, reward={reward:.4f}, done={terminated}")

            if terminated:
                obs = env.reset()
                print("Environment reset due to termination")

        env.close()
        print("✓ Basic environment test completed successfully")
        return True

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_neat_env()
