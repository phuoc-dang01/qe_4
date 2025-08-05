#!/usr/bin/env python3
"""
Main script to train RL policies for NEAT mutation.
"""
import argparse
import os
from datetime import datetime
from experiments.config import ExperimentConfig  # Fix import path
from train_option_critic import train_option_critic_quick, train_with_config

def main():
    parser = argparse.ArgumentParser(description="Train RL policy for NEAT mutations")
    parser.add_argument('--mode', choices=['quick', 'full', 'experiment'], default='quick',
                       help='Training mode: quick test, full training, or experiment')
    parser.add_argument('--env-name', default='Walker-v0', help='EvoGym environment')
    parser.add_argument('--timesteps', type=int, default=10000, help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--reward-function', default='improvement',
                       choices=['improvement', 'novelty', 'combined'],
                       help='Reward function to use')
    args = parser.parse_args()

    if args.mode == 'quick':
        # Quick test mode
        print("Running quick test training...")
        model_path, save_dir = train_option_critic_quick()
        print(f"\nQuick training complete!")
        print(f"Model saved to: {model_path}")

    elif args.mode == 'full':
        # Full training with specified parameters
        config = ExperimentConfig(
            env_name=args.env_name,
            n_envs=args.n_envs,
            total_timesteps=args.timesteps,
            reward_function=args.reward_function,
            evaluator_type="dummy"  # Use dummy for faster training
        )

        exp_name = f"{args.env_name}_{args.reward_function}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        _, model_path = train_with_config(exp_name, config)

        print(f"\nFull training complete!")
        print(f"Model saved to: {model_path}")

    elif args.mode == 'experiment':
        # Run multiple experiments
        from rl_mutation.experiments.experiments_run import run_reward_function_experiments  # Fix import
        run_reward_function_experiments()

    # Print integration instructions
    print("\n" + "="*60)
    print("INTEGRATION WITH test_t_run.py:")
    print("="*60)
    print("1. Copy the model path above")
    print("2. In test_t_run.py, add the RL policy path:")
    print("   rl_path = '<your_model_path>'")
    print("3. Update t_population.py to use the RL policy:")
    print("   pop = Population(config, rl_policy_path=rl_path)")
    print("4. Run: python test_t_run.py")
    print("="*60)

if __name__ == "__main__":
    main()
