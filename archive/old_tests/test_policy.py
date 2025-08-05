import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from env.env_neat import NeatMutationEnv
from option_critic.algorithm import OptionCriticPPO
from option_critic.policy import OptionCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from train_option_critic import (create_option_critic_model, init_train_args,
                                 make_vec_env, setup_neat_config)

device = torch.device("cpu")
import os

from stable_baselines3.common.utils import set_random_seed


def load_option_critic_model(model_path=None, args=None):
    """Clean model loading using manual reconstruction."""
    if model_path is None:
        # Your hard-coded path
        model_path = "/home/pd468/qe/rl_mutation/models/quick_test_20250714_0107/final_model.zip"

    if args is None:
        args = init_train_args()

    # Setup environment (same as training)
    neat_config = setup_neat_config(args)
    eval_env_fn = lambda: Monitor(NeatMutationEnv(neat_config))
    eval_env = make_vec_env(eval_env_fn, 1)

    # Get secondary action dimensions
    raw_env = NeatMutationEnv(neat_config)
    secondary_dims = [space.n for space in raw_env.action_spaces_consequences]
    raw_env.close()
    print(f"✓ Secondary action dimensions: {secondary_dims}")

    # Load VecNormalize if it exists
    vecnormalize_path = model_path.replace('final_model.zip', 'vecnormalize.pkl')
    if os.path.exists(vecnormalize_path):
        eval_env = VecNormalize.load(vecnormalize_path, eval_env)
        print(f"✓ VecNormalize loaded from {vecnormalize_path}")
    else:
        print(f"⚠ VecNormalize not found at {vecnormalize_path}, skipping")

    set_random_seed(42)

    try:
        # Method 1: Create new model and load state dict
        print("Trying to create new model and load weights...")

        # Create a new model with the correct parameters
        model = OptionCriticPPO(
            policy=OptionCriticPolicy,
            env=eval_env,
            policy_kwargs={
                'secondary_action_dims': secondary_dims,
                'num_options': 6,
                'entropy_reg': 0.02,
            },
            learning_rate=3e-4,  # Default from training
            n_steps=16,
            batch_size=16,
            n_epochs=2,
            gamma=0.9,
            gae_lambda=0.8,
            clip_range=0.1,
            seed=42,
            verbose=0,
            device='cpu',
            num_options=6,
            termination_reg=0.02,
        )

        # Load the saved parameters
        import zipfile

        import torch

        with zipfile.ZipFile(model_path, 'r') as zip_file:
            # Load the policy parameters
            with zip_file.open('policy.pth') as f:
                 policy_state = torch.load(f, map_location='cpu', weights_only=True)

            # Load other parameters if they exist
            try:
                with zip_file.open('pytorch_variables.pth') as f:
                    pytorch_vars  = torch.load(f, map_location='cpu', weights_only=True)
            except KeyError:
                pytorch_vars = {}

        # Load the state into the model
        model.policy.load_state_dict(policy_state)

        print("✓ Model weights loaded successfully")

        # Initialize required attributes
        n_envs = model.n_envs
        model.policy.current_option = torch.zeros(n_envs, dtype=torch.long, device='cpu')
        model.policy.current_bucket = torch.zeros(n_envs, dtype=torch.long, device='cpu')
        model.policy.current_termination_logits = torch.zeros(n_envs, device='cpu')
        model.policy.current_survive_mask = torch.zeros(n_envs, dtype=torch.bool, device='cpu')
        model.policy.eval_options = None
        print("✓ Policy attributes initialized")

        return model, eval_env

    except Exception as e:
        print(f"Manual loading failed: {e}")
        print("Trying simplified approach...")

        # Method 2: Simple approach - create model and ignore loading for now
        try:
            model = OptionCriticPPO(
                policy=OptionCriticPolicy,
                env=eval_env,
                policy_kwargs={
                    'secondary_action_dims': secondary_dims,
                    'num_options': 6,
                    'entropy_reg': 0.02,
                },
                learning_rate=3e-4,
                n_steps=16,
                batch_size=16,
                n_epochs=2,
                gamma=0.9,
                gae_lambda=0.8,
                clip_range=0.1,
                seed=42,
                verbose=0,
                device='cpu',
                num_options=6,
                termination_reg=0.02,
            )

            print("⚠ Created new model (not loading weights - for testing only)")

            # Initialize required attributes
            n_envs = model.n_envs
            model.policy.current_option = torch.zeros(n_envs, dtype=torch.long, device='cpu')
            model.policy.current_bucket = torch.zeros(n_envs, dtype=torch.long, device='cpu')
            model.policy.current_termination_logits = torch.zeros(n_envs, device='cpu')
            model.policy.current_survive_mask = torch.zeros(n_envs, dtype=torch.bool, device='cpu')
            model.policy.eval_options = None

            return model, eval_env

        except Exception as e2:
            print(f"All loading methods failed: {e2}")
            raise


def test_option_critic_model(model, eval_env, num_steps=20):
    """
    Test the Option-Critic model for multiple steps to verify option selection.

    Args:
        model: The loaded OptionCriticPPO model
        eval_env: The evaluation environment
        num_steps: Number of steps to run

    Returns:
        A record of options and actions chosen
    """
    # Reset the environment
    obs = eval_env.reset()

    # Initialize records
    history = []

    print(f"{'Step':^5} | {'Option':^10} | {'Primary':^10} | {'Secondary':^10}")
    print("-" * 45)

    # Run for num_steps
    for step in range(num_steps):
        # Convert observation to tensor
        obs_t = torch.tensor(obs, dtype=torch.float32)

        # Get actions using the policy
        with torch.no_grad():
            actions_t, _, _ = model.policy.forward(obs_t, deterministic=False)  # Use stochastic actions

        # Convert to numpy for environment
        actions = actions_t.cpu().numpy()

        # Get option chosen
        option = model.policy.current_option.cpu().numpy()[0]

        # Execute action in environment
        next_obs, reward, done, info = eval_env.step(actions)

        # Unpack the actions
        a1, a2 = actions[0]

        # Record results
        history.append({
            'step': step,
            'option': int(option),
            'primary_action': int(a1),
            'secondary_action': int(a2),
            'reward': float(reward[0]),
            'done': bool(done[0])
        })

        # Print current step info
        print(f"{step:^5} | {option:^10} | {a1:^10} | {a2:^10}")

        # Update observation
        obs = next_obs

        # Reset if episode is done
        if done.any():
            obs = eval_env.reset()
            print("-" * 45 + " Environment Reset " + "-" * 45)

    # Analyze option switches
    option_switches = sum(1 for i in range(1, len(history))
                        if history[i]['option'] != history[i-1]['option'])

    print("\nSummary:")
    print(f"Total steps: {num_steps}")
    print(f"Number of option switches: {option_switches}")
    print(f"Option switch rate: {option_switches/(num_steps-1)*100:.2f}%")

    # Count options chosen
    from collections import Counter
    option_counts = Counter(record['option'] for record in history)
    print("\nOption distribution:")
    for option, count in sorted(option_counts.items()):
        print(f"Option {option}: {count} times ({count/num_steps*100:.2f}%)")

    return history
