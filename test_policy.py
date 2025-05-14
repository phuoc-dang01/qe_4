import os

import torch
from env.neat_env import NeatMutationEnv
from option_critic.algorithm import OptionCriticPPO
from option_critic.policy import OptionCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from train_option_critic import (create_option_critic_model, init_train_args,
                                 make_vec_env, setup_neat_config)

device = torch.device("cpu")

import os


def load_option_critic_model(model_path, args):
    """Load model by creating a new one and loading parameters from saved file."""
    import io
    import os
    import zipfile

    import torch
    from stable_baselines3.common.utils import set_random_seed

    # Setup environment
    neat_config = setup_neat_config(args)
    eval_env_fn = lambda: Monitor(NeatMutationEnv(neat_config),
                                  os.path.join(os.path.join(args.save_dir, 'logs'), 'eval'))
    eval_env = make_vec_env(eval_env_fn, 1)
    eval_env = VecNormalize.load("/home/pd468/qe/rl_mutation/models/20250512_2236/vecnormalize.pkl", eval_env)

    # Get action dimensions
    raw_env = NeatMutationEnv(neat_config)
    secondary_dims = [space.n for space in raw_env.action_spaces_consequences]
    raw_env.close()

    # Create a new model with the correct parameters
    set_random_seed(42)  # Ensure reproducibility
    model = create_option_critic_model(eval_env, secondary_dims, args)

    try:
        # Use SB3's load_from_zip_file utility
        from stable_baselines3.common.save_util import load_from_zip_file

        data, params, pytorch_variables = load_from_zip_file(
            model_path,
            device='cpu',
            custom_objects={"policy_class": model.policy.__class__}
        )

        # Update model parameters
        model.set_parameters(params, exact_match=False)  # Use exact_match=False to allow for partial loading

        print("Successfully loaded model parameters!")
    except Exception as e:
        print(f"Error loading model with SB3 utilities: {e}")
        try:
            # Fallback to a more manual method
            print("Trying manual extraction...")
            with zipfile.ZipFile(model_path, 'r') as archive:
                # Try to load params directly
                try:
                    with archive.open('pytorch_variables.pth') as f:
                        pytorch_variables = torch.load(io.BytesIO(f.read()), map_location='cpu')

                    with archive.open('parameters.pth') as f:
                        params = torch.load(io.BytesIO(f.read()), map_location='cpu')

                    # Update model parameters
                    model.set_parameters(params, exact_match=False)
                    print("Successfully loaded model parameters!")
                except Exception as inner_e:
                    print(f"Error in manual extraction: {inner_e}")

                    # Last resort - try to extract the policy parameters only
                    print("Trying to extract policy parameters only...")
                    for filename in archive.namelist():
                        if 'policy' in filename and filename.endswith('.pth'):
                            with archive.open(filename) as f:
                                policy_params = torch.load(io.BytesIO(f.read()), map_location='cpu')

                            # Set only the policy parameters
                            for name, param in model.policy.named_parameters():
                                if name in policy_params:
                                    param.data.copy_(policy_params[name])

                            print(f"Loaded policy parameters from {filename}")
                            break
        except Exception as zip_e:
            print(f"Failed to extract from zip: {zip_e}")

    # Initialize the required attributes if they don't exist
    if not hasattr(model.policy, 'current_option'):
        n_envs = model.n_envs
        model.policy.current_option = torch.zeros(n_envs, dtype=torch.long, device='cpu')
        model.policy.current_bucket = torch.zeros(n_envs, dtype=torch.long, device='cpu')
        model.policy.current_termination_logits = torch.zeros(n_envs, device='cpu')
        model.policy.current_survive_mask = torch.zeros(n_envs, dtype=torch.bool, device='cpu')
        model.policy.eval_options = None

    return model, eval_env

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

# model_path = "/home/pd468/qe/rl_mutation/models/20250512_2236/final_model.zip"
# args = init_train_args()
# model, eval_env = load_option_critic_model(model_path, args)

# # Test the model for 50 steps
# history = test_option_critic_model(model, eval_env, num_steps=10)

# print("After history:")
# # If you want to visualize the option switches over time
# try:
#     # Set non-interactive backend before importing pyplot
#     import matplotlib
#     matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require a display
#     import matplotlib.pyplot as plt

#     # Extract options from history
#     steps = [record['step'] for record in history]
#     options = [record['option'] for record in history]
#     rewards = [record['reward'] for record in history]

#     # Create figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

#     # Plot options
#     ax1.plot(steps, options, marker='o', linestyle='-', markersize=8)
#     ax1.set_ylabel('Option')
#     ax1.set_title('Option Selection Over Time')
#     ax1.grid(True)

#     # Plot rewards
#     ax2.plot(steps, rewards, marker='x', linestyle='-', color='orange')
#     ax2.set_xlabel('Step')
#     ax2.set_ylabel('Reward')
#     ax2.set_title('Rewards Over Time')
#     ax2.grid(True)

#     plt.tight_layout()
#     plt.savefig('option_critic_test.png')
#     plt.close()  # Important: close the figure to free memory
#     print("Created visualization: option_critic_test.png")
# except ImportError:
#     print("Matplotlib not available - skipping visualization")
