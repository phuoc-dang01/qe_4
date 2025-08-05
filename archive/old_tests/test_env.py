# isort:skip_file
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONBREAKPOINT'] = '0'

import neat
from rl_mutation.env.env_neat import NeatMutationEnv
from option_critic.algorithm import OptionCriticPPO
from option_critic.policy import OptionCriticPolicy
from env.env_callbacks import OptionMetricsCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import wandb
from wandb.integration.sb3 import WandbCallback


def setup_neat_config():
    """Create and configure the NEAT configuration."""
    config_path = os.path.join(os.path.dirname(__file__), 'neat.cfg')

    # Reuse your existing setup_neat_config function
    import rl_mutation.evolution.ppo_evaluation_args as ppo_evaluation_args
    ppo_args = ppo_evaluation_args.create_ppo_eval_args()

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
        extra_info={
            'structure_shape': (5,5),
            'save_path': './models/option_critic',
            'structure_hashes': {},
            'args': ppo_args,
            'env_name': 'Walker-v0',
        }
    )
    return config


neat_config = setup_neat_config()
# Quick smoke-test with minimal settings

# 1) Initialize W&B
wandb.init(
    project="OptionCriticMutation",
    config={
        "num_options": 6,
        "termination_reg": 0.05,
        "entropy_reg": 0.01,
        "n_steps": 16,
        "batch_size": 8,
        "n_epochs": 1,
        "learning_rate": 1e-4,
        "gamma": 0.9,
        "gae_lambda": 0.9,
        "clip_range": 0.1,
    },
    sync_tensorboard=True
)

# 2) Create and wrap environment for logging
env = DummyVecEnv([lambda: Monitor(NeatMutationEnv(neat_config, evaluator_type="dummy"), filename=None)])

# 3) Instantiate the Option-Critic PPO model
model = OptionCriticPPO(
    OptionCriticPolicy,
    env,
    num_options=wandb.config.num_options,
    termination_reg=wandb.config.termination_reg,
    entropy_reg=wandb.config.entropy_reg,
    n_steps=wandb.config.n_steps,
    batch_size=wandb.config.batch_size,
    n_epochs=wandb.config.n_epochs,
    learning_rate=wandb.config.learning_rate,
    gamma=wandb.config.gamma,
    gae_lambda=wandb.config.gae_lambda,
    clip_range=wandb.config.clip_range,
    verbose=1,
    device="cpu",
    policy_kwargs={'secondary_action_dims': env.envs[0].secondary_action_dims}
)

# 4) Configure local logger (stdout + TensorBoard)
new_logger = configure(
    "./logs/tensorboard/",  # local path for TensorBoard logs
    ["stdout", "tensorboard"]
)
model.set_logger(new_logger)

# 5) Create callbacks
wandb_cb = WandbCallback(
    model_save_path="./logs/best_model/",
    gradient_save_freq=100,
    verbose=1,
)
option_cb = OptionMetricsCallback()

# 6) Train with both W&B and OptionMetricsCallback
model.learn(
    total_timesteps=16,
    callback=[wandb_cb, option_cb]
)

# 7) Finish W&B run
wandb.finish()
