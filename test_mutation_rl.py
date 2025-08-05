import sys
import os
import neat
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from evolution.envs.mutation_env import NeatMutationEnv
from rl.option_critic_policy import OptionCriticPolicy
from rl.algorithm import OptionCriticPPO
from evolution.args import create_evolution_args
from evolution.ppo_evaluation_args import create_ppo_eval_args


from stable_baselines3.common.vec_env import DummyVecEnv

# === 1. Evolution and RL setup ===
# Create evolution args just as in evolution pipeline
evolution_args = create_evolution_args(
    exp_name='mutation_rl_test',
    env_name='Walker-v0',
    pop_size=20,                   # Keep population tiny for fast test
    structure_shape=(5, 5),
    max_evaluations=100,            # Only 2 rollouts per env
    num_cores=8
)

# Load NEAT config with the same env/robot structure
config_path = "evolution/neat.cfg"
config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path
)

# Build the extra_info block as expected by the real fitness core
config.extra_info = dict(
    args=create_ppo_eval_args(),             # <--- your ppo args builder
    env_name=evolution_args.env_name,
    save_path="./tmp_mutation_rl_test",
    structure_shape=evolution_args.structure_shape
)

# === 2. Make real fitness envs ===
def make_env(rank=0):
    def _init():
        env = NeatMutationEnv(config, evaluator_type="real")
        env.reset(seed=rank)
        return env
    return _init

n_envs = os.cpu_count()//2  # Use half the cores for parallelism
vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

# === 3. Policy & PPO setup ===
dummy_env = vec_env.envs[0]
secondary_action_dims = dummy_env.secondary_action_dims
policy_kwargs = dict(
    secondary_action_dims=secondary_action_dims,
    num_options=len(secondary_action_dims),
)

model = OptionCriticPPO(
    policy=OptionCriticPolicy,
    env=vec_env,
    policy_kwargs=policy_kwargs,
    n_steps=128,         # Rollout length per env
    batch_size=256,      # Batch for PPO update (must be divisible by n_envs)
    n_epochs=8,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
)

print("\n=== Training OptionCriticPPO on NeatMutationEnv (REAL EvoGym fitness, test run) ===")
model.learn(total_timesteps=200_000)  # minimal run for quick feedback

model.save("oc_mutation_policy_test_real.zip")
print("\nSaved model as oc_mutation_policy_test_real.zip")

# --- Evaluate on a single rollout (optional) ---
obs = vec_env.reset()
for _ in range(2):
    action, _states = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    print("Reward:", reward)
