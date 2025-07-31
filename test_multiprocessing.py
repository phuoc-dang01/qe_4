import multiprocessing as mp
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def test_ppo_creation(idx):
    print(f"Process {os.getpid()} starting test {idx}")

    # Disable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    try:
        # Create a simple env
        env = gym.make('CartPole-v1')
        vec_env = DummyVecEnv([lambda: env])

        # Try to create PPO
        print(f"Process {os.getpid()} creating PPO model...")
        model = PPO('MlpPolicy', vec_env, verbose=0, device='cpu')
        print(f"Process {os.getpid()} PPO created successfully!")

        vec_env.close()
        return True
    except Exception as e:
        print(f"Process {os.getpid()} failed: {e}")
        return False

if __name__ == '__main__':
    # Test serial
    print("Testing serial execution...")
    test_ppo_creation(0)

    # Test parallel
    print("\nTesting parallel execution...")
    with mp.Pool(2) as pool:
        results = pool.map(test_ppo_creation, [1, 2])
        print(f"Results: {results}")
