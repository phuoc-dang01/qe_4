from experiments.config import ExperimentConfig  # Fix import - use relative
from rl_mutation.training.train_parallel_trainer import ParallelTrainingManager

def run_reward_function_experiments():
    """Run experiments with different reward functions."""

    # Base configuration
    base_config = ExperimentConfig(
        env_name="Walker-v0",
        n_envs=4,
        total_timesteps=50000,
        evaluator_type="dummy"  # Fast for experimentation
    )

    # Create training manager
    manager = ParallelTrainingManager(base_config)

    # Add experiments with different reward functions
    manager.add_experiment("improvement_reward", {
        "reward_function": "improvement",
        "termination_reg": 0.02
    })

    manager.add_experiment("novelty_reward", {
        "reward_function": "novelty",
        "termination_reg": 0.01
    })

    manager.add_experiment("combined_reward", {
        "reward_function": "combined",
        "entropy_reg": 0.03
    })

    # Run experiments (2 in parallel)
    results = manager.run_experiments(n_parallel=2)

    # Analyze results
    for exp_name, model_path in results:
        print(f"Experiment {exp_name} completed: {model_path}")

if __name__ == "__main__":
    run_reward_function_experiments()
