import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
from ab_testing.ab_population import ABTestingPopulation
from ab_testing.evaluator import GenomeEvaluator


def run_ab_testing_experiment(args: Any) -> Any:
    """Main entry point for A/B testing experiment."""
    import neat
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config,
        custom_config=[('NEAT','pop_size', args.pop_size)]
    )

    # Set extra info for experiment
    structure_shape = args.structure_shape
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join("saved_data",  f"{args.exp_name}_{timestamp}")
    print(save_path)
    os.makedirs(save_path, exist_ok=True)

    # Create subdirectories for checkpoints
    checkpoints_path = os.path.join(save_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    config.extra_info = {
        "structure_shape": structure_shape,
        "save_path": save_path,
        "env_name": args.env_name,
        "structure_hashes": {},
        "args": args,
    }

    # Save experiment metadata
    metadata_path = os.path.join(save_path, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write(f"ENV: {args.env_name}\n")
        f.write(f"STRUCTURE: {structure_shape}\n")
        f.write(f"POP_SIZE: {args.pop_size}\n")
        f.write(f"AB_RATIO: {args.ab_ratio}\n")
        f.write(f"RL_POLICY_PATH: {args.rl_policy_path}\n")

    # Init population
    pop = ABTestingPopulation(
        config=config,
        rl_policy_path=args.rl_policy_path,
        ab_ratio=args.ab_ratio,
    )

    # Add checkpoint reporter
    # checkpoint_interval = max(1, args.max_evaluations // (10 * args.pop_size))  # Save ~10 checkpoints
    # pop.add_reporter(neat.Checkpointer(checkpoint_interval, filename_prefix=f"{checkpoints_path}/neat-checkpoint-"))

    # print(f"Checkpoint interval: every {checkpoint_interval} generations")

    max_evaluations = args.max_evaluations
    generations = int(np.ceil(max_evaluations / args.pop_size))

    print(f"[INIT] Starting A/B test: {args.exp_name}, {generations} generations")

    # Use NEAT's internal run method with constraints
    best_genome = pop.run(
        GenomeEvaluator.batch_eval_fitness,
        GenomeEvaluator.batch_eval_constraint,
        n=generations
    )

    # Final report
    pop.ab_reporter._generate_report(pop.generation - 1)
    print(f"[COMPLETE] Best genome: {best_genome.key}, Fitness: {best_genome.fitness:.4f}")
    return best_genome
