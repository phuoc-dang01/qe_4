
import argparse
import pickle
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--genome-path', required=True)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--idx', type=int, required=True)
    parser.add_argument('--generation', type=int, required=True)
    args = parser.parse_args()

    # Load genome and config
    with open(args.genome_path, 'rb') as f:
        genome = pickle.load(f)

    with open(args.config_path, 'rb') as f:
        config = pickle.load(f)

    # Add the parent directory to the path so we can import the fitness function
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Import the fitness function from the parent directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rl_mutation.t_runner import eval_fitness

    # Run the fitness evaluation
    fitness = eval_fitness(genome, config, args.idx, args.generation)

    # Save the result
    with open(args.output_path, 'wb') as f:
        pickle.dump(fitness, f)

if __name__ == '__main__':
    main()
