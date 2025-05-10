# ab_testing/ab_reporter.py
import os

import matplotlib.pyplot as plt
import neat
import numpy as np


class SaveResultReporter(neat.BaseReporter):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation
        save_path_structure = os.path.join(self.save_path, f'generation_{generation}', 'structure')
        save_path_controller = os.path.join(self.save_path, f'generation_{generation}', 'controller')
        os.makedirs(save_path_structure, exist_ok=True)
        os.makedirs(save_path_controller, exist_ok=True)

    def post_evaluate(self, config, population, species, best_genome):
        save_path_ranking = os.path.join(self.save_path, f'generation_{self.generation}', 'output.txt')
        genome_id_list, genome_list = np.arange(len(population)), np.array(list(population.values()))
        sorted_idx = sorted(genome_id_list, key=lambda i: genome_list[i].fitness, reverse=True)
        genome_id_list, genome_list = list(genome_id_list[sorted_idx]), list(genome_list[sorted_idx])
        with open(save_path_ranking, 'w') as f:
            out = ''
            for genome_id, genome in zip(genome_id_list, genome_list):
                out += f'{genome_id}\t\t{genome.fitness}\n'
            f.write(out)


class ABTestingReporter(neat.BaseReporter):
    """Reporter that tracks and visualizes A/B testing results."""

    def __init__(self, reproduction, save_path, report_interval=10):
        """Initialize the reporter."""
        self.reproduction = reproduction
        self.save_path = save_path
        self.report_interval = report_interval
        self.generation = 0
        os.makedirs(os.path.join(save_path, 'ab_reports'), exist_ok=True)

    def start_generation(self, generation):
        """Called at the start of a generation."""
        self.generation = generation

    def end_generation(self, config, population, species):
        """Called at the end of a generation."""
        # Generate report at specified intervals
        if self.generation % self.report_interval == 0:
            self._generate_report(self.generation)

    def _generate_report(self, generation):
        """Generate visualization report."""
        stats = self.reproduction.stats

        # Create report directory
        report_path = os.path.join(self.save_path, 'ab_reports', f'generation_{generation}')
        os.makedirs(report_path, exist_ok=True)

        # 1. Plot fitness comparison
        if stats['rl']['fitness'] and stats['standard']['fitness']:
            plt.figure(figsize=(10, 6))

            # Plot RL group
            rl_gens = range(len(stats['rl']['fitness']))
            rl_min, rl_mean, rl_max = zip(*stats['rl']['fitness'])
            plt.plot(rl_gens, rl_mean, 'b-', label='RL-guided (mean)')
            plt.fill_between(rl_gens, rl_min, rl_max, color='b', alpha=0.2)

            # Plot Standard group
            std_gens = range(len(stats['standard']['fitness']))
            std_min, std_mean, std_max = zip(*stats['standard']['fitness'])
            plt.plot(std_gens, std_mean, 'r-', label='Standard (mean)')
            plt.fill_between(std_gens, std_min, std_max, color='r', alpha=0.2)

            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Fitness Comparison: RL-guided vs Standard Mutations')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(report_path, 'fitness_comparison.png'))
            plt.close()

        # 2. Plot complexity comparison
        if stats['rl']['complexity'] and stats['standard']['complexity']:
            plt.figure(figsize=(10, 6))

            # Plot RL group
            rl_gens = range(len(stats['rl']['complexity']))
            rl_min, rl_mean, rl_max = zip(*stats['rl']['complexity'])
            plt.plot(rl_gens, rl_mean, 'b-', label='RL-guided (mean)')
            plt.fill_between(rl_gens, rl_min, rl_max, color='b', alpha=0.2)

            # Plot Standard group
            std_gens = range(len(stats['standard']['complexity']))
            std_min, std_mean, std_max = zip(*stats['standard']['complexity'])
            plt.plot(std_gens, std_mean, 'r-', label='Standard (mean)')
            plt.fill_between(std_gens, std_min, std_max, color='r', alpha=0.2)

            plt.xlabel('Generation')
            plt.ylabel('Complexity (nodes + connections)')
            plt.title('Complexity Comparison: RL-guided vs Standard Mutations')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(report_path, 'complexity_comparison.png'))
            plt.close()

        # 3. Plot mutation statistics
        # This is more complex and depends on your specific tracking

        # 4. Generate text report
        with open(os.path.join(report_path, 'statistics.txt'), 'w') as f:
            f.write(f"A/B Testing Report - Generation {generation}\n")
            f.write("=" * 50 + "\n\n")

            f.write("RL-Guided Group Summary:\n")
            if stats['rl']['fitness']:
                latest_fitness = stats['rl']['fitness'][-1]
                f.write(f"  Fitness: min={latest_fitness[0]:.2f}, mean={latest_fitness[1]:.2f}, max={latest_fitness[2]:.2f}\n")

            if stats['rl']['complexity']:
                latest_complexity = stats['rl']['complexity'][-1]
                f.write(f"  Complexity: min={latest_complexity[0]:.2f}, mean={latest_complexity[1]:.2f}, max={latest_complexity[2]:.2f}\n")

            f.write("\nStandard Group Summary:\n")
            if stats['standard']['fitness']:
                latest_fitness = stats['standard']['fitness'][-1]
                f.write(f"  Fitness: min={latest_fitness[0]:.2f}, mean={latest_fitness[1]:.2f}, max={latest_fitness[2]:.2f}\n")

            if stats['standard']['complexity']:
                latest_complexity = stats['standard']['complexity'][-1]
                f.write(f"  Complexity: min={latest_complexity[0]:.2f}, mean={latest_complexity[1]:.2f}, max={latest_complexity[2]:.2f}\n")

            # Add more statistics as needed

    def update_stats(self, generation, stats_update):
        """Update statistics with new data."""
        for group, group_stats in stats_update.items():
            for stat_name, stat_value in group_stats.items():
                if stat_name not in self.reproduction.stats[group]:
                    self.reproduction.stats[group][stat_name] = []

                self.reproduction.stats[group][stat_name].append(stat_value)

        # Log the update
        self.info(f"Updated statistics for generation {generation}")
