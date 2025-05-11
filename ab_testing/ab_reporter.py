import json
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
        genome_id_list = np.arange(len(population))
        genome_list = np.array(list(population.values()))
        sorted_idx = sorted(genome_id_list, key=lambda i: genome_list[i].fitness, reverse=True)
        genome_id_sorted = list(genome_id_list[sorted_idx])
        genome_sorted = list(genome_list[sorted_idx])
        with open(save_path_ranking, 'w') as f:
            for genome_id, genome in zip(genome_id_sorted, genome_sorted):
                f.write(f'{genome_id}\t\t{genome.fitness}\n')


class ABTestingReporter(neat.BaseReporter):
    """Reporter that tracks and visualizes A/B testing results."""

    def __init__(self, reproduction, save_path, report_interval=10):
        """Initialize the reporter."""
        super().__init__()
        self.reproduction = reproduction
        self.save_path = save_path
        self.report_interval = report_interval
        self.generation = 0
        os.makedirs(os.path.join(save_path, 'ab_reports'), exist_ok=True)

    def start_generation(self, generation):
        """Called at the start of a generation."""
        self.generation = generation

    def update_stats(self, generation, stats_update):
        """Merge stats_update into reproduction.stats and persist to JSON."""
        # Append new stats tuples
        for group in ['rl', 'standard']:
            for key, tup in stats_update[group].items():
                self.reproduction.stats[group].setdefault(key, []).append(tup)
        # Persist full stats to JSON for offline report generation
        stats_path = os.path.join(self.save_path, 'ab_reports', 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.reproduction.stats, f)

    def end_generation(self, config, population, species):
        """Called at the end of a generation."""
        if self.generation % self.report_interval == 0:
            self._generate_report(self.generation)

    def _generate_report(self, generation):
        """Generate visualization report for a given generation."""
        stats = self.reproduction.stats

        rl_f = stats['rl']['fitness']
        std_f = stats['standard']['fitness']
        rl_c = stats['rl']['complexity']
        std_c = stats['standard']['complexity']

        print(f"[DBG] fitness lengths = {len(rl_f)}, {len(std_f)}; complexity lengths = {len(rl_c)}, {len(std_c)}")

        report_path = os.path.join(self.save_path, 'ab_reports', f'generation_{generation}')
        os.makedirs(report_path, exist_ok=True)

        # Only plot as many gens as both arms have data for
        n_f = min(len(rl_f), len(std_f))
        if n_f > 0:
            plt.figure(figsize=(10, 6))
            gens_f = range(n_f)
            # trim to matching length
            rl_min, rl_mean, rl_max = zip(*rl_f[:n_f])
            std_min, std_mean, std_max = zip(*std_f[:n_f])

            plt.plot(gens_f, rl_mean, 'b-', label='RL-guided (mean)')
            plt.fill_between(gens_f, rl_min, rl_max, color='b', alpha=0.2)
            plt.plot(gens_f, std_mean, 'r-', label='Standard (mean)')
            plt.fill_between(gens_f, std_min, std_max, color='r', alpha=0.2)

            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Fitness Comparison: RL-guided vs Standard')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(report_path, 'fitness_comparison.png'))
            plt.close()

        # 2. Complexity comparison
        rl_c = stats['rl']['complexity']
        std_c = stats['standard']['complexity']
        n_c = min(len(rl_c), len(std_c))
        if n_c > 0:
            plt.figure(figsize=(10, 6))
            gens_c = range(n_c)
            rl_min, rl_mean, rl_max = zip(*rl_c[:n_c])
            std_min, std_mean, std_max = zip(*std_c[:n_c])

            plt.plot(gens_c, rl_mean, 'b-', label='RL-guided (mean)')
            plt.fill_between(gens_c, rl_min, rl_max, color='b', alpha=0.2)
            plt.plot(gens_c, std_mean, 'r-', label='Standard (mean)')
            plt.fill_between(gens_c, std_min, std_max, color='r', alpha=0.2)

            plt.xlabel('Generation')
            plt.ylabel('Complexity (nodes + connections)')
            plt.title('Complexity Comparison: RL-guided vs Standard')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(report_path, 'complexity_comparison.png'))
            plt.close()

        # 3. Overall A/B advantage
        rl_means = [m for (_, m, _) in stats['rl']['fitness']]
        std_means = [m for (_, m, _) in stats['standard']['fitness']]
        if rl_means and std_means:
            overall_rl = np.mean(rl_means)
            overall_std = np.mean(std_means)
            advantage = overall_rl - overall_std

            # Text summary
            with open(os.path.join(report_path, 'ab_advantage.txt'), 'w') as f:
                f.write(f"Overall RL avg fitness: {overall_rl:.4f}\n")
                f.write(f"Overall Standard avg fitness: {overall_std:.4f}\n")
                f.write(f"A/B Advantage (RL–Std): {advantage:+.4f}\n")

            # Bar chart
            plt.figure(figsize=(6, 4))
            plt.bar(['RL', 'Standard'], [overall_rl, overall_std])
            plt.ylabel('Average Fitness')
            plt.title('Overall A/B Avg Fitness')
            plt.savefig(os.path.join(report_path, 'ab_advantage.png'))
            plt.close()

        # 4. Generation summary text
        with open(os.path.join(report_path, 'statistics.txt'), 'w') as f:
            f.write(f"A/B Testing Report - Generation {generation}\n")
            f.write("=" * 50 + "\n\n")
            f.write("RL-Guided Group Summary:\n")
            if stats['rl']['fitness']:
                mn, mean, mx = stats['rl']['fitness'][-1]
                f.write(f"  Fitness: min={mn:.2f}, mean={mean:.2f}, max={mx:.2f}\n")
            if stats['rl']['complexity']:
                mn, mean, mx = stats['rl']['complexity'][-1]
                f.write(f"  Complexity: min={mn:.2f}, mean={mean:.2f}, max={mx:.2f}\n")
            f.write("\nStandard Group Summary:\n")
            if stats['standard']['fitness']:
                mn, mean, mx = stats['standard']['fitness'][-1]
                f.write(f"  Fitness: min={mn:.2f}, mean={mean:.2f}, max={mx:.2f}\n")
            if stats['standard']['complexity']:
                mn, mean, mx = stats['standard']['complexity'][-1]
                f.write(f"  Complexity: min={mn:.2f}, mean={mean:.2f}, max={mx:.2f}\n")

    def post_evaluate(self, config, population, species, best_genome):
        # 1) compute this generation's metrics
        stats_update = self.reproduction.parent_population._update_group_stats(
            self.generation, population
        )
        # 2) merge into the rolling stats and persist
        self.update_stats(self.generation, stats_update)
