import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import pandas as pd
import seaborn as sns


class MutationAnalyzer:
    """Analyze mutation effectiveness from saved policies."""

    def __init__(self, policy_path: str):
        """Load mutation policy data."""
        with open(policy_path, 'rb') as f:
            self.policy_data = pickle.load(f)

        self.mutation_stats = self.policy_data['mutation_stats']
        self.generation = self.policy_data.get('generation', 0)

    def get_success_rates(self) -> Dict[str, float]:
        """Calculate success rates for each mutation type."""
        success_rates = {}
        for mutation, stats in self.mutation_stats.items():
            if stats['attempts'] > 0:
                success_rates[mutation] = stats['successes'] / stats['attempts']
            else:
                success_rates[mutation] = 0.0
        return success_rates

    def get_average_improvements(self) -> Dict[str, float]:
        """Get average fitness improvements for each mutation type."""
        avg_improvements = {}
        for mutation, stats in self.mutation_stats.items():
            avg_improvements[mutation] = stats.get('avg_improvement', 0.0)
        return avg_improvements

    def plot_mutation_effectiveness(self, save_path: Optional[str] = None):
        """Create visualization of mutation effectiveness."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Success rates
        success_rates = self.get_success_rates()
        mutations = list(success_rates.keys())
        rates = list(success_rates.values())

        ax1.bar(mutations, rates)
        ax1.set_xlabel('Mutation Type')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Mutation Success Rates')
        ax1.set_xticklabels(mutations, rotation=45, ha='right')
        ax1.set_ylim(0, 1)

        # Average improvements
        avg_improvements = self.get_average_improvements()
        improvements = [avg_improvements[m] for m in mutations]

        colors = ['green' if i > 0 else 'red' for i in improvements]
        ax2.bar(mutations, improvements, color=colors)
        ax2.set_xlabel('Mutation Type')
        ax2.set_ylabel('Average Fitness Improvement')
        ax2.set_title('Average Fitness Improvement by Mutation')
        ax2.set_xticklabels(mutations, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_mutation_history(self, mutation_type: str, save_path: Optional[str] = None):
        """Plot improvement history for a specific mutation type."""
        if mutation_type not in self.mutation_stats:
            print(f"Mutation type '{mutation_type}' not found")
            return

        stats = self.mutation_stats[mutation_type]
        improvements = stats.get('recent_improvements', [])

        if not improvements:
            print(f"No improvement history for '{mutation_type}'")
            return

        plt.figure(figsize=(10, 6))

        # Plot individual improvements
        plt.scatter(range(len(improvements)), improvements, alpha=0.6, label='Individual')

        # Plot rolling average
        window = min(10, len(improvements) // 4)
        if window > 1:
            rolling_avg = pd.Series(improvements).rolling(window).mean()
            plt.plot(rolling_avg, color='red', linewidth=2, label=f'{window}-step rolling avg')

        plt.xlabel('Mutation Application')
        plt.ylabel('Fitness Improvement')
        plt.title(f'{mutation_type} Mutation Performance History')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def compare_policies(self, other_policy_path: str, save_path: Optional[str] = None):
        """Compare two mutation policies."""
        other_analyzer = MutationAnalyzer(other_policy_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Compare success rates
        success_rates1 = self.get_success_rates()
        success_rates2 = other_analyzer.get_success_rates()

        mutations = sorted(set(success_rates1.keys()) | set(success_rates2.keys()))

        x = np.arange(len(mutations))
        width = 0.35

        rates1 = [success_rates1.get(m, 0) for m in mutations]
        rates2 = [success_rates2.get(m, 0) for m in mutations]

        ax1.bar(x - width/2, rates1, width, label='Policy 1')
        ax1.bar(x + width/2, rates2, width, label='Policy 2')
        ax1.set_xlabel('Mutation Type')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(mutations, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Compare improvements
        improvements1 = self.get_average_improvements()
        improvements2 = other_analyzer.get_average_improvements()

        imp1 = [improvements1.get(m, 0) for m in mutations]
        imp2 = [improvements2.get(m, 0) for m in mutations]

        ax2.bar(x - width/2, imp1, width, label='Policy 1')
        ax2.bar(x + width/2, imp2, width, label='Policy 2')
        ax2.set_xlabel('Mutation Type')
        ax2.set_ylabel('Average Improvement')
        ax2.set_title('Average Improvement Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(mutations, rotation=45, ha='right')
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self) -> str:
        """Generate text report of mutation statistics."""
        report = []
        report.append("=== Mutation Policy Analysis Report ===")
        report.append(f"Generation: {self.generation}")
        report.append(f"Exploration rate: {self.policy_data.get('exploration_rate', 'N/A')}")
        report.append("")

        success_rates = self.get_success_rates()
        improvements = self.get_average_improvements()

        # Sort by success rate
        sorted_mutations = sorted(success_rates.keys(),
                                key=lambda x: success_rates[x],
                                reverse=True)

        report.append("Mutation Statistics (sorted by success rate):")
        report.append("-" * 50)

        for mutation in sorted_mutations:
            stats = self.mutation_stats[mutation]
            report.append(f"\n{mutation}:")
            report.append(f"  Attempts: {stats['attempts']}")
            report.append(f"  Success rate: {success_rates[mutation]:.3f}")
            report.append(f"  Average improvement: {improvements[mutation]:.4f}")
            report.append(f"  Total improvement: {stats['total_improvement']:.4f}")

        # Best and worst mutations
        if sorted_mutations:
            report.append("\n" + "=" * 50)
            report.append(f"Best mutation: {sorted_mutations[0]} "
                        f"(success rate: {success_rates[sorted_mutations[0]]:.3f})")
            report.append(f"Worst mutation: {sorted_mutations[-1]} "
                        f"(success rate: {success_rates[sorted_mutations[-1]]:.3f})")

        return "\n".join(report)


def analyze_evolution_run(save_path: str):
    """Analyze a complete evolution run."""
    import os
    import glob

    # Find all generation outputs
    generation_files = sorted(glob.glob(os.path.join(save_path, "generation_*/output.txt")))

    if not generation_files:
        print(f"No generation files found in {save_path}")
        return

    # Extract fitness progression
    best_fitnesses = []
    avg_fitnesses = []

    for gen_file in generation_files:
        fitnesses = []
        with open(gen_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t\t')
                if len(parts) == 2:
                    try:
                        fitness = float(parts[1])
                        fitnesses.append(fitness)
                    except ValueError:
                        pass

        if fitnesses:
            best_fitnesses.append(max(fitnesses))
            avg_fitnesses.append(np.mean(fitnesses))

    # Plot fitness progression
    plt.figure(figsize=(10, 6))
    generations = range(len(best_fitnesses))

    plt.plot(generations, best_fitnesses, 'b-', linewidth=2, label='Best')
    plt.plot(generations, avg_fitnesses, 'r--', linewidth=2, label='Average')
    plt.fill_between(generations, avg_fitnesses, alpha=0.3)

    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(save_path, 'fitness_progression.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # If mutation policy exists, analyze it
    policy_path = os.path.join(save_path, 'final_mutation_policy.pkl')
    if os.path.exists(policy_path):
        print("\nAnalyzing mutation policy...")
        analyzer = MutationAnalyzer(policy_path)
        print(analyzer.generate_report())
        analyzer.plot_mutation_effectiveness(
            os.path.join(save_path, 'mutation_effectiveness.png')
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_evolution_run(sys.argv[1])
    else:
        print("Usage: python mutation_analysis.py <path_to_evolution_run>")
