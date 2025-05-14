# First, let's add a new file called metrics_tracker.py

import matplotlib

matplotlib.use('Agg')

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class MetricsTracker:
    """Tracks comprehensive metrics for A/B testing experiments."""
    def __init__(self, save_path):
        """Initialize the metrics tracker."""
        self.save_path = save_path
        self.metrics_dir = os.path.join(save_path, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        # Main data structures
        self.genome_history = defaultdict(list)  # Tracks history for each genome
        self.generation_stats = {}  # Changed from defaultdict(list) to regular dict
        self.mutation_effects = {"rl": [], "standard": []}  # Fitness changes by mutation type
        self.action_effects = defaultdict(list)  # Effects by action type (for RL mutations)
        self.cumulative_improvements = {"rl": 0.0, "standard": 0.0}
        self.peak_performance = {"rl": {"fitness": -float('inf'), "generation": -1, "genome_id": None},
                            "standard": {"fitness": -float('inf'), "generation": -1, "genome_id": None}}
        self.worst_performance = {"rl": {"fitness": float('inf'), "generation": -1, "genome_id": None},
                                "standard": {"fitness": float('inf'), "generation": -1, "genome_id": None}}

    def record_genome(self, generation, genome_id, genome, group, parent_ids=None,
                    pre_mutation_fitness=None, post_mutation_fitness=None):
        """Record data for a single genome."""
        if not hasattr(genome, "fitness") or genome.fitness is None:
            return

        # Try to find mutation effect data if not directly provided
        if (pre_mutation_fitness is None or post_mutation_fitness is None):
            # Look for this genome in mutation effects
            for effect in self.mutation_effects.get(group, []):
                if effect.get("genome_id") == genome_id:
                    pre_mutation_fitness = effect.get("pre_fitness")
                    post_mutation_fitness = effect.get("post_fitness")
                    break

        record = {
            "generation": generation,
            "genome_id": genome_id,
            "group": group,
            "fitness": genome.fitness if hasattr(genome, "fitness") and genome.fitness is not None else -1.0,
            "nodes": len(genome.nodes),
            "connections": len(genome.connections),
            "enabled_connections": sum(1 for conn in genome.connections.values() if conn.enabled),
            "parent_ids": parent_ids,
            "pre_mutation_fitness": pre_mutation_fitness,
            "post_mutation_fitness": post_mutation_fitness
        }

        self.genome_history[genome_id].append(record)

        # Track mutation effects if we have before/after data
        if pre_mutation_fitness is not None and post_mutation_fitness is not None:
            improvement = post_mutation_fitness - pre_mutation_fitness

            # Check if this mutation effect is already recorded
            already_recorded = False
            for effect in self.mutation_effects.get(group, []):
                if effect.get("genome_id") == genome_id:
                    already_recorded = True
                    break

            if not already_recorded:
                # This is a new mutation effect - add it to records and update cumulative improvements
                self.mutation_effects[group].append({
                    "genome_id": genome_id,
                    "generation": generation,
                    "pre_fitness": pre_mutation_fitness,
                    "post_fitness": post_mutation_fitness,
                    "improvement": improvement
                })

                # Only update cumulative improvements for new records with positive improvement
                if improvement > 0:
                    self.cumulative_improvements[group] += improvement

        # Update peak and worst performance
        if record["fitness"] != -1.0:
            if record["fitness"] > self.peak_performance[group]["fitness"]:
                self.peak_performance[group] = {
                    "fitness": record["fitness"],
                    "generation": generation,
                    "genome_id": genome_id
                }
            if record["fitness"] < self.worst_performance[group]["fitness"]:
                self.worst_performance[group] = {
                    "fitness": record["fitness"],
                    "generation": generation,
                    "genome_id": genome_id
                }

    def record_mutation_effect(self, genome_id, group, pre_fitness, post_fitness, best_action=None):
        """Record the effect of a mutation on fitness."""
        if pre_fitness is None or post_fitness is None:
            return  # Skip if we don't have before/after data

        improvement = post_fitness - pre_fitness

        # Check if this effect is already recorded
        already_recorded = False
        for effect in self.mutation_effects.get(group, []):
            if effect.get("genome_id") == genome_id:
                already_recorded = True
                break

        if already_recorded:
            # If already recorded, just update the existing record - don't add to cumulative
            for effect in self.mutation_effects[group]:
                if effect.get("genome_id") == genome_id:
                    effect["pre_fitness"] = pre_fitness
                    effect["post_fitness"] = post_fitness
                    effect["improvement"] = improvement
                    if group == "rl" and best_action is not None:
                        effect["best_action"] = best_action
                    break
        else:
            # This is a new effect - add it to records and update cumulative improvements
            effect = {
                "genome_id": genome_id,
                "pre_fitness": pre_fitness,
                "post_fitness": post_fitness,
                "improvement": improvement,
                "best_action": best_action
            }
            self.mutation_effects[group].append(effect)

            # Track which actions produce the best results (for RL mutations)
            if group == "rl" and best_action is not None:
                action_str = str(best_action)  # Convert the action to a string key
                self.action_effects[action_str].append({
                    "genome_id": genome_id,
                    "improvement": improvement
                })

            # Only update cumulative improvements for new records with positive improvement
            if improvement > 0:
                self.cumulative_improvements[group] += improvement

    def record_generation_stats(self, generation, population, group_assignments):
        """Record statistics for an entire generation."""

        rl_genomes = []
        std_genomes = []

        for gid, g in population.items():
            if group_assignments.get(gid, "") == "rl":
                rl_genomes.append(g)
            elif group_assignments.get(gid, "") == "standard":
                std_genomes.append(g)

        # Get fitness values, handling None or non-finite values
        rl_fitness = [g.fitness for g in rl_genomes if hasattr(g, "fitness") and
                     g.fitness is not None and np.isfinite(g.fitness)]
        std_fitness = [g.fitness for g in std_genomes if hasattr(g, "fitness") and
                      g.fitness is not None and np.isfinite(g.fitness)]

        # Compute stats
        stats = {
            "generation": generation,
            "rl": {
                "count": len(rl_genomes),
                "fitness_min": min(rl_fitness) if rl_fitness else None,
                "fitness_max": max(rl_fitness) if rl_fitness else None,
                "fitness_mean": np.mean(rl_fitness) if rl_fitness else None,
                "fitness_std": np.std(rl_fitness) if rl_fitness else None,
                "complexity_mean": np.mean([len(g.nodes) + len(g.connections) for g in rl_genomes]) if rl_genomes else None
            },
            "standard": {
                "count": len(std_genomes),
                "fitness_min": min(std_fitness) if std_fitness else None,
                "fitness_max": max(std_fitness) if std_fitness else None,
                "fitness_mean": np.mean(std_fitness) if std_fitness else None,
                "fitness_std": np.std(std_fitness) if std_fitness else None,
                "complexity_mean": np.mean([len(g.nodes) + len(g.connections) for g in std_genomes]) if std_genomes else None
            }
        }

        self.generation_stats[generation] = stats

        # Calculate per-generation improvement
        if generation > 0 and generation-1 in self.generation_stats:
            prev_gen = self.generation_stats[generation-1]

            # RL improvement
            if stats["rl"]["fitness_mean"] is not None and prev_gen["rl"]["fitness_mean"] is not None:
                stats["rl"]["improvement"] = max(0, stats["rl"]["fitness_mean"] - prev_gen["rl"]["fitness_mean"])
            else:
                stats["rl"]["improvement"] = 0

            # Standard improvement
            if stats["standard"]["fitness_mean"] is not None and prev_gen["standard"]["fitness_mean"] is not None:
                stats["standard"]["improvement"] = max(0, stats["standard"]["fitness_mean"] - prev_gen["standard"]["fitness_mean"])
            else:
                stats["standard"]["improvement"] = 0
        else:
            # First generation has no improvement
            stats["rl"]["improvement"] = 0
            stats["standard"]["improvement"] = 0

    def save_metrics(self):
        """Save all tracked metrics to disk."""
        try:
            # Helper function to convert NumPy types to Python types
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            # Save genome history as jsonl (one line per record for efficiency)
            with open(os.path.join(self.metrics_dir, "genome_history.jsonl"), "w") as f:
                for genome_id, records in self.genome_history.items():
                    for record in records:
                        # Convert record to serializable format
                        serializable_record = convert_numpy_types({"genome_id": genome_id, **record})
                        f.write(json.dumps(serializable_record) + "\n")

            # Save generation stats
            with open(os.path.join(self.metrics_dir, "generation_stats.json"), "w") as f:
                json.dump(convert_numpy_types(self.generation_stats), f, indent=2)

            # Save mutation effects
            with open(os.path.join(self.metrics_dir, "mutation_effects.json"), "w") as f:
                json.dump(convert_numpy_types(self.mutation_effects), f, indent=2)

            # Save action effects
            with open(os.path.join(self.metrics_dir, "action_effects.json"), "w") as f:
                json.dump(convert_numpy_types(dict(self.action_effects)), f, indent=2)

            # Save cumulative improvements
            with open(os.path.join(self.metrics_dir, "cumulative_improvements.json"), "w") as f:
                json.dump(convert_numpy_types(self.cumulative_improvements), f, indent=2)

            # Save peak performance
            with open(os.path.join(self.metrics_dir, "peak_performance.json"), "w") as f:
                json.dump(convert_numpy_types(self.peak_performance), f, indent=2)

            # Save worst performance
            with open(os.path.join(self.metrics_dir, "worst_performance.json"), "w") as f:
                json.dump(convert_numpy_types(self.worst_performance), f, indent=2)

        except Exception as e:
            print(f"Error saving metrics: {e}")
            import traceback
            traceback.print_exc()

    def generate_reports(self):
        """Generate comprehensive reports and visualizations."""
        self._generate_mutation_effect_report()
        self._generate_cumulative_improvement_report()
        self._generate_peak_timing_report()
        self._generate_fitness_trajectory_report()
        self._generate_action_analysis_report()
        self._generate_summary_report()

    def _generate_mutation_effect_report(self):
        """Generate a report showing the effect of mutations."""
        # Also look through genome history for mutation data
        for genome_id, records in self.genome_history.items():
            for record in records:
                if (record.get("pre_mutation_fitness") is not None and
                    record.get("post_mutation_fitness") is not None):

                    group = record.get("group")
                    if group not in ["rl", "standard"]:
                        continue

                    # Check if this mutation is already recorded
                    already_recorded = False
                    for mut in self.mutation_effects.get(group, []):
                        if mut.get("genome_id") == genome_id:
                            already_recorded = True
                            break

                    if not already_recorded:
                        # Add to mutation effects
                        mutation = {
                            "genome_id": genome_id,
                            "pre_fitness": record.get("pre_mutation_fitness"),
                            "post_fitness": record.get("post_mutation_fitness"),
                            "improvement": record.get("post_mutation_fitness") - record.get("pre_mutation_fitness")
                        }
                        self.mutation_effects.setdefault(group, []).append(mutation)

        if not self.mutation_effects["rl"] and not self.mutation_effects["standard"]:
            return

        # Convert to DataFrame for easier analysis
        rl_df = pd.DataFrame(self.mutation_effects["rl"])
        std_df = pd.DataFrame(self.mutation_effects["standard"])

        if not rl_df.empty and not std_df.empty:
            plt.figure(figsize=(12, 10))

            # Box plot of improvement distribution
            plt.subplot(2, 2, 1)
            data = [rl_df["improvement"], std_df["improvement"]]
            plt.boxplot(data, labels=["RL-guided", "Standard"])
            plt.title("Mutation Effect Distribution")
            plt.ylabel("Fitness Improvement")

            # Violin plot for more detailed distribution view
            plt.subplot(2, 2, 2)
            if len(rl_df) > 5 and len(std_df) > 5:  # Only if we have enough data
                combined = pd.concat([
                    pd.DataFrame({"improvement": rl_df["improvement"], "type": "RL-guided"}),
                    pd.DataFrame({"improvement": std_df["improvement"], "type": "Standard"})
                ])
                sns.violinplot(x="type", y="improvement", data=combined)
                plt.title("Mutation Effect Distribution (Detailed)")

            # Histogram of improvements
            plt.subplot(2, 2, 3)
            plt.hist(rl_df["improvement"], alpha=0.5, label="RL-guided", bins=20)
            plt.hist(std_df["improvement"], alpha=0.5, label="Standard", bins=20)
            plt.legend()
            plt.title("Histogram of Fitness Improvements")

            # Scatter plot of before vs after fitness
            plt.subplot(2, 2, 4)
            plt.scatter(rl_df["pre_fitness"], rl_df["post_fitness"], alpha=0.5, label="RL-guided")
            plt.scatter(std_df["pre_fitness"], std_df["post_fitness"], alpha=0.5, label="Standard")
            plt.plot([0, max(rl_df["pre_fitness"].max(), std_df["pre_fitness"].max())],
                     [0, max(rl_df["pre_fitness"].max(), std_df["pre_fitness"].max())],
                     'k--')  # Diagonal line
            plt.xlabel("Pre-Mutation Fitness")
            plt.ylabel("Post-Mutation Fitness")
            plt.legend()
            plt.title("Fitness Before vs After Mutation")

            plt.tight_layout()
            plt.savefig(os.path.join(self.metrics_dir, "mutation_effect.png"))
            plt.close()

            # Summary statistics
            with open(os.path.join(self.metrics_dir, "mutation_effect_stats.txt"), "w") as f:
                f.write("RL-guided Mutation Effects:\n")
                f.write(f"  Count: {len(rl_df)}\n")
                f.write(f"  Mean Improvement: {rl_df['improvement'].mean():.4f}\n")
                f.write(f"  Median Improvement: {rl_df['improvement'].median():.4f}\n")
                f.write(f"  Std Dev: {rl_df['improvement'].std():.4f}\n")
                f.write(f"  Positive Improvements: {(rl_df['improvement'] > 0).sum()} ({(rl_df['improvement'] > 0).mean()*100:.1f}%)\n\n")

                f.write("Standard Mutation Effects:\n")
                f.write(f"  Count: {len(std_df)}\n")
                f.write(f"  Mean Improvement: {std_df['improvement'].mean():.4f}\n")
                f.write(f"  Median Improvement: {std_df['improvement'].median():.4f}\n")
                f.write(f"  Std Dev: {std_df['improvement'].std():.4f}\n")
                f.write(f"  Positive Improvements: {(std_df['improvement'] > 0).sum()} ({(std_df['improvement'] > 0).mean()*100:.1f}%)\n")

    def _generate_cumulative_improvement_report(self):
        """Generate a report showing cumulative improvements over generations."""
        generations = sorted(self.generation_stats.keys())
        if not generations:
            return

        # Extract per-generation improvements
        rl_improvements = [self.generation_stats[g]["rl"]["improvement"] for g in generations]
        std_improvements = [self.generation_stats[g]["standard"]["improvement"] for g in generations]

        # Compute cumulative sums
        rl_cumulative = np.cumsum(rl_improvements)
        std_cumulative = np.cumsum(std_improvements)

        # Plot cumulative improvements
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.bar(range(len(generations)), rl_improvements, label="RL-guided", alpha=0.7)
        plt.bar(range(len(generations)), std_improvements, bottom=rl_improvements, label="Standard", alpha=0.7)
        plt.xlabel("Generation")
        plt.ylabel("Improvement")
        plt.title("Per-Generation Fitness Improvements")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(generations, rl_cumulative, 'b-', label="RL-guided")
        plt.plot(generations, std_cumulative, 'r-', label="Standard")
        plt.xlabel("Generation")
        plt.ylabel("Cumulative Improvement")
        plt.title("Cumulative Fitness Improvements")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, "cumulative_improvements.png"))
        plt.close()

        # Save summary
        with open(os.path.join(self.metrics_dir, "cumulative_improvement_stats.txt"), "w") as f:
            f.write(f"Total Cumulative Improvement (RL-guided): {rl_cumulative[-1]:.4f}\n")
            f.write(f"Total Cumulative Improvement (Standard): {std_cumulative[-1]:.4f}\n")
            f.write(f"Difference (RL - Standard): {rl_cumulative[-1] - std_cumulative[-1]:+.4f}\n")

    def _generate_peak_timing_report(self):
        """Generate a report showing when peak performance was reached."""
        plt.figure(figsize=(10, 6))

        # Plot peak fitness vs generation
        plt.axvline(x=self.peak_performance["rl"]["generation"], color='b', linestyle='--', alpha=0.7,
                   label=f"RL Peak: Gen {self.peak_performance['rl']['generation']}")
        plt.axvline(x=self.peak_performance["standard"]["generation"], color='r', linestyle='--', alpha=0.7,
                   label=f"Standard Peak: Gen {self.peak_performance['standard']['generation']}")

        # Plot fitness trajectory
        generations = sorted(self.generation_stats.keys())
        if generations:
            rl_mean = [self.generation_stats[g]["rl"]["fitness_mean"] for g in generations]
            std_mean = [self.generation_stats[g]["standard"]["fitness_mean"] for g in generations]

            plt.plot(generations, rl_mean, 'b-', label="RL-guided (mean)")
            plt.plot(generations, std_mean, 'r-', label="Standard (mean)")

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Peak Performance Timing")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(os.path.join(self.metrics_dir, "peak_timing.png"))
        plt.close()

        with open(os.path.join(self.metrics_dir, "peak_timing_report.txt"), "w") as f:
            f.write("Peak Performance Timing:\n")
            f.write(f"  RL-guided: Generation {self.peak_performance['rl']['generation']}, " +
                   f"Fitness: {self.peak_performance['rl']['fitness']:.4f}, " +
                   f"Genome ID: {self.peak_performance['rl']['genome_id']}\n")
            f.write(f"  Standard: Generation {self.peak_performance['standard']['generation']}, " +
                   f"Fitness: {self.peak_performance['standard']['fitness']:.4f}, " +
                   f"Genome ID: {self.peak_performance['standard']['genome_id']}\n\n")

            f.write("Worst Performance Timing:\n")
            f.write(f"  RL-guided: Generation {self.worst_performance['rl']['generation']}, " +
                   f"Fitness: {self.worst_performance['rl']['fitness']:.4f}, " +
                   f"Genome ID: {self.worst_performance['rl']['genome_id']}\n")
            f.write(f"  Standard: Generation {self.worst_performance['standard']['generation']}, " +
                   f"Fitness: {self.worst_performance['standard']['fitness']:.4f}, " +
                   f"Genome ID: {self.worst_performance['standard']['genome_id']}\n")

    def _generate_fitness_trajectory_report(self):
        """Generate a report showing fitness trajectories over generations."""
        generations = sorted(self.generation_stats.keys())
        if not generations:
            return

        # Extract data, replacing None with NaN (which matplotlib can handle)
        rl_mean = [self.generation_stats[g]["rl"]["fitness_mean"] if self.generation_stats[g]["rl"]["fitness_mean"] is not None else np.nan for g in generations]
        rl_min = [self.generation_stats[g]["rl"]["fitness_min"] if self.generation_stats[g]["rl"]["fitness_min"] is not None else np.nan for g in generations]
        rl_max = [self.generation_stats[g]["rl"]["fitness_max"] if self.generation_stats[g]["rl"]["fitness_max"] is not None else np.nan for g in generations]

        std_mean = [self.generation_stats[g]["standard"]["fitness_mean"] if self.generation_stats[g]["standard"]["fitness_mean"] is not None else np.nan for g in generations]
        std_min = [self.generation_stats[g]["standard"]["fitness_min"] if self.generation_stats[g]["standard"]["fitness_min"] is not None else np.nan for g in generations]
        std_max = [self.generation_stats[g]["standard"]["fitness_max"] if self.generation_stats[g]["standard"]["fitness_max"] is not None else np.nan for g in generations]

        # Create arrays and mask NaN values
        rl_mean = np.array(rl_mean, dtype=float)
        rl_min = np.array(rl_min, dtype=float)
        rl_max = np.array(rl_max, dtype=float)
        std_mean = np.array(std_mean, dtype=float)
        std_min = np.array(std_min, dtype=float)
        std_max = np.array(std_max, dtype=float)

        # Convert generations to numpy array for consistency
        generations_array = np.array(generations)

        plt.figure(figsize=(12, 8))

        # Mean fitness
        plt.subplot(2, 1, 1)

        # Plot mean lines (only for non-NaN values)
        plt.plot(generations_array[~np.isnan(rl_mean)], rl_mean[~np.isnan(rl_mean)], 'b-', label="RL-guided (mean)")
        plt.plot(generations_array[~np.isnan(std_mean)], std_mean[~np.isnan(std_mean)], 'r-', label="Standard (mean)")

        # Only fill between where all values exist
        valid_rl = ~np.isnan(rl_min) & ~np.isnan(rl_max)
        valid_std = ~np.isnan(std_min) & ~np.isnan(std_max)

        if np.any(valid_rl):
            plt.fill_between(generations_array[valid_rl], rl_min[valid_rl], rl_max[valid_rl], color='b', alpha=0.2)
        if np.any(valid_std):
            plt.fill_between(generations_array[valid_std], std_min[valid_std], std_max[valid_std], color='r', alpha=0.2)

        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Fitness Trajectory Over Generations")
        plt.legend()
        plt.grid(True)

        # Min-max range (only calculate where both min and max exist)
        plt.subplot(2, 1, 2)

        # Calculate ranges only where both min and max are valid
        rl_range = np.full_like(rl_max, np.nan)
        std_range = np.full_like(std_max, np.nan)

        rl_range[valid_rl] = rl_max[valid_rl] - rl_min[valid_rl]
        std_range[valid_std] = std_max[valid_std] - std_min[valid_std]

        plt.plot(generations_array[~np.isnan(rl_range)], rl_range[~np.isnan(rl_range)], 'b-', label="RL-guided (range)")
        plt.plot(generations_array[~np.isnan(std_range)], std_range[~np.isnan(std_range)], 'r-', label="Standard (range)")

        plt.xlabel("Generation")
        plt.ylabel("Fitness Range")
        plt.title("Fitness Diversity Over Generations")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, "fitness_trajectory.png"))
        plt.close()

    def _generate_action_analysis_report(self):
        """Generate a report analyzing which RL actions produce the best results."""
        if not self.action_effects:
            return

        try:
            # Calculate average improvement for each action
            action_stats = {}
            for action, effects in self.action_effects.items():
                if not effects:  # Skip empty lists
                    continue

                improvements = [float(e["improvement"]) for e in effects if "improvement" in e]
                if not improvements:  # Skip if no improvements
                    continue

                action_stats[action] = {
                    "count": len(improvements),
                    "mean_improvement": float(np.mean(improvements)),
                    "positive_rate": float(np.mean([i > 0 for i in improvements])),
                    "total_improvement": float(sum(improvements))
                }

            if not action_stats:  # If no valid actions, return
                return

            # Sort actions by average improvement
            sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]["mean_improvement"], reverse=True)

            # Plot results
            plt.figure(figsize=(12, 8))

            # Mean improvement by action
            plt.subplot(2, 1, 1)
            actions = [str(a[0]) for a in sorted_actions]  # Convert to string to ensure it's safe
            means = [a[1]["mean_improvement"] for a in sorted_actions]
            counts = [a[1]["count"] for a in sorted_actions]

            # Safety check
            if not actions or not means or not counts:
                return

            # Use count for sizing the bars
            max_count = max(counts)
            if max_count <= 0:
                max_count = 1  # Avoid division by zero
            normalized_counts = [0.3 + 0.7 * (c / max_count) for c in counts]

            bars = plt.bar(range(len(actions)), means, width=normalized_counts)

            # Add count annotations
            for i, bar in enumerate(bars):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"n={counts[i]}", ha='center', va='bottom', fontsize=8)

            plt.xlabel("Action")
            plt.ylabel("Mean Fitness Improvement")
            plt.title("Mean Fitness Improvement by Action")
            plt.xticks(range(len(actions)), actions, rotation=45, ha="right")

            # Positive improvement rate by action
            plt.subplot(2, 1, 2)
            positive_rates = [a[1]["positive_rate"] * 100 for a in sorted_actions]
            plt.bar(range(len(actions)), positive_rates)
            plt.xlabel("Action")
            plt.ylabel("Positive Improvement Rate (%)")
            plt.title("Rate of Positive Improvements by Action")
            plt.xticks(range(len(actions)), actions, rotation=45, ha="right")

            plt.tight_layout()
            plt.savefig(os.path.join(self.metrics_dir, "action_analysis.png"))
            plt.close()

            # Save detailed stats
            with open(os.path.join(self.metrics_dir, "action_analysis.txt"), "w") as f:
                f.write("Action Analysis:\n")
                f.write("=" * 50 + "\n\n")

                for action, stats in sorted_actions:
                    f.write(f"Action: {str(action)}\n")
                    f.write(f"  Count: {stats['count']}\n")
                    f.write(f"  Mean Improvement: {stats['mean_improvement']:.4f}\n")
                    f.write(f"  Positive Rate: {stats['positive_rate']*100:.1f}%\n")
                    f.write(f"  Total Improvement: {stats['total_improvement']:.4f}\n\n")
        except Exception as e:
            print(f"Error generating action analysis: {e}")
            import traceback
            traceback.print_exc()

    def _generate_summary_report(self):
        """Generate an overall summary report."""
        with open(os.path.join(self.metrics_dir, "summary_report.txt"), "w") as f:
            f.write("A/B Testing Experiment Summary\n")
            f.write("=============================\n\n")

            # Overall statistics
            rl_mutations = len(self.mutation_effects["rl"])
            std_mutations = len(self.mutation_effects["standard"])

            # Calculate success rates
            rl_improvements = [e["improvement"] for e in self.mutation_effects["rl"]]
            std_improvements = [e["improvement"] for e in self.mutation_effects["standard"]]

            rl_success_rate = sum(1 for i in rl_improvements if i > 0) / max(1, len(rl_improvements))
            std_success_rate = sum(1 for i in std_improvements if i > 0) / max(1, len(std_improvements))

            f.write("Overall Statistics:\n")
            f.write(f"  Total RL-guided mutations: {rl_mutations}\n")
            f.write(f"  Total Standard mutations: {std_mutations}\n")
            f.write(f"  RL-guided success rate: {rl_success_rate*100:.1f}%\n")
            f.write(f"  Standard success rate: {std_success_rate*100:.1f}%\n\n")

            # Peak fitness
            f.write("Peak Performance:\n")
            f.write(f"  RL-guided peak fitness: {self.peak_performance['rl']['fitness']:.4f} (Gen {self.peak_performance['rl']['generation']})\n")
            f.write(f"  Standard peak fitness: {self.peak_performance['standard']['fitness']:.4f} (Gen {self.peak_performance['standard']['generation']})\n")

            # Whether RL peaked earlier
            if self.peak_performance['rl']['generation'] < self.peak_performance['standard']['generation']:
                f.write("  RL-guided mutations reached peak performance earlier than Standard mutations\n\n")
            else:
                f.write("  Standard mutations reached peak performance earlier than RL-guided mutations\n\n")

            # Cumulative improvements
            if hasattr(self, 'cumulative_improvements'):
                f.write("Cumulative Fitness Improvements:\n")
                f.write(f"  RL-guided total: {self.cumulative_improvements['rl']:.4f}\n")
                f.write(f"  Standard total: {self.cumulative_improvements['standard']:.4f}\n")

                # Compare
                if self.cumulative_improvements['rl'] > self.cumulative_improvements['standard']:
                    diff = self.cumulative_improvements['rl'] - self.cumulative_improvements['standard']
                    ratio = self.cumulative_improvements['rl'] / max(0.0001, self.cumulative_improvements['standard'])
                    f.write(f"  RL-guided produced {diff:.4f} more total improvement ({ratio:.1f}x more)\n\n")
                else:
                    diff = self.cumulative_improvements['standard'] - self.cumulative_improvements['rl']
                    ratio = self.cumulative_improvements['standard'] / max(0.0001, self.cumulative_improvements['rl'])
                    f.write(f"  Standard produced {diff:.4f} more total improvement ({ratio:.1f}x more)\n\n")

            # Best actions (if available)
            if self.action_effects:
                f.write("Top 3 Most Effective RL Actions:\n")

                action_means = {a: np.mean([e["improvement"] for e in effects])
                              for a, effects in self.action_effects.items()}
                top_actions = sorted(action_means.items(), key=lambda x: x[1], reverse=True)[:3]

                for i, (action, mean_imp) in enumerate(top_actions):
                    f.write(f"  {i+1}. Action {action}: Mean improvement {mean_imp:.4f}\n")
