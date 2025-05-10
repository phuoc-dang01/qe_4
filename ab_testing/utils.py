# ab_testing/utils.py
import os

import numpy as np
import torch


def get_observation_from_genome(genome, activation_functions):
    """Extract features from the genome to feed into RL policy."""
    features = []

    # 1. Network structure features
    num_nodes = len(genome.nodes)
    num_connections = len(genome.connections)
    features.extend([num_nodes, num_connections])

    # 2. Connectivity features
    if num_connections > 0:
        enabled_connections = sum(
            1 for conn in genome.connections.values() if conn.enabled
        )
        avg_weight = np.mean([conn.weight for conn in genome.connections.values()])
        features.extend([enabled_connections / num_connections, avg_weight])
    else:
        features.extend([0, 0])

    # 3. Node activation distribution (optional)
    activation_counts = {af: 0 for af in activation_functions}
    for node in genome.nodes.values():
        if hasattr(node, "activation") and node.activation in activation_counts:
            activation_counts[node.activation] += 1

    # 4. Recent fitness history (placeholder)
    features.extend([0, 0, 0, 0])  # In a real implementation, track fitness history

    return np.array(features, dtype=np.float32)

def save_ab_testing_results(save_path, experiment_metadata, stats, best_genome_info):
    """Save A/B testing results to file."""
    os.makedirs(save_path, exist_ok=True)

    # Save metadata
    with open(os.path.join(save_path, "metadata.txt"), "w") as f:
        for key, value in experiment_metadata.items():
            f.write(f"{key}: {value}\n")

    # Save statistics
    # [Additional code to save statistics to files]
