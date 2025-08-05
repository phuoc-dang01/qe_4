#!/usr/bin/env python3
import sys

from rl_mutation.ab_testing.offline_viz import generate_from_stats

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: generate_ab_reports.py STATS_JSON OUT_DIR")
        sys.exit(1)

    stats_json = sys.argv[1]
    out_dir    = sys.argv[2]
    generate_from_stats(stats_json, out_dir)
