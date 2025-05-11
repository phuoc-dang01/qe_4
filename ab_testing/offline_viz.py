import argparse
import json
import os

from ab_reporter import ABTestingReporter


def generate_from_stats(stats_json: str, out_dir: str, report_interval: int = 1):
    # 1) Load the persisted stats
    with open(stats_json, 'r') as f:
        stats = json.load(f)

    # 2) Create a dummy reproduction carrying only stats
    class DummyReproduction:
        def __init__(self, stats):
            self.stats = stats

    # 3) Instantiate the reporter and force a full report pass
    reporter = ABTestingReporter(DummyReproduction(stats), out_dir, report_interval)
    reporter._generate_report('offline')  # single, “offline” report
    print(f"Offline A/B reports generated under: {os.path.join(out_dir, 'ab_reports', 'generation_offline')}")

_json_stat="/home/pd468/qe/rl_mutation/saved_data/rl_ab_testing_20250510_2113/ab_reports/stats.json"

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Offline A/B Report Generator")
    p.add_argument("--stats_json", default=_json_stat,
                   help="Path to ab_reports/stats.json (output of live run)")
    p.add_argument("--out_dir", default="/home/pd468/qe/rl_mutation/ab_testing/offine_viz/", help="Where to write offline reports")
    p.add_argument("--report_interval", type=int, default=1,
                   help="Not used in offline mode, but kept for compatibility")
    args = p.parse_args()

    generate_from_stats(args.stats_json, args.out_dir, args.report_interval)
