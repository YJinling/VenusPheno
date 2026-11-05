"""Aggregate fixed-seed MTL summaries without touching model selection or test labels."""
import argparse
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    rows = []
    for name in sorted(os.listdir(args.root)):
        path = os.path.join(args.root, name, "summary.json")
        if not os.path.isfile(path):
            continue
        with open(path) as handle:
            summary = json.load(handle)
        for task, metrics in summary["test_per_task"].items():
            rows.append({"seed": name, "task": task, **metrics})
    if not rows:
        raise RuntimeError(f"No seed summary.json files found under {args.root}")
    per_run = pd.DataFrame(rows)
    per_run.to_csv(os.path.join(args.root, "test_per_task_metrics_all_seeds.csv"), index=False)
    numeric = per_run.select_dtypes(include="number").columns
    per_task = per_run.groupby("task")[numeric].agg(["mean", "std"])
    per_task.to_csv(os.path.join(args.root, "test_per_task_metrics_mean_std.csv"))
    macro = per_run.groupby("seed")[numeric].mean().agg(["mean", "std"])
    macro.to_csv(os.path.join(args.root, "test_macro_metrics_mean_std.csv"))
    print("aggregated", len(per_run["seed"].unique()), "seeds", "and", len(per_task), "tasks")


if __name__ == "__main__":
    main()
