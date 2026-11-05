"""Re-score completed FINAL STL checkpoints using validation-selected BAcc thresholds."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader

from models.classifier import GenomeClassifier
from utils.dataset import GenomeBagDataset, collate_fn


METRICS = ("AUROC", "AUPRC", "ACC", "F1", "MCC", "BAcc")


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_roots", nargs="*", default=[])
    p.add_argument("--output_dir", required=False)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Inference batch size; does not change eval-mode predictions.")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--allow_fixed05_mismatch", action="store_true",
                   help="Write a reproduction audit instead of aborting if legacy summary metrics differ.")
    p.add_argument("--self_test", action="store_true")
    return p.parse_args()


def run_self_test():
    threshold, score = best_bacc_threshold(np.array([0, 0, 1, 1]),
                                           np.array([0.1, 0.4, 0.6, 0.9]))
    assert np.isclose(threshold, 0.5) and np.isclose(score, 1.0)
    print("threshold self-test passed")


def sigmoid(logits):
    return 1.0 / (1.0 + np.exp(-logits))


def best_bacc_threshold(y_true, probabilities):
    """Exact validation sweep; ties use the threshold closest to 0.5."""
    candidates = np.unique(np.r_[probabilities, 0.5, np.nextafter(probabilities.max(), np.inf)])
    scored = [(float(t), balanced_accuracy_score(y_true, (probabilities >= t).astype(int)))
              for t in candidates if 0.0 <= t <= 1.0]
    best_score = max(score for _, score in scored)
    return min((t for t, score in scored if np.isclose(score, best_score)), key=lambda t: (abs(t - 0.5), t)), best_score


def metric_dict(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    return {
        "AUROC": float(roc_auc_score(y_true, probabilities)),
        "AUPRC": float(average_precision_score(y_true, probabilities)),
        "ACC": float(accuracy_score(y_true, pred)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, pred)),
        "BAcc": float(balanced_accuracy_score(y_true, pred)),
    }


def read_ids(path):
    with open(path) as handle:
        return {line.strip() for line in handle if line.strip()}


def fixed_frame(metadata, split_path):
    frame = pd.read_csv(metadata, dtype={"gcf": str})
    frame["gcf"] = frame.gcf.str.strip()
    ids = read_ids(split_path)
    missing = ids - set(frame.gcf)
    if missing:
        raise RuntimeError(f"{len(missing)} split IDs are absent from {metadata}")
    return frame[frame.gcf.isin(ids)].reset_index(drop=True)


def instantiate(checkpoint):
    cfg = checkpoint["cfg"]
    return GenomeClassifier(
        input_dim=cfg["input_dim"], agg_mode=cfg["agg_mode"], num_heads=cfg["num_heads"],
        num_inds=cfg["num_inds"], num_seeds=cfg["num_seeds"], hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"], num_classes=1, ln=cfg["ln"], head_type=cfg["head_type"],
        input_proj=cfg["input_proj"], seed_mab=cfg["seed_mab"],
        hsp_out_tokens=cfg["hsp_out_tokens"], hsp_rank=cfg["hsp_rank"],
    )


def predict(model, frame, batch_size, device):
    loader = DataLoader(GenomeBagDataset(frame), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    y_all, logits_all = [], []
    model.eval()
    with torch.no_grad():
        for x, mask, y, _ in loader:
            logits, _ = model(x.to(device), mask.to(device))
            y_all.extend(y.tolist())
            logits_all.extend(logits.reshape(-1).tolist())
    return np.asarray(y_all, dtype=int), sigmoid(np.asarray(logits_all))


def final_task_dirs(experiment_roots):
    """Select only finalized task directories; earlier roots have priority."""
    selected = {}
    for root_text in experiment_roots:
        root = Path(root_text)
        for summary_path in sorted(root.glob("*/final_metrics_summary.json")):
            selected.setdefault(summary_path.parent.name, summary_path.parent)
    if not selected:
        raise RuntimeError("No task directory with final_metrics_summary.json found")
    return selected


def main():
    args = arguments()
    if args.self_test:
        run_self_test()
        return
    if not args.experiment_roots or not args.output_dir:
        raise ValueError("--experiment_roots and --output_dir are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    if args.device == "cpu":
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, audit_rows = [], []
    for task, task_dir in sorted(final_task_dirs(args.experiment_roots).items()):
        checkpoint_paths = sorted(task_dir.glob("run[123]_best_model.pt"))
        if len(checkpoint_paths) != 3:
            raise RuntimeError(f"{task}: expected exactly three finalized checkpoints, found {len(checkpoint_paths)}")
        summary = json.loads((task_dir / "final_metrics_summary.json").read_text())
        for checkpoint_path in checkpoint_paths:
            run_name = checkpoint_path.stem.split("_")[0]
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            cfg = checkpoint["cfg"]
            val_frame = fixed_frame(cfg["metadata"], cfg["val_txt"])
            test_frame = fixed_frame(cfg["metadata"], cfg["test_txt"])
            model = instantiate(checkpoint).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            val_y, val_prob = predict(model, val_frame, args.batch_size, device)
            test_y, test_prob = predict(model, test_frame, args.batch_size, device)
            threshold, val_bacc = best_bacc_threshold(val_y, val_prob)
            fixed = metric_dict(test_y, test_prob, 0.5)
            tuned = metric_dict(test_y, test_prob, threshold)
            row = {"task": task, "run": int(run_name.removeprefix("run")), "seed": checkpoint["seed"],
                   "checkpoint": str(checkpoint_path), "threshold": threshold,
                   "val_BAcc_at_0.5": metric_dict(val_y, val_prob, 0.5)["BAcc"],
                   "val_BAcc_tuned": val_bacc}
            row.update({f"test_fixed05_{metric}": fixed[metric] for metric in METRICS})
            row.update({f"test_valtuned_{metric}": tuned[metric] for metric in METRICS})
            for metric in METRICS:
                expected = summary[metric]["all_runs"][row["run"] - 1]
                difference = abs(fixed[metric] - expected)
                audit_rows.append({"task": task, "run": row["run"], "metric": metric,
                                   "saved_summary": expected, "recomputed_fixed05": fixed[metric],
                                   "abs_difference": difference,
                                   "matches_atol_1e-6": bool(np.isclose(fixed[metric], expected, atol=1e-6))})
                if not np.isclose(fixed[metric], expected, atol=1e-6) and not args.allow_fixed05_mismatch:
                    raise RuntimeError(f"{task}/{run_name}: fixed-0.5 {metric} differs from saved final summary")
            rows.append(row)
            print(f"{task} {run_name}: threshold={threshold:.6f}, val_BAcc={val_bacc:.4f}, test_BAcc={fixed['BAcc']:.4f}->{tuned['BAcc']:.4f}")
    if not rows:
        raise RuntimeError("No completed run[1-3]_best_model.pt checkpoints found")
    per_run = pd.DataFrame(rows).sort_values(["task", "run"])
    if per_run.groupby("task").size().ne(3).any():
        raise RuntimeError("Only tasks with exactly three completed checkpoints may be summarized")
    per_run.to_csv(output_dir / "per_run_validation_threshold_comparison.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output_dir / "fixed05_reproduction_audit.csv", index=False)
    summary_rows = []
    for task, group in per_run.groupby("task", sort=True):
        for metric in METRICS:
            fixed = group[f"test_fixed05_{metric}"]
            tuned = group[f"test_valtuned_{metric}"]
            summary_rows.append({"task": task, "metric": metric,
                                 "fixed05_mean": fixed.mean(), "fixed05_std": fixed.std(ddof=0),
                                 "validation_tuned_mean": tuned.mean(), "validation_tuned_std": tuned.std(ddof=0),
                                 "delta_mean": (tuned - fixed).mean(),
                                 "threshold_mean": group.threshold.mean(), "threshold_std": group.threshold.std(ddof=0)})
    pd.DataFrame(summary_rows).to_csv(output_dir / "test_metrics_fixed05_vs_validation_bacc_threshold_mean_std.csv", index=False)
    per_run.groupby("task", sort=True)[["threshold", "val_BAcc_at_0.5", "val_BAcc_tuned"]].agg(["mean", "std"]).to_csv(
        output_dir / "validation_thresholds_mean_std.csv")
    with open(output_dir / "README.txt", "w") as handle:
        handle.write("Thresholds were selected independently for each saved run using only validation BAcc.\n")
        handle.write("Test labels were never used to choose thresholds. AUROC/AUPRC do not depend on threshold.\n")
        handle.write("Only task directories with final_metrics_summary.json were included; earlier roots had priority.\n")
    print(f"Wrote threshold analysis for {per_run.task.nunique()} tasks to {output_dir}")


if __name__ == "__main__":
    main()
