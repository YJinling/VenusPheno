"""Score one grouped-MTL best checkpoint at fixed and validation-calibrated thresholds.

This is an evaluation-only post-step.  It never updates the model and never
uses test labels to select a threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from torch.utils.data import DataLoader

from models.classifier import GenomeMultiClassifier2
from utils.dataset import GenomeBagMultiTaskDataset, collate_fn_multitask


METRICS = ("AUROC", "AUPRC", "ACC", "F1", "MCC", "BAcc")


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def best_bacc_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Exact validation sweep; ties are closest to 0.5, then lower."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positives, negatives = int(y_true.sum()), int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Validation threshold selection requires both classes")
    order = np.argsort(probabilities)[::-1]
    values, labels = probabilities[order], y_true[order]
    candidates, cursor, tp, fp = [(float(np.nextafter(values[0], np.inf)), 0, 0)], 0, 0, 0
    while cursor < len(values):
        threshold, end = values[cursor], cursor
        while end < len(values) and values[end] == threshold:
            tp += int(labels[end] == 1); fp += int(labels[end] == 0); end += 1
        candidates.append((float(threshold), tp, fp)); cursor = end
    if not np.any(values == 0.5):
        prediction = probabilities >= 0.5
        candidates.append((0.5, int(y_true[prediction].sum()), int(prediction.sum() - y_true[prediction].sum())))
    scored = [(threshold, .5 * (tp / positives + (negatives - fp) / negatives))
              for threshold, tp, fp in candidates]
    best = max(score for _, score in scored)
    threshold = min((value for value, score in scored if abs(score - best) <= 1e-12),
                    key=lambda value: (abs(value - .5), value))
    return threshold, best


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = probabilities >= threshold
    return {
        "AUROC": float(roc_auc_score(y_true, probabilities)),
        "AUPRC": float(average_precision_score(y_true, probabilities)),
        "ACC": float(accuracy_score(y_true, prediction)),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, prediction)),
        "BAcc": float(balanced_accuracy_score(y_true, prediction)),
    }


def fixed_task_table_validation(task_table: str, tasks: list[str]) -> pd.DataFrame:
    """Read the same fixed genus-split TSV used by training, without importing training-only modules."""
    table = pd.read_csv(task_table, sep="\t", dtype=str).fillna("")
    required = {"task", "metadata", "train_txt", "val_txt", "test_txt"}
    if not required.issubset(table.columns):
        raise ValueError(f"Task table must contain {sorted(required)}")
    if set(table["task"]) != set(tasks) or table["task"].duplicated().any():
        raise ValueError("Task table must contain exactly the evaluated tasks")
    rows, partition = [], {}
    for row in table.set_index("task").loc[tasks].reset_index().itertuples(index=False):
        metadata = pd.read_csv(row.metadata, dtype={"gcf": str})
        if not {"gcf", "h5_path", "label"}.issubset(metadata.columns):
            raise ValueError(f"{row.metadata} must contain gcf, h5_path, label")
        metadata = metadata.rename(columns={"gcf": "genome_id"})
        metadata["genome_id"] = metadata["genome_id"].astype(str).str.strip()
        with open(row.val_txt) as handle:
            validation_ids = {line.strip() for line in handle if line.strip()}
        missing = validation_ids - set(metadata["genome_id"])
        if missing:
            raise ValueError(f"{row.task}/val: {len(missing)} split IDs absent from metadata")
        for genome_id in validation_ids:
            previous = partition.setdefault(genome_id, "val")
            if previous != "val":
                raise ValueError(f"Cross-task split mismatch for validation genome {genome_id}")
        frame = metadata[metadata["genome_id"].isin(validation_ids)].copy()
        if frame.empty:
            raise ValueError(f"{row.task}/val is empty")
        frame["task"] = row.task
        rows.append(frame[["genome_id", "h5_path", "task", "label"]])
    return pd.concat(rows, ignore_index=True)


def validation_predictions(model, loader, tasks: list[str], device: torch.device):
    values = {task: {"y": [], "p": []} for task in tasks}
    rows = []
    model.eval()
    with torch.no_grad():
        for x, instance_mask, y, task_mask, genome_ids in loader:
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _ = model(x.to(device), instance_mask.to(device))
            for index, task in enumerate(tasks):
                valid = task_mask[:, index].bool().numpy()
                labels = y[:, index].numpy()[valid].astype(int)
                probabilities = torch.sigmoid(logits[task]).detach().cpu().numpy().reshape(-1)[valid]
                values[task]["y"].extend(labels.tolist()); values[task]["p"].extend(probabilities.tolist())
                for local_index, label, probability in zip(np.flatnonzero(valid), labels, probabilities):
                    rows.append({"genome_id": str(genome_ids[int(local_index)]), "task": task,
                                 "y_true": int(label), "y_prob": float(probability),
                                 "y_pred_0p5": int(probability >= .5)})
    return {task: (np.asarray(value["y"], int), np.asarray(value["p"], float)) for task, value in values.items()}, rows


def saved_test_predictions(path: Path, tasks: list[str]):
    """Use the exact test probabilities written during the training evaluation."""
    values = {task: {"y": [], "p": []} for task in tasks}
    with path.open() as handle:
        if handle.readline().strip() != "genome_id,task,y_true,y_prob,y_pred_0p5":
            raise RuntimeError(f"Unexpected test-prediction header in {path}")
        for line_number, line in enumerate(handle, start=2):
            fields = line.rstrip("\r\n").rsplit(",", 4)
            if len(fields) != 5:
                raise RuntimeError(f"Malformed test row {line_number} in {path}")
            _, task, label, probability, _ = fields
            if task not in values:
                raise RuntimeError(f"Unexpected task {task!r} in {path}")
            values[task]["y"].append(int(label)); values[task]["p"].append(float(probability))
    return {task: (np.asarray(value["y"], int), np.asarray(value["p"], float)) for task, value in values.items()}


def main():
    args = arguments()
    root = Path(args.run_root)
    config = json.loads((root / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    tasks = [task for task in config["tasks"].split(",") if task]
    if config["model_type"] != "hps_equivalent" or config["strategy"] != "baseline":
        raise RuntimeError("This evaluator is locked to the requested HPS-equivalent baseline grouped MTL")
    task_table = root / "task_table.tsv"
    if not task_table.is_file():
        raise FileNotFoundError(task_table)
    validation = fixed_task_table_validation(str(task_table), tasks)
    validation_loader = DataLoader(GenomeBagMultiTaskDataset(validation, tasks), batch_size=args.batch_size,
                                   shuffle=False, collate_fn=collate_fn_multitask, num_workers=args.num_workers,
                                   pin_memory=True, persistent_workers=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(root / "run1_best_model.pt", map_location="cpu")
    model = GenomeMultiClassifier2(
        input_dim=1024, task_list=tasks, task_num_classes={task: 2 for task in tasks},
        agg_mode="isab_pma", num_heads=4, num_inds=64, num_seeds=4,
        hidden_dim=512, dropout=.3, shared_mlp_type="res_mlp",
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    validation, prediction_rows = validation_predictions(model, validation_loader, tasks, device)
    test = saved_test_predictions(root / "test_predictions.csv", tasks)
    threshold_rows, test_rows, discrepancies = [], [], []
    for task in tasks:
        val_y, val_p = validation[task]; test_y, test_p = test[task]
        threshold, val_tuned = best_bacc_threshold(val_y, val_p)
        val_fixed = metrics(val_y, val_p, .5)
        fixed, tuned = metrics(test_y, test_p, .5), metrics(test_y, test_p, threshold)
        threshold_rows.append({"task": task, "threshold_validation_bacc": threshold,
                               "val_fixed05_BAcc": val_fixed["BAcc"], "val_tuned_BAcc": val_tuned,
                               "n_validation": len(val_y), "n_test": len(test_y)})
        for mode, threshold_used, values in (("fixed05", .5, fixed), ("validation_calibrated_bacc", threshold, tuned)):
            test_rows.append({"task": task, "threshold_mode": mode, "threshold": threshold_used, **values})
        saved = summary["test_per_task"][task]
        discrepancies.append({"task": task, **{f"abs_error_fixed05_{metric}": abs(fixed[metric] - saved[metric])
                                                for metric in METRICS}})
        print(f"{task}: validation threshold={threshold:.6f}; test BAcc {fixed['BAcc']:.4f}->{tuned['BAcc']:.4f}", flush=True)
    pd.DataFrame(prediction_rows).to_csv(root / "validation_predictions.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(root / "validation_bacc_thresholds.csv", index=False)
    per_task = pd.DataFrame(test_rows)
    per_task.to_csv(root / "test_per_task_metrics_fixed05_vs_validation_calibrated.csv", index=False)
    per_task.groupby("threshold_mode", sort=False)[list(METRICS)].mean().reset_index().to_csv(
        root / "test_macro_metrics_fixed05_vs_validation_calibrated.csv", index=False)
    pd.DataFrame(discrepancies).to_csv(root / "fixed05_test_metric_reproduction_audit.csv", index=False)
    protocol = {
        "checkpoint": str(root / "run1_best_model.pt"), "best_epoch": summary["best_epoch"],
        "checkpoint_selection": "existing training protocol: validation macro BAcc at fixed threshold 0.5",
        "threshold_selection": "per-task validation BAcc only; test labels never select thresholds",
        "fixed05_predictions": str(root / "test_predictions.csv"),
        "validation_predictions": str(root / "validation_predictions.csv"),
    }
    (root / "validation_threshold_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")


if __name__ == "__main__":
    main()
