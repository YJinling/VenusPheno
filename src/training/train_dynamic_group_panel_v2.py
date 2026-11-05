"""Dynamic task grouping experiments on the frozen HPS-equivalent backbone.

This file deliberately adds no model parameters.  `baseline` performs the
ordinary task-balanced joint update; `loss_weighting` changes only the loss
weights from gradient-affinity groups; and `sequential` implements the
selective sequential group updates of Jeong & Yoon (ICLR 2025).
"""
import argparse
import copy
import csv
import gc
import json
import os
import random
import shutil
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import AffinityPropagation
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from models.classifier import GenomeMultiClassifier2, GenomeMultiClassifierTaskAdapter
from utils.dataset import GenomeBagMultiTaskDataset, collate_fn_multitask
from utils.earlystopping import EarlyStopping
from utils.logger import get_logger, log_model_info, log_training_time
from utils.metrics import compute_metrics2


TASKS = [
    "motility", "sporulation", "gram_stain", "cell_shape__bacillus",
    "cell_shape__coccus", "metabolism__aerobic", "metabolism__anaerobic",
    "metabolism__facultative", "carb_citrate", "carb_glucose",
    "carb_glycerol", "carb_lactose", "carb_maltose", "carb_sucrose",
]


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=("baseline", "loss_weighting", "sequential"), required=True)
    p.add_argument("--model_type", choices=("hps_equivalent", "corrected_task_adapter"),
                   default="hps_equivalent")
    p.add_argument("--save_root", required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--seed_orth_lambda", type=float, default=0.01)
    p.add_argument("--loss_normalization", choices=("fixed_panel", "active_task_mean"),
                   default="fixed_panel",
                   help="Normalize the joint loss by the panel size or by the number of tasks observed in the batch.")
    p.add_argument("--affinity_ema", type=float, default=0.10)
    p.add_argument("--warmup_examples", type=int, default=2048)
    p.add_argument("--regroup_every_examples", type=int, default=1024)
    p.add_argument("--affinity_every_examples", type=int, default=1024)
    p.add_argument("--affinity_probe_batch_size", type=int, default=0,
                   help="Deterministic prefix size used only for affinity gradients; 0 uses full train batch.")
    p.add_argument("--diagnostic_every_examples", type=int, default=1024)
    p.add_argument("--group_weight_strength", type=float, default=1.0)
    p.add_argument("--ap_damping", type=float, default=0.75)
    p.add_argument("--adapter_monitor_batches", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--task_table", required=True,
                   help="Fixed genus-split TSV: no, task, metadata, train_txt, val_txt, test_txt.")
    p.add_argument("--tasks", default=",".join(TASKS),
                   help="Comma-separated non-empty subset of the frozen 14-task panel.")
    p.add_argument("--check_splits", action="store_true",
                   help="Validate the fixed task table and exit before model construction.")
    return p.parse_args()


def parse_tasks(text):
    tasks = [task.strip() for task in text.split(",") if task.strip()]
    if not tasks or len(tasks) != len(set(tasks)) or any(task not in TASKS for task in tasks):
        raise ValueError("--tasks must be a non-empty, duplicate-free subset of the frozen 14-task panel")
    return tasks


def split_by_txt(meta_csv, train_txt, test_txt, seed):
    df = pd.read_csv(meta_csv, dtype={"genome_id": str})
    df["genome_id"] = df["genome_id"].astype(str).str.strip()
    with open(train_txt) as f:
        train_ids = {line.strip() for line in f if line.strip()}
    with open(test_txt) as f:
        test_ids = {line.strip() for line in f if line.strip()}
    trainval = df[df["genome_id"].isin(train_ids)].copy()
    test = df[df["genome_id"].isin(test_ids)].copy()
    gids = trainval["genome_id"].drop_duplicates().values
    train_gids, val_gids = train_test_split(gids, test_size=0.2, random_state=seed, shuffle=True)
    return (
        trainval[trainval["genome_id"].isin(train_gids)].reset_index(drop=True),
        trainval[trainval["genome_id"].isin(val_gids)].reset_index(drop=True),
        test.reset_index(drop=True),
    )


def fixed_task_table_split(task_table):
    """Build masked MTL frames from the fixed per-task genus partitions."""
    table = pd.read_csv(task_table, sep="\t", dtype=str).fillna("")
    required = {"task", "metadata", "train_txt", "val_txt", "test_txt"}
    if not required.issubset(table.columns):
        raise ValueError(f"Task table must contain {sorted(required)}")
    if not set(TASKS).issubset(set(table["task"])) or table["task"].duplicated().any():
        raise ValueError("Task table must contain each requested task exactly once")
    table = table.set_index("task").loc[TASKS].reset_index()

    split_rows = {name: [] for name in ("train", "val", "test")}
    genome_partition = {}
    for row in table.itertuples(index=False):
        meta = pd.read_csv(row.metadata, dtype={"gcf": str})
        if not {"gcf", "h5_path", "label"}.issubset(meta.columns):
            raise ValueError(f"{row.metadata} must contain gcf, h5_path, label")
        meta = meta.rename(columns={"gcf": "genome_id"})
        meta["genome_id"] = meta["genome_id"].astype(str).str.strip()
        if meta["genome_id"].duplicated().any():
            raise ValueError(f"Duplicate genomes in metadata: {row.metadata}")
        for split in ("train", "val", "test"):
            with open(getattr(row, f"{split}_txt")) as handle:
                ids = {line.strip() for line in handle if line.strip()}
            missing = ids - set(meta["genome_id"])
            if missing:
                raise ValueError(f"{row.task}/{split}: {len(missing)} split IDs absent from metadata")
            for genome_id in ids:
                previous = genome_partition.setdefault(genome_id, split)
                if previous != split:
                    raise ValueError(f"Cross-task split leakage: {genome_id} is {previous} and {split}")
            part = meta[meta["genome_id"].isin(ids)].copy()
            if part.empty:
                raise ValueError(f"{row.task}/{split} is empty")
            part["task"] = row.task
            split_rows[split].append(part[["genome_id", "h5_path", "task", "label"]])
    frames = tuple(pd.concat(split_rows[name], ignore_index=True) for name in ("train", "val", "test"))
    return frames


def task_losses(logits, y, mask, pos_weights):
    losses, counts = {}, {}
    for idx, task in enumerate(TASKS):
        valid = mask[:, idx].bool()
        if valid.any():
            losses[task] = F.binary_cross_entropy_with_logits(
                logits[task][valid].reshape(-1), y[valid, idx].float(),
                pos_weight=pos_weights[task], reduction="mean",
            )
            counts[task] = int(valid.sum().item())
    return losses, counts


def seed_orth_loss(seed_embeddings):
    if seed_embeddings is None or seed_embeddings.ndim != 3 or seed_embeddings.size(1) < 2:
        return torch.zeros((), device=seed_embeddings.device if seed_embeddings is not None else "cuda")
    z = F.normalize(seed_embeddings, dim=-1)
    gram = z @ z.transpose(1, 2)
    eye = torch.eye(gram.size(1), device=gram.device).unsqueeze(0)
    return ((gram * (1 - eye)) ** 2).sum() / (gram.size(0) * gram.size(1) * (gram.size(1) - 1))


def model_forward_losses(model, x, inst_mask, y, task_mask, pos_weights, training):
    if training:
        model.train()
    else:
        model.eval()
    logits, _, seeds = model(x, inst_mask, return_seed_emb=True)
    losses, counts = task_losses(logits, y, task_mask, pos_weights)
    return losses, counts, seeds


def symmetric_affinity(matrix):
    a = np.asarray(matrix, dtype=float).copy()
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = (a + a.T) / 2.0
    # AP's diagonal is its exemplar preference, not self-affinity.  A high
    # diagonal would force 14 singleton groups irrespective of the evidence.
    offdiag = a[~np.eye(len(a), dtype=bool)]
    np.fill_diagonal(a, float(np.median(offdiag)))
    return a


def affinity_groups(affinity, tasks, damping, seed):
    """Affinity Propagation with a safe singleton fallback for no signal."""
    offdiag = affinity.copy()
    np.fill_diagonal(offdiag, 0.0)
    if not np.any(np.abs(offdiag) > 1e-12):
        return [[task] for task in tasks]
    labels = AffinityPropagation(
        affinity="precomputed", damping=damping, random_state=seed,
    ).fit_predict(symmetric_affinity(affinity))
    groups = {}
    for label, task in zip(labels, tasks):
        groups.setdefault(int(label), []).append(task)
    return [groups[k] for k in sorted(groups)]


def reference_affinity_groups(affinity, tasks, damping, previous_groups):
    """Official AP preprocessing, retaining the prior valid partition on failure."""
    a = np.asarray(affinity, dtype=float).copy()
    if not np.isfinite(a).all():
        return previous_groups, "regroup_fallback_nonfinite_affinity"
    diagonal = np.diag(a)
    if np.any(np.abs(diagonal) <= 1e-8):
        return previous_groups, "regroup_fallback_zero_self_affinity"
    a /= diagonal[np.newaxis, :]
    a = (a + a.T) / 2.0
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            cluster = AffinityPropagation(
                affinity="precomputed", damping=damping, convergence_iter=50,
            ).fit(a)
        labels = cluster.labels_
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            return previous_groups, "regroup_fallback_ap_nonconverged"
        if cluster.cluster_centers_indices_ is None or len(cluster.cluster_centers_indices_) == 0:
            return previous_groups, "regroup_fallback_no_centers"
        grouped = {}
        for label, task in zip(labels, tasks):
            grouped.setdefault(int(label), []).append(task)
        candidate = [grouped[key] for key in sorted(grouped)]
        flattened = [task for group in candidate for task in group]
        if len(flattened) != len(tasks) or len(set(flattened)) != len(tasks) or set(flattened) != set(tasks):
            return previous_groups, "regroup_fallback_invalid_partition"
        return candidate, "regroup_reference_ap"
    except (FloatingPointError, ValueError):
        return previous_groups, "regroup_fallback_ap_error"


def groups_text(groups):
    return " | ".join(",".join(group) for group in groups)


def group_weights(groups, affinity, strength):
    """Return per-task weights with mean one and affinity-dependent group mass."""
    scores = []
    index = {task: i for i, task in enumerate(TASKS)}
    for group in groups:
        if len(group) == 1:
            scores.append(0.0)
            continue
        ids = [index[t] for t in group]
        block = affinity[np.ix_(ids, ids)]
        scores.append(float((block.sum() - np.trace(block)) / (len(ids) * (len(ids) - 1))))
    scores = np.asarray(scores)
    masses = np.exp(strength * (scores - scores.max()))
    masses /= masses.sum()
    weights = {}
    for mass, group in zip(masses, groups):
        for task in group:
            weights[task] = float(len(TASKS) * mass / len(group))
    return weights, scores.tolist(), masses.tolist()


def gradient_affinity(model, losses):
    """Deterministic shared-backbone gradient cosine affinity on a train batch."""
    params = [p for name, p in model.named_parameters() if name.startswith(("aggregator", "shared_mlp"))]
    vectors = {}
    items = list(losses.items())
    for item_idx, (task, loss) in enumerate(items):
        # The final VJP releases the probe graph instead of retaining it until
        # the next B=32 training forward.
        grads = torch.autograd.grad(loss, params, retain_graph=(item_idx + 1 < len(items)), allow_unused=True)
        pieces = [torch.zeros_like(p).reshape(-1) if g is None else g.detach().reshape(-1) for p, g in zip(params, grads)]
        vec = torch.cat(pieces)
        if vec.norm() > 1e-12:
            vectors[task] = vec / vec.norm()
    matrix = np.zeros((len(TASKS), len(TASKS)), dtype=float)
    for i, source in enumerate(TASKS):
        for j, target in enumerate(TASKS):
            if source in vectors and target in vectors:
                matrix[i, j] = float((vectors[source] * vectors[target]).sum().cpu())
    return matrix


def affinity_probe_batch(x, inst_mask, y, mask, max_size):
    """Use a deterministic subset only for the no-step affinity diagnostic."""
    if max_size <= 0 or x.size(0) <= max_size:
        return x, inst_mask, y, mask
    return x[:max_size], inst_mask[:max_size], y[:max_size], mask[:max_size]


def proximal_update(ema, before, after, source_group, beta):
    """Reference PIA update, plus raw group-to-task observations for auditing."""
    updated = ema.copy()
    group_set = set(source_group)
    group_size = len(source_group)
    raw_group_to_task = {}
    for target in TASKS:
        if target in before and target in after:
            raw_group_to_task[target] = (
                (before[target] - after[target]) / max(abs(before[target]), 1e-8)
            )
    for source in source_group:
        if source not in before:
            continue
        i = TASKS.index(source)
        for target in TASKS:
            if target not in raw_group_to_task:
                continue
            j = TASKS.index(target)
            observed = raw_group_to_task[target] / group_size
            reverse = raw_group_to_task[source] / group_size
            if target == source:
                if observed < 0:
                    continue
                updated[i, j] = (1 - beta) * updated[i, j] + beta * observed
            elif target not in group_set or observed * reverse >= 0:
                updated[i, j] = (1 - beta) * updated[i, j] + beta * observed
            else:
                updated[i, j] = (1 - beta) * updated[i, j] - beta * max(observed, reverse)
    return updated, raw_group_to_task


def adapter_monitor(model, loader, device, epoch, phase, max_batches):
    """One fixed validation-batch probe per epoch confirms adapter activity cheaply."""
    if not hasattr(model, "adapters"):
        return []
    totals = {task: {"delta_sq": 0.0, "shared_sq": 0.0, "count": 0}
              for task in TASKS}
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, inst_mask, _, _, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            _, _, info = model(x.to(device), inst_mask.to(device), return_adapter_info=True)
            shared = info["shared_embedding"]
            shared_sq = float(shared.square().sum().cpu())
            count = shared.numel()
            for task, contribution in info["adapter_contributions"].items():
                totals[task]["delta_sq"] += float(contribution.square().sum().cpu())
                totals[task]["shared_sq"] += shared_sq
                totals[task]["count"] += count
    records = []
    for task in TASKS:
        adapter_sq = sum(float(param.detach().square().sum().cpu())
                         for param in model.adapters[task].parameters())
        gamma = float(model.adapter_gammas[task].detach().cpu())
        count = max(totals[task]["count"], 1)
        delta_rms = (totals[task]["delta_sq"] / count) ** 0.5
        shared_rms = (totals[task]["shared_sq"] / count) ** 0.5
        records.append({
            "epoch": epoch, "phase": phase, "task": task, "gamma": gamma,
            "adapter_parameter_norm": adapter_sq ** 0.5,
            "adapter_contribution_rms": delta_rms,
            "adapter_to_shared_rms": delta_rms / max(shared_rms, 1e-12),
        })
    return records


def assert_private_gradient_scope(model, active_group):
    """Sequential updates must not create gradients for another task's private branch."""
    if not hasattr(model, "adapters"):
        return
    active = set(active_group)
    for task in model.task_list:
        if task in active:
            continue
        private = list(model.adapters[task].parameters())
        private += list(model.task_heads[task].parameters())
        private += [model.adapter_gammas[task]]
        if any(param.grad is not None and bool(torch.count_nonzero(param.grad)) for param in private):
            raise RuntimeError(f"Inactive private branch received gradient: {task}")


def evaluate(model, loader, device, task_num_classes, return_predictions=False):
    model.eval()
    truth = {task: [] for task in TASKS}
    pred = {task: [] for task in TASKS}
    prediction_rows = []
    with torch.no_grad():
        for x, inst_mask, y, mask, genome_ids in loader:
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits, _ = model(x.to(device), inst_mask.to(device))
            for index, task in enumerate(TASKS):
                valid = mask[:, index].bool()
                if valid.any():
                    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1).tolist()
                    y_values = y[valid, index].numpy()
                    prob_values = torch.sigmoid(logits[task][valid]).cpu().numpy().reshape(-1)
                    truth[task].extend(y_values)
                    pred[task].extend(prob_values)
                    prediction_rows.extend({
                        "genome_id": str(genome_ids[row]), "task": task,
                        "y_true": int(label), "y_prob": float(prob),
                        "y_pred_0p5": int(prob >= 0.5),
                    } for row, label, prob in zip(valid_indices, y_values, prob_values))
    per_task = {task: compute_metrics2(truth[task], pred[task], num_classes=task_num_classes[task])
                for task in TASKS if truth[task]}
    macro = {key: float(np.mean([metrics[key] for metrics in per_task.values()]))
             for key in next(iter(per_task.values()))}
    if return_predictions:
        return per_task, macro, prediction_rows
    return per_task, macro


class Recorder:
    def __init__(self, root):
        self.root = root
        self.events = []
        self.losses = []
        self.raw_pia = []
        self.co_group_sum = np.zeros((len(TASKS), len(TASKS)), dtype=float)
        self.co_group_count = 0

    def groups(self, step, seen, groups, affinity, note):
        membership = np.zeros_like(self.co_group_sum)
        for group in groups:
            ids = [TASKS.index(task) for task in group]
            membership[np.ix_(ids, ids)] = 1.0
        self.co_group_sum += membership
        self.co_group_count += 1
        self.events.append({"event": note, "step": step, "examples_seen": seen,
                            "n_groups": len(groups), "groups": groups_text(groups)})
        pd.DataFrame(affinity, index=TASKS, columns=TASKS).to_csv(
            os.path.join(self.root, f"affinity_ema_step{step:06d}.csv"))

    def proximal(self, step, seen, group, raw_group_to_task):
        for target, raw_value in raw_group_to_task.items():
            self.raw_pia.append({
                "step": step, "examples_seen": seen, "source_group": groups_text([group]),
                "group_size": len(group), "target": target,
                "raw_relative_loss_change": raw_value,
                "per_task_attribution": raw_value / len(group),
            })

    def loss(self, step, seen, losses, weights, groups):
        row = {"step": step, "examples_seen": seen, "groups": groups_text(groups)}
        row.update({f"loss_{task}": float(value) for task, value in losses.items()})
        row.update({f"weight_{task}": float(weights.get(task, 0.0)) for task in TASKS})
        self.losses.append(row)

    def save(self):
        pd.DataFrame(self.events).to_csv(os.path.join(self.root, "group_trajectory.csv"), index=False)
        pd.DataFrame(self.losses).to_csv(os.path.join(self.root, "task_loss_weight_trajectory.csv"), index=False)
        pd.DataFrame(self.raw_pia).to_csv(os.path.join(self.root, "proximal_group_to_task_pia_raw.csv"), index=False)
        co_group = self.co_group_sum / max(self.co_group_count, 1)
        pd.DataFrame(co_group, index=TASKS, columns=TASKS).to_csv(
            os.path.join(self.root, "task_cogrouping_matrix.csv"))


def main():
    args = arguments()
    global TASKS
    TASKS = parse_tasks(args.tasks)
    os.makedirs(args.save_root, exist_ok=True)
    shutil.copy2(__file__, os.path.join(args.save_root, os.path.basename(__file__)))
    shutil.copy2(args.task_table, os.path.join(args.save_root, "task_table.tsv"))
    with open(os.path.join(args.save_root, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    logger = get_logger(args.save_root, name="dynamic_groups")
    logger.info("strategy=%s; no trainable grouping module is present", args.strategy)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train, val, test = fixed_task_table_split(args.task_table)
    logger.info("fixed task-table split: train=%d val=%d test=%d task_table=%s",
                len(train), len(val), len(test), args.task_table)
    if args.check_splits:
        logger.info("fixed split validation passed")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task_num_classes = {task: len(train.loc[train.task == task, "label"].dropna().unique()) for task in TASKS}
    if any(count != 2 for count in task_num_classes.values()):
        raise RuntimeError(f"Expected 14 binary tasks, got {task_num_classes}")
    pos_weights = {}
    for task in TASKS:
        labels = train.loc[train.task == task, "label"].dropna()
        pos, neg = (labels == 1).sum(), (labels == 0).sum()
        if pos == 0 or neg == 0:
            raise RuntimeError(f"{task} has one class in training split")
        pos_weights[task] = torch.tensor(float(neg / pos), device=device)
    loaders = {}
    for name, frame, shuffle in (("train", train, True), ("val", val, False), ("test", test, False)):
        loaders[name] = DataLoader(GenomeBagMultiTaskDataset(frame, TASKS), batch_size=args.batch_size,
            shuffle=shuffle, collate_fn=collate_fn_multitask, num_workers=args.num_workers,
            pin_memory=True, persistent_workers=False)
    model_kwargs = dict(
        input_dim=1024, task_list=TASKS, task_num_classes=task_num_classes,
        agg_mode="isab_pma", num_heads=4, num_inds=64, num_seeds=4,
        hidden_dim=512, dropout=0.3, shared_mlp_type="res_mlp",
    )
    if args.model_type == "corrected_task_adapter":
        model = GenomeMultiClassifierTaskAdapter(
            **model_kwargs, adapter_bottleneck_dim=64, adapter_gamma_init=1.0).to(device)
        logger.info("corrected task adapters: up=0, gamma=1; initial logits match HPS-equivalent model")
    else:
        model = GenomeMultiClassifier2(**model_kwargs).to(device)
    log_model_info(logger, model, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    ema = np.zeros((len(TASKS), len(TASKS)), dtype=float)
    groups = [[task] for task in TASKS]
    weights = {task: 1.0 for task in TASKS}
    recorder = Recorder(args.save_root)
    recorder.groups(0, 0, groups, ema, "initial_singletons")
    best_bacc, best_state, best_epoch = -float("inf"), None, None
    stopper = EarlyStopping(patience=10)
    global_step = examples_seen = examples_since_group = examples_since_affinity = examples_since_diag = 0
    affinity_observations = optimizer_steps = 0
    memory_rows = []
    adapter_records = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_losses, train_batches = 0.0, 0
        for x, inst_mask, y, mask, _ in loaders["train"]:
            x, inst_mask, y, mask = (tensor.to(device) for tensor in (x, inst_mask, y, mask))
            batch_examples = x.size(0)
            examples_seen += batch_examples; examples_since_group += batch_examples
            examples_since_affinity += batch_examples; examples_since_diag += batch_examples
            active_groups = [[task for task in group if mask[:, TASKS.index(task)].any()] for group in groups]
            active_groups = [group for group in active_groups if group]
            if args.strategy == "sequential":
                random.shuffle(active_groups)
                model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                        before_losses, _, _ = model_forward_losses(model, x, inst_mask, y, mask, pos_weights, training=False)
                    before_values = {task: float(value.cpu()) for task, value in before_losses.items()}
                batch_loss = 0.0
                for group in active_groups:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                        current_losses, _, seed_embeddings = model_forward_losses(model, x, inst_mask, y, mask, pos_weights, training=True)
                        # Equal task mass across the entire batch; orthogonality is split across updates.
                        group_loss = sum(current_losses[task] for task in group) / len(TASKS)
                        group_loss = group_loss + args.seed_orth_lambda * seed_orth_loss(seed_embeddings) / len(active_groups)
                    scaler.scale(group_loss).backward()
                    scaler.unscale_(optimizer)
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    if global_step == 0:
                        assert_private_gradient_scope(model, group)
                    scaler.step(optimizer); scaler.update(); optimizer_steps += 1
                    batch_loss += float(group_loss.detach().cpu())
                    model.eval()
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                            after_losses, _, _ = model_forward_losses(model, x, inst_mask, y, mask, pos_weights, training=False)
                    after_values = {task: float(value.cpu()) for task, value in after_losses.items()}
                    ema, raw_group_to_task = proximal_update(
                        ema, before_values, after_values, group, args.affinity_ema)
                    recorder.proximal(global_step + 1, examples_seen, group, raw_group_to_task)
                    affinity_observations += len(group)
                    before_values = after_values
                weights = {task: 1.0 for task in TASKS}
                train_losses += batch_loss
            else:
                optimizer.zero_grad(set_to_none=True)
                if args.strategy == "loss_weighting" and examples_since_affinity >= args.affinity_every_examples:
                    model.eval()
                    probe_x, probe_mask, probe_y, probe_task_mask = affinity_probe_batch(
                        x, inst_mask, y, mask, args.affinity_probe_batch_size)
                    with torch.enable_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
                        measurement_losses, _, _ = model_forward_losses(
                            model, probe_x, probe_mask, probe_y, probe_task_mask, pos_weights, training=False)
                        observed = gradient_affinity(model, measurement_losses)
                        ema = (1 - args.affinity_ema) * ema + args.affinity_ema * observed
                        affinity_observations += 1
                        examples_since_affinity = 0
                    del measurement_losses, observed, probe_x, probe_mask, probe_y, probe_task_mask
                    if device == "cuda":
                        torch.cuda.empty_cache()
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    losses, _, seed_embeddings = model_forward_losses(model, x, inst_mask, y, mask, pos_weights, training=True)
                    denominator = len(losses) if args.loss_normalization == "active_task_mean" else len(TASKS)
                    weighted = sum(weights[task] * loss for task, loss in losses.items()) / denominator
                    total = weighted + args.seed_orth_lambda * seed_orth_loss(seed_embeddings)
                scaler.scale(total).backward(); scaler.unscale_(optimizer)
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer); scaler.update(); optimizer_steps += 1
                train_losses += float(total.detach().cpu())
            global_step += 1; train_batches += 1
            if args.strategy != "baseline" and examples_seen >= args.warmup_examples and examples_since_group >= args.regroup_every_examples:
                if args.strategy == "sequential":
                    groups, regroup_note = reference_affinity_groups(
                        ema, TASKS, args.ap_damping, groups)
                else:
                    groups = affinity_groups(ema, TASKS, args.ap_damping, args.seed + global_step)
                    regroup_note = "regroup_robust_ap"
                if args.strategy == "loss_weighting":
                    weights, cohesion, masses = group_weights(groups, ema, args.group_weight_strength)
                    logger.info("step=%d groups=%s cohesion=%s mass=%s", global_step, groups_text(groups), cohesion, masses)
                recorder.groups(global_step, examples_seen, groups, ema, regroup_note)
                examples_since_group = 0
            if examples_since_diag >= args.diagnostic_every_examples:
                model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                        diagnostic_losses, _, _ = model_forward_losses(model, x, inst_mask, y, mask, pos_weights, training=False)
                recorder.loss(global_step, examples_seen, diagnostic_losses, weights, groups)
                examples_since_diag = 0
        per_val, macro_val = evaluate(model, loaders["val"], device, task_num_classes)
        val_bacc = macro_val["BAcc"]
        logger.info("epoch=%d train_loss=%.6f val=%s groups=%s optimizer_steps=%d",
                    epoch, train_losses / max(train_batches, 1), macro_val, groups_text(groups), optimizer_steps)
        pd.DataFrame.from_dict(per_val, orient="index").to_csv(os.path.join(args.save_root, f"validation_epoch{epoch:03d}.csv"))
        if val_bacc > best_bacc:
            best_bacc, best_epoch, best_state = val_bacc, epoch, copy.deepcopy(model.state_dict())
        adapter_records.extend(adapter_monitor(
            model, loaders["val"], device, epoch, "epoch", args.adapter_monitor_batches))
        if stopper.step(val_bacc):
            logger.info("early stopping at epoch %d", epoch)
            break
        if device == "cuda":
            memory_rows.append({
                "epoch": epoch,
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes_before_empty_cache": int(torch.cuda.memory_reserved()),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            })
            logger.info("epoch=%d CUDA allocated=%.2fGiB reserved=%.2fGiB peak=%.2fGiB; releasing cache",
                        epoch, memory_rows[-1]["allocated_bytes"] / 2**30,
                        memory_rows[-1]["reserved_bytes_before_empty_cache"] / 2**30,
                        memory_rows[-1]["peak_allocated_bytes"] / 2**30)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    model.load_state_dict(best_state)
    adapter_records.extend(adapter_monitor(
        model, loaders["val"], device, best_epoch, "best_checkpoint", args.adapter_monitor_batches))
    checkpoint = os.path.join(args.save_root, "run1_best_model.pt")
    torch.save({"state_dict": best_state, "best_val_macro_bacc": best_bacc, "best_epoch": best_epoch}, checkpoint)
    per_test, macro_test, test_predictions = evaluate(
        model, loaders["test"], device, task_num_classes, return_predictions=True)
    pd.DataFrame.from_dict(per_test, orient="index").to_csv(os.path.join(args.save_root, "test_per_task_metrics.csv"))
    pd.DataFrame(test_predictions).to_csv(os.path.join(args.save_root, "test_predictions.csv"), index=False)
    recorder.save()
    pd.DataFrame(adapter_records).to_csv(os.path.join(args.save_root, "adapter_epoch_monitor.csv"), index=False)
    pd.DataFrame(memory_rows).to_csv(os.path.join(args.save_root, "cuda_memory_by_epoch.csv"), index=False)
    summary = {"strategy": args.strategy, "model_type": args.model_type, "tasks": TASKS, "params": n_params, "best_epoch": best_epoch,
               "best_val_macro_bacc": best_bacc, "test_macro": macro_test, "test_per_task": per_test,
               "optimizer_steps": optimizer_steps, "affinity_observations": affinity_observations,
               "examples_seen": examples_seen, "runtime_seconds": time.time() - started,
               "peak_cuda_allocated_bytes": max((row["peak_allocated_bytes"] for row in memory_rows), default=None),
               "checkpoint": checkpoint, "final_groups": groups,
               "adapter_monitor": (os.path.join(args.save_root, "adapter_epoch_monitor.csv")
                                   if args.model_type == "corrected_task_adapter" else None)}
    with open(os.path.join(args.save_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("DONE %s", json.dumps(summary, default=str))
    log_training_time(logger, started, time.time())


if __name__ == "__main__":
    main()
