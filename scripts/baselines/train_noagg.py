# 60 / 20 / 20（train / val / test）,独立随机划分 × 3
import numpy as np
import json
import argparse
import os
import time
import torch
import pandas as pd
import sys
from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from venuspheno.models.classifier import GenomeClassifierNoAgg
from venuspheno.utils.metrics import compute_metrics
from venuspheno.utils.dataset import GenomeEmbeddingDataset
from venuspheno.utils.logger import get_logger, log_model_info, log_training_time
from venuspheno.utils.early_stopping import EarlyStopping

# =====================================================
# Args
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--save_root", type=str, default="outputs")
args = parser.parse_args()


# =====================================================
# Config
# =====================================================
cfg = {
    "metadata":  "/home/jinling/MIL/Multitask/data/phenotype_long_bfbi.csv",
    "train_txt": "/home/jinling/MIL/Multitask/data/train_bi.txt",
    "test_txt":  "/home/jinling/MIL/Multitask/data/test_bi.txt",
    "save_dir": f"{args.save_root}/bacformer_result0313",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "input_dim": 480,
    "hidden_dim": 512,
    "dropout": 0.3,

    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "lr": args.lr,
    "seed": 42,
    "n_repeat": 3
}

os.makedirs(cfg["save_dir"], exist_ok=True)

logger = get_logger(cfg["save_dir"], name="train_bacformer_mlp")
logger.info("Training started")


# =====================================================
# Utils
# =====================================================
def split_by_txt(meta_csv, train_txt, test_txt, seed):
    
    df = pd.read_csv(meta_csv)

    # ---------- read ids ----------
    with open(train_txt) as f:
        train_ids = set(x.strip() for x in f)

    with open(test_txt) as f:
        test_ids = set(x.strip() for x in f)

    # ---------- filter ----------
    df_trainval = df[df["genome_id"].isin(train_ids)].reset_index(drop=True)
    df_test = df[df["genome_id"].isin(test_ids)].reset_index(drop=True)

    # ---------- split train / val ----------
    stratify = None
    if df_trainval["label"].nunique() > 1:
        stratify = df_trainval["label"]

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=0.2,
        random_state=seed,
        stratify=stratify
    )

    return (
        df_train.reset_index(drop=True),
        df_val.reset_index(drop=True),
        df_test.reset_index(drop=True)
    )


# =====================================================
# Main
# =====================================================
all_test_metrics = []
start_time = time.time()

for run in range(cfg["n_repeat"]):

    logger.info(f"\n========== Run {run+1}/{cfg['n_repeat']} ==========")

    # ---------- split ----------
    df_train, df_val, df_test = split_by_txt(
        cfg["metadata"],
        cfg["train_txt"],
        cfg["test_txt"],
        seed=cfg["seed"] + run
    )

    train_dataset = GenomeEmbeddingDataset(df_train)
    val_dataset   = GenomeEmbeddingDataset(df_val)
    test_dataset  = GenomeEmbeddingDataset(df_test)

    num_classes = df_train["label"].nunique()
    is_binary = (num_classes == 2)

    # ---------- sampler ----------
    if is_binary:
        train_labels = df_train["label"].values
        pos = (train_labels == 1).sum()
        neg = (train_labels == 0).sum()

        if pos == 0 or neg == 0:
            sampler = None
            shuffle = True
        else:
            sample_weights = np.where(train_labels == 1, 1 / pos, 1 / neg)
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg["batch_size"],
                              sampler=sampler,
                              shuffle=shuffle)

    val_loader = DataLoader(val_dataset,
                            batch_size=cfg["batch_size"],
                            shuffle=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg["batch_size"],
                             shuffle=False)

    # ---------- model ----------
    model_num_classes = 1 if is_binary else num_classes

    model = GenomeClassifierNoAgg(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        num_classes=model_num_classes
    ).to(cfg["device"])

    log_model_info(logger, model, device=cfg["device"])

    # ---------- loss ----------
    if is_binary:
        pos_weight = neg / pos
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=cfg["device"])
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # ---------- training ----------
    best_val_bacc = -1.0
    best_state = None
    early_stopper = EarlyStopping(patience=5)

    for epoch in range(1, cfg["epochs"] + 1):

        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(cfg["device"])
            y = y.to(cfg["device"])

            logits, _ = model(x)

            if is_binary:
                loss = criterion(logits, y.float())
            else:
                loss = criterion(logits, y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch} | Train loss = {total_loss / len(train_loader):.4f}")

        # ---------- validation ----------
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(cfg["device"])
                logits, _ = model(x)
                if is_binary:
                    y_pred.extend(logits.view(-1).cpu().tolist())
                else:
                    y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())

                y_true.extend(y.tolist())

        metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
        val_bacc = metrics["BAcc"]

        logger.info(
            " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        )

        if val_bacc > best_val_bacc:
            best_val_bacc = val_bacc
            best_state = model.state_dict()

        if early_stopper.step(val_bacc):
            logger.info("Early stopping triggered")
            break

    # ---------- test ----------
    model.load_state_dict(best_state)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(cfg["device"])

            logits, _ = model(x)

            if is_binary:
                y_pred.extend(logits.view(-1).cpu().tolist())
            else:
                y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())

            y_true.extend(y.tolist())

    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    all_test_metrics.append(metrics)

    logger.info(
        f"Run {run+1} | " +
        " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    )


# =====================================================
# Final report
# =====================================================
summary = {}

metric_keys = all_test_metrics[0].keys()

for k in metric_keys:
    vals = [m[k] for m in all_test_metrics]
    summary[k] = {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "all_runs": vals
    }

with open(os.path.join(cfg["save_dir"], "final_metrics_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)

logger.info("======================================")
for k, v in summary.items():
    logger.info(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")
logger.info("======================================")

end_time = time.time()
log_training_time(logger, start_time, end_time)
logger.info("Training completed successfully.")
