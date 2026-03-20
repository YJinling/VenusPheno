import numpy as np
import json
import argparse
import os, time, torch
import pandas as pd
import sys
from pathlib import Path

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from venuspheno.models.classifier import GenomeClassifier
from venuspheno.utils.metrics import compute_metrics
from venuspheno.utils.dataset import GenomeBagDataset, collate_fn
from venuspheno.utils.logger import get_logger, log_model_info, log_training_time
from venuspheno.utils.early_stopping import EarlyStopping


# =====================================================
# Args
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="mean",
                    choices=["mean", "max", "attention", "pma", "isab_mean", "isab_pma"])
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--save_root", type=str, default="outputs")
parser.add_argument("--n_repeat", type=int, default=3)
parser.add_argument("--metadata", type=str, default="/home/jinling/MIL/embedding/bacformer_h5_by_g_labels/carbon/meta_carb_maltose.csv")
parser.add_argument("--train_txt", type=str, default="/home/jinling/MIL/embedding/bacformer_h5_by_g_labels/carbon/train.txt")
parser.add_argument("--test_txt", type=str, default="/home/jinling/MIL/embedding/bacformer_h5_by_g_labels/carbon/test.txt")
args = parser.parse_args()


# =====================================================
# Config 
# =====================================================
cfg = {
    "metadata": args.metadata,
    "train_txt": args.train_txt,
    "test_txt": args.test_txt,
    "save_dir": f"{args.save_root}/{args.mode}",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # model
    "input_dim": 1024,
    "agg_mode": args.mode,
    # "num_heads": 4,
    # "num_inds": 64,
    # "num_seeds": 2,
    # "hidden_dim": 512,
    # "dropout": 0.3,
    "num_heads": 2,
    "num_inds": 32,
    "num_seeds": 1,
    "hidden_dim": 256,
    "dropout": 0.5,

    # training
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "lr": args.lr,
    "seed": 42,
    "n_repeat":3,
    # "n_repeat": 5,
}

os.makedirs(cfg["save_dir"], exist_ok=True)
logger = get_logger(cfg["save_dir"], name=f"train_{cfg['agg_mode']}")
logger.info("Training started")


# =====================================================
# Utils
# =====================================================
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def find_best_threshold(y_true, y_prob, metric="f1"):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_score = -1

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == "bacc":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError("metric must be f1/mcc/bacc")

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score

def split_by_txt(meta_csv, train_txt, test_txt, val_ratio=0.2, seed=42):
    df = pd.read_csv(meta_csv, dtype={"gcf": str})
    df["gcf"] = df["gcf"].str.strip()

    with open(train_txt) as f:
        train_ids = set(x.strip() for x in f)

    with open(test_txt) as f:
        test_ids = set(x.strip() for x in f)

    print("Meta size:", len(df))
    print("Train IDs:", len(train_ids))
    print("Intersection (train):", len(set(df["gcf"]) & train_ids))

    df_trainval = df[df["gcf"].isin(train_ids)].reset_index(drop=True)
    df_test = df[df["gcf"].isin(test_ids)].reset_index(drop=True)

    if len(df_trainval) == 0:
        raise RuntimeError("No overlap between meta and train.txt IDs")

    stratify = None
    if df_trainval["label"].nunique() > 1:
        stratify = df_trainval["label"]

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify
    )

    return df_train, df_val, df_test


# =====================================================
# Main
# =====================================================
all_test_metrics = []   # 每个 run 一个 dict
start_time = time.time()

for run in range(cfg["n_repeat"]):
    logger.info(f"\n========== Run {run+1}/{cfg['n_repeat']} ==========")

    # ---------- split ----------
    df_train, df_val, df_test = split_by_txt(
        cfg["metadata"],
        train_txt=cfg["train_txt"],
        test_txt=cfg["test_txt"],
        seed=cfg["seed"] + run
    )

    train_dataset = GenomeBagDataset(df_train)
    val_dataset   = GenomeBagDataset(df_val)
    test_dataset  = GenomeBagDataset(df_test)

    num_classes = df_train["label"].nunique()
    is_binary = True
    logger.info(f"Detected task type: {'binary' if is_binary else f'{num_classes}-class'}")

    # ---------- sampler (binary only) ----------

    if is_binary:
        train_labels = train_dataset.df["label"].values
        pos = (train_labels == 1).sum()
        neg = (train_labels == 0).sum()

        logger.info(f"Train split: +{pos} / -{neg}")

        sample_weights = np.where(train_labels == 1, 1 / pos, 1 / neg)
        # sampler = WeightedRandomSampler(
        #             weights=sample_weights,
        #             num_samples=len(sample_weights),
        #             replacement=True
        #         )
        # shuffle = False
        sampler = None
        shuffle = True


    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    # ---------- model ----------
    model_num_classes = 1 if is_binary else num_classes
    model = GenomeClassifier(
        input_dim=cfg["input_dim"],
        agg_mode=cfg["agg_mode"],
        num_heads=cfg["num_heads"],
        num_inds=cfg["num_inds"],
        num_seeds=cfg["num_seeds"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
        num_classes=model_num_classes
    ).to(cfg["device"])
    log_model_info(logger, model, device=cfg["device"])

    # ---------- loss ----------
    if is_binary:
        if pos == 0 or neg == 0:
            logger.warning("Train split has only one class; skip this run.")
            continue
        pos_weight = neg / pos
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=cfg["device"])
        )
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # ---------- training ----------
    best_val_auprc = -1.0
    best_state = None
    early_stopper = EarlyStopping(patience=10)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for x, mask, y, _ in train_loader:
            x, mask, y = x.to(cfg["device"]), mask.to(cfg["device"]), y.to(cfg["device"])
            logits, _ = model(x, mask)

            if is_binary:
                loss = criterion(logits, y.float())
            else:
                loss = criterion(logits, y.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch} | Train loss={total_loss / len(train_loader):.4f}")

        # ---------- validation ----------
        model.eval()
        y_true, y_logits = [], []

        with torch.no_grad():
            for x, mask, y, _ in val_loader:
                x, mask = x.to(cfg["device"]), mask.to(cfg["device"])
                logits, _ = model(x, mask)

                y_logits.extend(logits.view(-1).cpu().tolist())
                y_true.extend(y.tolist())

        y_true = np.array(y_true)
        y_prob = sigmoid(np.array(y_logits))

        val_auprc = average_precision_score(y_true, y_prob)
        val_auroc = roc_auc_score(y_true, y_prob)

        logger.info(f"Epoch {epoch} | Val AUROC={val_auroc:.4f} | Val AUPRC={val_auprc:.4f}")

        # early stop based on AUPRC
        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_state = model.state_dict()

        if early_stopper.step(val_auprc):
            logger.info("Early stopping triggered")
            break

    # ---------- find best threshold on validation ----------
    model.load_state_dict(best_state)
    model.eval()

    val_true, val_logits = [], []
    with torch.no_grad():
        for x, mask, y, _ in val_loader:
            x, mask = x.to(cfg["device"]), mask.to(cfg["device"])
            logits, _ = model(x, mask)

            val_logits.extend(logits.view(-1).cpu().tolist())
            val_true.extend(y.tolist())

    val_true = np.array(val_true)
    val_prob = sigmoid(np.array(val_logits))

    best_threshold, best_score = find_best_threshold(
        val_true,
        val_prob,
        metric="mcc"   # 推荐用 mcc，比 f1 稳
    )
    logger.info(f"Best threshold on val: {best_threshold:.4f} | MCC={best_score:.4f}")

    # ---------- test ----------
    model.eval()
    y_true, y_logits = [], []

    with torch.no_grad():
        for x, mask, y, _ in test_loader:
            x, mask = x.to(cfg["device"]), mask.to(cfg["device"])
            logits, _ = model(x, mask)

            y_logits.extend(logits.view(-1).cpu().tolist())
            y_true.extend(y.tolist())

    y_true = np.array(y_true)
    y_prob = sigmoid(np.array(y_logits))
    y_pred = (y_prob >= best_threshold).astype(int)

    test_metrics = {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "BAcc": balanced_accuracy_score(y_true, y_pred),
    }

    all_test_metrics.append(test_metrics)

    logger.info(
        f"Run {run+1} | " +
        " | ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
    )

    # 保存 test 预测
    np.save(os.path.join(cfg["save_dir"], f"run{run+1}_test_y_true.npy"), np.array(y_true))
    np.save(os.path.join(cfg["save_dir"], f"run{run+1}_test_y_pred.npy"), np.array(y_pred))

# =====================================================
# Final report (all metrics)
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

out_json = os.path.join(cfg["save_dir"], "final_metrics_summary.json")
with open(out_json, "w") as f:
    json.dump(summary, f, indent=4)

logger.info("======================================")
for k, v in summary.items():
    logger.info(f"{k}: {v['mean']:.4f} ± {v['std']:.4f}")
logger.info("======================================")

end_time = time.time()
log_training_time(logger, start_time, end_time)
logger.info("Training completed successfully.")
