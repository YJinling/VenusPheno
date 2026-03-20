import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from venuspheno.models.classifier import GenomeMultiClassifierNoAgg2
from venuspheno.utils.dataset import GenomeEmbeddingMultiTaskDataset
from venuspheno.utils.metrics import compute_metrics2
from venuspheno.utils.logger import get_logger, log_model_info, log_training_time
from venuspheno.utils.early_stopping import EarlyStopping


# ==============================
# Args
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--save_root", type=str, default="outputs")
args = parser.parse_args()

task_list = [
    # "metabolism",
    # "cell_shape",
    "motility",
    "sporulation",
    "cell_shape__bacillus",
    "cell_shape__coccus",
    "metabolism__aerobic",
    "metabolism__anaerobic",
    "metabolism__facultative", 
    "carb_citrate",
    "carb_glucose",
    "carb_glycerol",
    "carb_lactose",
    "carb_maltose",
    "carb_sucrose"
]

# ==============================
# Config
# ==============================
cfg = {
    "metadata":  "/home/jinling/MIL/Multitask/data/phenotype_long_bfbi.csv",
    "train_txt": "/home/jinling/MIL/Multitask/data/train_bi.txt",
    "test_txt":  "/home/jinling/MIL/Multitask/data/test_bi.txt",
    "save_dir": f"{args.save_root}/bacformer_emb_multitask_binary",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "hidden_dim": 512,
    "dropout": 0.3,
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "lr": args.lr,
    "seed": 42,
    "n_repeat": 3
}

os.makedirs(cfg["save_dir"], exist_ok=True)
logger = get_logger(cfg["save_dir"], name="train_bacformer_emb_multitask")
logger.info("Training started")


# ==============================
# Utils
# ==============================
def split_by_txt(meta_csv, train_txt, test_txt, val_ratio=0.2, seed=42):
    df = pd.read_csv(meta_csv, dtype={"genome_id": str})
    df["genome_id"] = df["genome_id"].astype(str).str.strip()

    with open(train_txt) as f:
        train_ids = [x.strip() for x in f if x.strip()]
    with open(test_txt) as f:
        test_ids = [x.strip() for x in f if x.strip()]

    train_ids = set(train_ids)
    test_ids = set(test_ids)

    df_trainval = df[df["genome_id"].isin(train_ids)].copy()
    df_test     = df[df["genome_id"].isin(test_ids)].copy()

    if df_trainval["genome_id"].nunique() == 0:
        raise RuntimeError("No overlap between meta and train.txt IDs")

    genome_ids = df_trainval["genome_id"].drop_duplicates().values

    train_gids, val_gids = train_test_split(
        genome_ids,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True
    )

    df_train = df_trainval[df_trainval["genome_id"].isin(train_gids)].reset_index(drop=True)
    df_val   = df_trainval[df_trainval["genome_id"].isin(val_gids)].reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    print("Meta rows:", len(df))
    print("Trainval genomes:", df_trainval["genome_id"].nunique())
    print("Train genomes:", df_train["genome_id"].nunique())
    print("Val genomes:", df_val["genome_id"].nunique())
    print("Test genomes:", df_test["genome_id"].nunique())

    return df_train, df_val, df_test


def masked_multitask_loss(
    logits_dict,
    y,
    task_mask,
    task_list,
    task_is_binary
):
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
    ce_loss = torch.nn.CrossEntropyLoss(reduction="none")

    total = 0.0
    denom = 0.0

    for t_idx, task in enumerate(task_list):
        valid = task_mask[:, t_idx]
        if valid.sum().item() == 0:
            continue

        logits_t = logits_dict[task][valid]
        y_t = y[valid, t_idx]

        if task_is_binary[task]:
            loss_vec = bce_loss(logits_t.view(-1), y_t.float())
        else:
            loss_vec = ce_loss(logits_t, y_t.long())

        total += loss_vec.sum()
        denom += valid.sum().item()

    if denom == 0:
        return None

    return total / denom


def eval_multitask(
    model,
    loader,
    device,
    task_list,
    task_is_binary,
    task_num_classes
):
    model.eval()

    y_true = {t: [] for t in task_list}
    y_pred = {t: [] for t in task_list}

    with torch.no_grad():
        for x, y, task_mask in loader:
            x = x.to(device)
            y = y.to(device)
            task_mask = task_mask.to(device)

            logits_dict, _ = model(x)

            for t_idx, task in enumerate(task_list):
                valid = task_mask[:, t_idx]
                if valid.sum().item() == 0:
                    continue

                logits_t = logits_dict[task][valid]

                if task_is_binary[task]:
                    probs = torch.sigmoid(logits_t.view(-1)).cpu().numpy()
                else:
                    probs = torch.softmax(logits_t, dim=1).cpu().numpy()

                y_true[task].extend(y[valid, t_idx].cpu().numpy().tolist())
                y_pred[task].extend(probs.tolist())

    per_task = {}
    for task in task_list:
        if len(y_true[task]) == 0:
            continue
        per_task[task] = compute_metrics2(
            y_true[task],
            y_pred[task],
            num_classes=task_num_classes[task]
        )

    macro = {}
    if len(per_task) > 0:
        keys = list(next(iter(per_task.values())).keys())
        for k in keys:
            macro[k] = float(np.mean([per_task[t][k] for t in per_task]))
    else:
        macro = {"BAcc": float("nan")}

    return per_task, macro, y_true, y_pred


# ==============================
# Main
# ==============================
all_runs = []
start_time = time.time()

df_all = pd.read_csv(cfg["metadata"])

for run in range(cfg["n_repeat"]):
    logger.info(f"\n========== Run {run+1}/{cfg['n_repeat']} ==========")

    df_train, df_val, df_test = split_by_txt(
        cfg["metadata"],
        train_txt=cfg["train_txt"],
        test_txt=cfg["test_txt"],
        seed=cfg["seed"] + run
    )

    task_num_classes = {}
    task_is_binary = {}
    for task in task_list:
        vals = df_train[df_train["task"] == task]["label"].dropna().unique()
        n_cls = len(vals)
        task_num_classes[task] = n_cls
        task_is_binary[task] = (n_cls == 2)
        logger.info(
            f"{task}: {n_cls} classes "
            f"({'binary' if task_is_binary[task] else 'multi'})"
        )

    train_dataset = GenomeEmbeddingMultiTaskDataset(df_train, task_list)
    val_dataset = GenomeEmbeddingMultiTaskDataset(df_val, task_list)
    test_dataset = GenomeEmbeddingMultiTaskDataset(df_test, task_list)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    sample_x, _, _ = train_dataset[0]
    input_dim = int(sample_x.shape[0])

    model = GenomeMultiClassifierNoAgg2(
        input_dim=input_dim,
        task_list=task_list,
        task_num_classes=task_num_classes,
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"]
    ).to(cfg["device"])

    log_model_info(logger, model, device=cfg["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=1e-4
    )

    best_val_macro_bacc = -1.0
    best_state = None
    early_stopper = EarlyStopping(patience=10)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, y, task_mask in train_loader:
            x = x.to(cfg["device"])
            y = y.to(cfg["device"])
            task_mask = task_mask.to(cfg["device"])

            logits_dict, _ = model(x)

            loss = masked_multitask_loss(
                logits_dict,
                y,
                task_mask,
                task_list,
                task_is_binary
            )
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        logger.info(f"Epoch {epoch} | Train loss={total_loss / max(n_batches, 1):.4f}")

        per_task_val, macro_val, _, _ = eval_multitask(
            model, val_loader, cfg["device"], task_list, task_is_binary, task_num_classes
        )
        val_macro_bacc = macro_val.get("BAcc", float("nan"))

        logger.info(
            f"Epoch {epoch} | Val macro: " +
            " | ".join([f"{k}={v:.4f}" for k, v in macro_val.items()])
        )

        if val_macro_bacc > best_val_macro_bacc:
            best_val_macro_bacc = val_macro_bacc
            best_state = model.state_dict()

        if early_stopper.step(val_macro_bacc):
            logger.info("Early stopping triggered")
            break

    model.load_state_dict(best_state)

    per_task_test, macro_test, y_true, y_pred = eval_multitask(
        model, test_loader, cfg["device"], task_list, task_is_binary, task_num_classes
    )

    for task in per_task_test:
        np.save(os.path.join(cfg["save_dir"], f"run{run+1}_test_{task}_y_true.npy"), np.array(y_true[task]))
        np.save(os.path.join(cfg["save_dir"], f"run{run+1}_test_{task}_y_pred.npy"), np.array(y_pred[task], dtype=object))

    all_runs.append({
        "run": run + 1,
        "macro": macro_test,
        "per_task": per_task_test
    })

    logger.info(
        f"Run {run+1} | Test macro: " +
        " | ".join([f"{k}={v:.4f}" for k, v in macro_test.items()])
    )


# ==============================
# Save summary
# ==============================
out_json = os.path.join(cfg["save_dir"], "bacformer_emb_multitask_summary.json")
with open(out_json, "w") as f:
    json.dump(all_runs, f, indent=2)

logger.info(f"[DONE] saved summary to {out_json}")

end_time = time.time()
log_training_time(logger, start_time, end_time)
logger.info("Training completed successfully.")
