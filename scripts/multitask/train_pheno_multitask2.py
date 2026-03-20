import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import copy
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from venuspheno.models.classifier import GenomeMultiClassifier2
from venuspheno.utils.dataset import GenomeBagMultiTaskDataset, collate_fn_multitask
from venuspheno.utils.metrics import compute_metrics2
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
parser.add_argument("--num_seed", type=int, default=1)
args = parser.parse_args()

task_list = [
    # "motility",
    # "sporulation",
    # "cell_shape__bacillus",
    # # "cell_shape__coccus",
    # "metabolism__aerobic",
    # "metabolism__anaerobic",
    # "metabolism__facultative", 
    "carb_citrate",
    "carb_glucose",
    "carb_glycerol",
    "carb_lactose",
    "carb_maltose",
    "carb_sucrose"
]

# =====================================================
# Config
# =====================================================
cfg = {
    "metadata":  "/home/jinling/MIL/Multitask/data/phenotype_long_bi.csv",
    "train_txt": "/home/jinling/MIL/Multitask/data/train_bi.txt",
    "test_txt":  "/home/jinling/MIL/Multitask/data/test_bi.txt",
    "save_dir": f"{args.save_root}/{args.mode}",
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    "input_dim": 1024,
    "agg_mode": args.mode,
    "num_heads": 4,
    "num_inds": 64,
    "num_seeds": args.num_seed,
    "hidden_dim": 512,
    "dropout": 0.3,

    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "lr": args.lr,
    "seed": 42,
    "n_repeat": 3,
}

os.makedirs(cfg["save_dir"], exist_ok=True)
logger = get_logger(cfg["save_dir"], name=f"train_multitask_{cfg['agg_mode']}")
logger.info("Training started")


# =====================================================
# Utils
# =====================================================
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

def masked_multitask_loss(logits_dict, y, task_mask, task_list, task_pw_tensors):
    total, denom = 0.0, 0.0
    for t_idx, task in enumerate(task_list):
        valid = task_mask[:, t_idx]
        if valid.sum().item() == 0:
            continue
        logits_t = logits_dict[task][valid]
        y_t      = y[valid, t_idx]
        loss_vec = torch.nn.BCEWithLogitsLoss(
            pos_weight=task_pw_tensors[task], reduction="none"  # 直接用预计算的
        )(logits_t.view(-1), y_t.float())
        total += loss_vec.sum()
        denom += valid.sum().item()
    return (total / denom) if denom > 0 else None


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
        for x, inst_mask, y, task_mask, _ in loader:
            x = x.to(device)
            inst_mask = inst_mask.to(device)
            y = y.to(device)
            task_mask = task_mask.to(device)

            logits_dict, _ = model(x, inst_mask)

            for t_idx, task in enumerate(task_list):
                valid = task_mask[:, t_idx]
                if valid.sum().item() == 0:
                    continue

                logits_t = logits_dict[task][valid]

                if task_is_binary[task]:
                    probs = torch.sigmoid(logits_t.view(-1)).cpu().numpy()
                else:
                    probs = torch.softmax(logits_t, dim=1).cpu().numpy()

                y_true[task].extend(y[valid, t_idx].cpu().numpy())
                y_pred[task].extend(probs)

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


# =====================================================
# Main
# =====================================================
all_run_summary = []
start_time = time.time()

for run in range(cfg["n_repeat"]):
    logger.info(f"\n========== Run {run+1}/{cfg['n_repeat']} ==========")

    df_train, df_val, df_test = split_by_txt(
        cfg["metadata"],
        train_txt=cfg["train_txt"],
        test_txt=cfg["test_txt"],
        seed=cfg["seed"] + run
    )
    df_train = df_train[df_train["task"].isin(task_list)].reset_index(drop=True)
    df_val   = df_val[df_val["task"].isin(task_list)].reset_index(drop=True)
    df_test  = df_test[df_test["task"].isin(task_list)].reset_index(drop=True)


    # 自动检测每个 task 的类别数
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

    task_pos_weights = {}
    for task in task_list:
        subset = df_train[df_train["task"] == task]["label"].dropna()
        pos = (subset == 1).sum()
        neg = (subset == 0).sum()
        pw = float(neg / pos) if pos > 0 else 1.0
        task_pos_weights[task] = pw
        logger.info(f"[pos_weight] {task}: neg={neg}, pos={pos}, weight={pw:.2f}")

    task_pw_tensors = {
        task: torch.tensor([pw], device=cfg["device"])
        for task, pw in task_pos_weights.items()
    }

    train_dataset = GenomeBagMultiTaskDataset(df_train, task_list)
    val_dataset = GenomeBagMultiTaskDataset(df_val, task_list)
    test_dataset = GenomeBagMultiTaskDataset(df_test, task_list)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn_multitask,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn_multitask,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn_multitask,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = GenomeMultiClassifier2(
        input_dim=cfg["input_dim"],
        task_list=task_list,
        task_num_classes=task_num_classes,
        agg_mode=cfg["agg_mode"],
        num_heads=cfg["num_heads"],
        num_inds=cfg["num_inds"],
        num_seeds=cfg["num_seeds"],
        hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"],
    ).to(cfg["device"])

    log_model_info(logger, model, device=cfg["device"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=1e-4
    )
    scaler = torch.cuda.amp.GradScaler()
    best_val_macro_bacc = -1.0
    best_state = None
    early_stopper = EarlyStopping(patience=10)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x, inst_mask, y, task_mask, _ in train_loader:
            x = x.to(cfg["device"])
            inst_mask = inst_mask.to(cfg["device"])
            y = y.to(cfg["device"])
            task_mask = task_mask.to(cfg["device"])

            with torch.cuda.amp.autocast():
                logits_dict, _ = model(x, inst_mask)

                loss = masked_multitask_loss(
                    logits_dict,
                    y,
                    task_mask,
                    task_list,
                    task_pw_tensors  
                )

            if loss is None:
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch} | Train loss={avg_loss:.4f}")

        per_task_val, macro_val, _, _ = eval_multitask(
            model,
            val_loader,
            cfg["device"],
            task_list,
            task_is_binary,
            task_num_classes
        )

        val_macro_bacc = macro_val.get("BAcc", float("nan"))

        logger.info(
            f"Epoch {epoch} | Val macro: " +
            " | ".join([f"{k}={v:.4f}" for k, v in macro_val.items()])
        )

        if val_macro_bacc > best_val_macro_bacc:
            best_val_macro_bacc = val_macro_bacc
            best_state = copy.deepcopy(model.state_dict())

        if early_stopper.step(val_macro_bacc):
            logger.info("Early stopping triggered")
            break

    model.load_state_dict(best_state)

    # ── 保存 encoder 权重 ──────────────────────────────────────────────
    encoder_ckpt = os.path.join(cfg["save_dir"], f"encoder_run{run+1}.pt")
    encoder_state = {k: v for k, v in best_state.items()
                     if k.startswith("aggregator") or k.startswith("shared_mlp")}
    torch.save(encoder_state, encoder_ckpt)
    logger.info(f"[Saved] Encoder weights → {encoder_ckpt}")
    logger.info(f"[Saved] Encoder key count: {len(encoder_state)}, "
                f"sample keys: {list(encoder_state.keys())[:3]}")
    # ──────────────────────────────────────────────────────────────────

    per_task_test, macro_test, y_true, y_pred = eval_multitask(
        model,
        test_loader,
        cfg["device"],
        task_list,
        task_is_binary,
        task_num_classes
    )

    run_summary = {
        "run": run + 1,
        "macro": macro_test,
        "per_task": per_task_test
    }
    all_run_summary.append(run_summary)

    logger.info(
        f"Run {run+1} | Test macro: " +
        " | ".join([f"{k}={v:.4f}" for k, v in macro_test.items()])
    )

    out_json = os.path.join(
        cfg["save_dir"],
        f"{cfg['agg_mode']}_multitask_summary_perrun.json"
    )

    with open(out_json, "w") as f:
        json.dump(all_run_summary, f, indent=2)


# =====================================================
# Save summary
# =====================================================
out_json = os.path.join(cfg["save_dir"], f"{cfg['agg_mode']}_multitask_summary.json")
with open(out_json, "w") as f:
    json.dump(all_run_summary, f, indent=2)

logger.info(f"[DONE] saved summary to {out_json}")

end_time = time.time()
log_training_time(logger, start_time, end_time)
logger.info("Training completed successfully.")
