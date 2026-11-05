import numpy as np
import json
import argparse
import gc
import os, time, torch
import pandas as pd
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from models.classifier import GenomeClassifier
from utils.metrics import compute_metrics
from utils.dataset import GenomeBagDataset, collate_fn
from utils.logger import get_logger, log_model_info, log_training_time
from utils.earlystopping import EarlyStopping


# =====================================================
# Args
# =====================================================
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="isab_pma",
                    choices=["mean", "max", "attention", "pma", "isab_mean", "isab_pma", "hsp"])
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--save_root", type=str, default="Result")
parser.add_argument("--metadata",  type=str, required=True)
parser.add_argument("--train_txt", type=str, required=True)
parser.add_argument("--val_txt", type=str, default=None,
                    help="Optional fixed validation IDs. When supplied, no train/validation resplit is performed.")
parser.add_argument("--test_txt",  type=str, required=True)
parser.add_argument("--ckpt", type=str, default=None,
                    help="Path to trained model checkpoint (.pt)")
parser.add_argument("--attn_only", action="store_true",
                    help="Skip training and only extract attention")
parser.add_argument("--extract_attn", action="store_true",
                    help="Extract PMA attention after training and save per-sample scores")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_repeat", type=int, default=1)
parser.add_argument("--ln", action="store_true",
                    help="Enable LayerNorm inside Set Transformer aggregator blocks")
parser.add_argument("--head_type", type=str, default="baseline",
                    choices=["baseline", "norm_mlp", "res_mlp"])
parser.add_argument("--input_proj", action="store_true",
                    help="Enable pre-aggregator input projection")
parser.add_argument("--seed_mab", action="store_true",
                    help="Refine multi-seed PMA outputs with one seed-level SAB/MAB block")
parser.add_argument("--optimizer", type=str, default="adam",
                    choices=["adam", "adamw"])
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--grad_clip_norm", type=float, default=0.0)
parser.add_argument("--num_heads", type=int, default=2)
parser.add_argument("--num_inds", type=int, default=32)
parser.add_argument("--num_seeds", type=int, default=1)
parser.add_argument("--hsp_out_tokens", type=int, default=4)
parser.add_argument("--hsp_rank", type=int, default=4)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--seed_orth_lambda", type=float, default=0.0)
parser.add_argument("--checkpoint_metric", type=str, default="auprc",
                    choices=["auprc", "bacc"],
                    help="Validation metric used for checkpoint selection; BAcc uses the fixed 0.5 logit threshold.")
parser.add_argument("--decision_threshold", type=float, default=None,
                    help="Fixed probability threshold for final binary metrics; omit to retain legacy validation-MCC tuning.")
parser.add_argument("--amp", action="store_true",
                    help="Use CUDA automatic mixed precision to reduce B32 attention-memory peaks.")
parser.add_argument("--empty_cache_each_epoch", action="store_true",
                    help="Release unused CUDA cache after validation and between repeat seeds.")
args = parser.parse_args()


# =====================================================
# Config
# =====================================================
cfg = {
    "metadata":  args.metadata,
    "train_txt": args.train_txt,
    "val_txt":   args.val_txt,
    "test_txt":  args.test_txt,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": args.save_root,

    # model
    "input_dim": 1024,
    "agg_mode": args.mode,
    "num_heads": args.num_heads,
    "num_inds": args.num_inds,
    "num_seeds": args.num_seeds,
    "hsp_out_tokens": args.hsp_out_tokens,
    "hsp_rank": args.hsp_rank,
    "hidden_dim": args.hidden_dim,
    "dropout": args.dropout,
    "seed_orth_lambda": args.seed_orth_lambda,
    "ln": args.ln,
    "head_type": args.head_type,
    "input_proj": args.input_proj,
    "seed_mab": args.seed_mab,

    # training
    "batch_size": args.batch_size,
    "epochs": args.epochs,
    "lr": args.lr,
    "seed": args.seed,
    "n_repeat": args.n_repeat,
    "optimizer": args.optimizer,
    "weight_decay": args.weight_decay,
    "grad_clip_norm": args.grad_clip_norm,
    "amp": args.amp,
    "empty_cache_each_epoch": args.empty_cache_each_epoch,
}

os.makedirs(cfg["save_dir"], exist_ok=True)
logger = get_logger(cfg["save_dir"], name=f"train_{cfg['agg_mode']}")
logger.info("Training started")


# =====================================================
# Utils
# =====================================================
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.metrics import f1_score, matthews_corrcoef, balanced_accuracy_score, accuracy_score

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


def seed_orth_loss(seed_emb):
    if seed_emb is None or seed_emb.dim() != 3 or seed_emb.size(1) < 2:
        return seed_emb.new_tensor(0.0) if seed_emb is not None else torch.tensor(0.0)
    z = F.normalize(seed_emb, dim=-1)
    gram = torch.bmm(z, z.transpose(1, 2))
    eye = torch.eye(gram.size(1), device=gram.device, dtype=gram.dtype).unsqueeze(0)
    return ((gram * (1.0 - eye)) ** 2).sum() / (gram.size(0) * gram.size(1) * (gram.size(1) - 1))


def split_by_txt(meta_csv, train_txt, test_txt, val_txt=None, val_ratio=0.2, seed=42):
    df = pd.read_csv(meta_csv, dtype={"gcf": str})
    df["gcf"] = df["gcf"].str.strip()

    def read_ids(path):
        with open(path) as f:
            return {line.strip() for line in f if line.strip()}

    train_ids = read_ids(train_txt)
    test_ids = read_ids(test_txt)

    print("Meta size:", len(df))
    print("Train IDs:", len(train_ids))
    print("Intersection (train):", len(set(df["gcf"]) & train_ids))

    df_trainval = df[df["gcf"].isin(train_ids)].reset_index(drop=True)
    df_test     = df[df["gcf"].isin(test_ids)].reset_index(drop=True)

    if len(df_trainval) == 0:
        raise RuntimeError("No overlap between meta and train.txt IDs")

    if val_txt is not None:
        val_ids = read_ids(val_txt)
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise RuntimeError("train/val/test ID files overlap")
        missing_ids = (train_ids | val_ids | test_ids) - set(df["gcf"])
        if missing_ids:
            raise RuntimeError(f"{len(missing_ids)} split IDs are absent from metadata")
        df_val = df[df["gcf"].isin(val_ids)].reset_index(drop=True)
        if df_val.empty or df_test.empty:
            raise RuntimeError("Fixed validation or test split is empty")
        return df_trainval, df_val, df_test

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
# [NEW] Attention extraction
# =====================================================
def extract_pma_attention(model, loader, device, agg_mode):
    """
    逐样本提取 PMA attention，返回每个样本的蛋白重要性分数。

    对于 isab_pma / pma：
        PMA attention shape: (B, num_heads, num_seeds, N_padded)
        我们对 num_heads 取均值，去掉 seed 维，
        再用 mask 去除 padding，得到每个蛋白真实的 attention 权重。

    返回值：
        results: list of dict，每个 dict 包含：
            - gcf:        基因组 ID（str）
            - label:      真实标签（int）
            - prob:       预测概率（float）
            - attn:       np.array shape (N_real,)，每个蛋白的 attention 分数
                          已归一化（softmax over real tokens）
    """
    if agg_mode not in ("pma", "isab_pma"):
        logger.warning(f"agg_mode={agg_mode} 不支持 PMA attention 提取，跳过。")
        return []

    model.eval()
    results = []

    with torch.no_grad():
        for x, mask, y, gcf_ids in loader:
            # x:    (B, N_pad, D)
            # mask: (B, N_pad)   1=real token, 0=padding
            # y:    (B,)
            # gcf_ids: list of str, len=B

            x    = x.to(device)
            mask = mask.to(device)

            # return_attn=True 触发 aggregator 里的 attention 返回路径
            # GenomeClassifier.forward 返回 (logits, bag_emb, attn_dict)
            logits, _, attn_info = model(x, mask, return_attn=True)

            probs = sigmoid(logits.cpu().numpy())

            # --------------------------------------------------
            # 解析 PMA attention
            # attn_info 结构（来自 aggregator.forward isab_pma 分支）：
            #   {
            #     "isab": {"induce_to_gene": ..., "gene_to_induce": ...},
            #     "pma":  (B, num_heads, num_seeds, N_pad)
            #   }
            # 对于纯 pma 模式，attn_info 直接是 (B, num_heads, num_seeds, N_pad)
            # --------------------------------------------------
            if agg_mode == "isab_pma":
                pma_attn = attn_info["pma"]   # (B, H, S, N_pad)
            else:  # pma
                pma_attn = attn_info          # (B, H, S, N_pad)

            # 对 head 取均值，去掉 seed 维 → (B, N_pad)
            pma_attn = pma_attn.mean(dim=1)   # (B, S, N_pad)
            if pma_attn.shape[1] == 1:
                pma_attn = pma_attn.squeeze(1)  # (B, N)
            else:
                raise ValueError(f"Expected num_seeds=1, got shape {pma_attn.shape}")

            pma_attn = pma_attn.cpu().numpy()
            mask_np  = mask.cpu().numpy()     # (B, N_pad)

            for i, gcf in enumerate(gcf_ids):
                real_len  = int(mask_np[i].sum())          # 真实蛋白数量
                attn_real = pma_attn[i, :real_len]         # (N_real,)

                # 重新对真实 token 做 softmax（去掉 padding 后归一化）
                attn_real = attn_real - attn_real.max()    # 数值稳定
                attn_real = np.exp(attn_real)
                attn_real = attn_real / attn_real.sum()

                results.append({
                    "gcf":   gcf,
                    "label": int(y[i].item()),
                    "prob":  float(probs[i]) if probs.ndim == 1 else float(probs[i][0]),
                    "attn":  attn_real,       # np.array (N_real,)
                })

    return results


def save_attention_results(results, save_dir, split_name, run_id):
    """
    保存 attention 结果：
    1. 每个样本单独存一个 .npy（蛋白级别 attention 向量）
    2. 汇总的 summary CSV（gcf, label, prob, top10_indices, top10_scores）
    """
    attn_dir = os.path.join(save_dir, f"run{run_id}_attn_{split_name}")
    os.makedirs(attn_dir, exist_ok=True)

    summary_rows = []

    for r in results:

        gcf  = r["gcf"]
        attn = r["attn"]

        if isinstance(gcf, (list, tuple)):
            gcf = gcf[0]

        gcf = str(gcf).split("_prot_")[0]

        np.save(os.path.join(attn_dir, f"{gcf}.npy"), attn)

        # Top-10 蛋白 index（按 attention 降序）
        top_k  = min(10, len(attn))
        top_idx = np.argsort(attn)[::-1][:top_k]

        summary_rows.append({
            "gcf":          gcf,
            "label":        r["label"],
            "prob":         round(r["prob"], 4),
            "top10_indices": list(top_idx),
            "top10_scores":  [round(float(attn[j]), 6) for j in top_idx],
            "attn_entropy":  float(-np.sum(attn * np.log(attn + 1e-12))),  # 分散度
            "attn_max":      float(attn.max()),
            "n_proteins":    len(attn),
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(save_dir, f"run{run_id}_attn_{split_name}_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    logger.info(f"[Attention] Saved {len(results)} samples → {attn_dir}")
    logger.info(f"[Attention] Summary → {csv_path}")

    return df_summary


# =====================================================
# Main
# =====================================================
all_test_metrics = []
start_time = time.time()

for run in range(cfg["n_repeat"]):
    logger.info(f"\n========== Run {run+1}/{cfg['n_repeat']} ==========")
    run_seed = cfg["seed"] + run
    logger.info(f"Run seed: {run_seed}")

    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    # ---------- split ----------
    df_train, df_val, df_test = split_by_txt(
        cfg["metadata"],
        train_txt=cfg["train_txt"],
        val_txt=cfg["val_txt"],
        test_txt=cfg["test_txt"],
        seed=run_seed
    )

    train_dataset = GenomeBagDataset(df_train)
    val_dataset   = GenomeBagDataset(df_val)
    test_dataset  = GenomeBagDataset(df_test)

    num_classes = df_train["label"].nunique()
    is_binary   = True
    logger.info(f"Detected task type: {'binary' if is_binary else f'{num_classes}-class'}")

    # ---------- sampler ----------
    if is_binary:
        train_labels = train_dataset.df["label"].values
        pos = (train_labels == 1).sum()
        neg = (train_labels == 0).sum()
        logger.info(f"Train split: +{pos} / -{neg}")
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                              shuffle=shuffle, sampler=sampler, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=collate_fn)

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
        num_classes=model_num_classes,
        ln=cfg["ln"],
        head_type=cfg["head_type"],
        input_proj=cfg["input_proj"],
        seed_mab=cfg["seed_mab"],
        hsp_out_tokens=cfg["hsp_out_tokens"],
        hsp_rank=cfg["hsp_rank"],
    ).to(cfg["device"])
    log_model_info(logger, model, device=cfg["device"])

    if args.ckpt is not None:
        logger.info(f"Loading checkpoint: {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=cfg["device"])
        model.load_state_dict(ckpt["state_dict"])

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

    if cfg["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    use_amp = cfg["amp"] and cfg["device"] == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if not args.attn_only:
        # ---------- training ----------
        best_val_score = -1.0
        best_state     = None
        early_stopper  = EarlyStopping(patience=10)

        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total_loss = 0.0

            for x, mask, y, _ in train_loader:
                x, mask, y = x.to(cfg["device"]), mask.to(cfg["device"]), y.to(cfg["device"])
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    if cfg["seed_orth_lambda"] > 0:
                        logits, _, seed_emb = model(x, mask, return_seed_emb=True)
                    else:
                        logits, _ = model(x, mask)
                    loss = criterion(logits, y.float() if is_binary else y.long())
                    if cfg["seed_orth_lambda"] > 0:
                        loss = loss + cfg["seed_orth_lambda"] * seed_orth_loss(seed_emb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if cfg["grad_clip_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            logger.info(f"Epoch {epoch} | Train loss={total_loss / len(train_loader):.4f}")

            # ---------- validation ----------
            model.eval()
            y_true, y_logits = [], []

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                for x, mask, y, _ in val_loader:
                    x, mask = x.to(cfg["device"]), mask.to(cfg["device"])
                    logits, _ = model(x, mask)
                    y_logits.extend(logits.view(-1).cpu().tolist())
                    y_true.extend(y.tolist())

            y_true = np.array(y_true)
            y_prob = sigmoid(np.array(y_logits))

            val_auprc = average_precision_score(y_true, y_prob)
            val_auroc = roc_auc_score(y_true, y_prob)
            val_bacc = balanced_accuracy_score(y_true, (y_prob >= 0.5).astype(int))
            val_score = val_bacc if args.checkpoint_metric == "bacc" else val_auprc
            logger.info(
                f"Epoch {epoch} | Val AUROC={val_auroc:.4f} | Val AUPRC={val_auprc:.4f} "
                f"| Val BAcc@0.5={val_bacc:.4f} | checkpoint_{args.checkpoint_metric}={val_score:.4f}"
            )

            if cfg["empty_cache_each_epoch"] and cfg["device"] == "cuda":
                del x, mask, y, logits, loss
                if cfg["seed_orth_lambda"] > 0:
                    del seed_emb
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Epoch %d | released unused CUDA cache", epoch)

            if val_score > best_val_score:
                best_val_score = val_score
                best_state = model.state_dict()

            if early_stopper.step(val_score):
                logger.info("Early stopping triggered")
                break

        # [NEW] 保存 best model 权重
        model.load_state_dict(best_state)
        ckpt_path = os.path.join(cfg["save_dir"], f"run{run+1}_best_model.pt")
        torch.save({
            "run":          run + 1,
            "seed":         run_seed,
            "state_dict":   best_state,
            "checkpoint_metric": args.checkpoint_metric,
            "best_val_score": best_val_score,
            "cfg":          cfg,
        }, ckpt_path)
        logger.info(f"[Checkpoint] Saved best model → {ckpt_path}")

        # ---------- find best threshold on validation ----------
        val_true, val_logits = [], []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            for x, mask, y, _ in val_loader:
                x, mask = x.to(cfg["device"]), mask.to(cfg["device"])
                logits, _ = model(x, mask)
                val_logits.extend(logits.view(-1).cpu().tolist())
                val_true.extend(y.tolist())

        val_true = np.array(val_true)
        val_prob = sigmoid(np.array(val_logits))

        if args.decision_threshold is None:
            best_threshold, best_score = find_best_threshold(val_true, val_prob, metric="mcc")
            logger.info(f"Best threshold on val: {best_threshold:.4f} | MCC={best_score:.4f}")
        else:
            best_threshold = args.decision_threshold
            logger.info(f"Fixed decision threshold: {best_threshold:.4f}")

        # ---------- test ----------
        y_true, y_logits = [], []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
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
            "F1":    f1_score(y_true, y_pred),
            "MCC":   matthews_corrcoef(y_true, y_pred),
            "BAcc":  balanced_accuracy_score(y_true, y_pred),
            "ACC":   accuracy_score(y_true, y_pred),
        }
        all_test_metrics.append(test_metrics)

        logger.info(
            f"Run {run+1} | " +
            " | ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
        )

        np.save(os.path.join(cfg["save_dir"], f"run{run+1}_test_y_true.npy"), y_true)
        np.save(os.path.join(cfg["save_dir"], f"run{run+1}_test_y_pred.npy"), y_pred)

    if args.attn_only:

        logger.info("Running attention extraction only")

        for split_name, split_loader in [
            ("train", train_loader),
            ("val",   val_loader),
            ("test",  test_loader)
        ]:

            results = extract_pma_attention(
                model, split_loader, cfg["device"], cfg["agg_mode"]
            )

            save_attention_results(results, cfg["save_dir"], split_name, run + 1)

        logger.info("Attention extraction finished")
        sys.exit(0)

    # [NEW] 提取 PMA attention（train/val/test 三个 split 都存）
    if args.extract_attn and cfg["agg_mode"] in ("pma", "isab_pma"):
        logger.info(f"[Attention] Extracting PMA attention for run {run+1} ...")

        for split_name, split_loader in [("train", train_loader),
                                          ("val",   val_loader),
                                          ("test",  test_loader)]:
            results = extract_pma_attention(
                model, split_loader, cfg["device"], cfg["agg_mode"]
            )
            save_attention_results(results, cfg["save_dir"], split_name, run + 1)

    if not args.attn_only:
        del best_state
    del model, optimizer, criterion, scaler, train_loader, val_loader, test_loader
    if cfg["empty_cache_each_epoch"] and cfg["device"] == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("Run %d | released CUDA cache before next seed", run + 1)

if len(all_test_metrics) == 0:
    logger.info("Attention-only mode: skip metric summary")
    sys.exit(0)
# =====================================================
# Final report
# =====================================================
summary = {}
metric_keys = all_test_metrics[0].keys()

for k in metric_keys:
    vals = [m[k] for m in all_test_metrics]
    summary[k] = {
        "mean":     float(np.mean(vals)),
        "std":      float(np.std(vals)),
        "all_runs": vals,
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
