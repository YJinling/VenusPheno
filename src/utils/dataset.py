import os
import torch
import pandas as pd
import numpy as np
import h5py
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class GenomeBagDataset(Dataset):
    def __init__(self, metadata):
        """
        metadata:
            - str: path to metadata.csv
            - pd.DataFrame: already-loaded metadata
        metadata 必须包含列：gcf, h5_path, label
        """
        if isinstance(metadata, pd.DataFrame):
            self.df = metadata.reset_index(drop=True)
        elif isinstance(metadata, str):
            self.df = pd.read_csv(metadata)
        else:
            raise TypeError(
                f"GenomeBagDataset expects str or DataFrame, got {type(metadata)}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        h5_path = row["h5_path"]
        label = int(row["label"])

        with h5py.File(h5_path, "r") as f:
            x = torch.from_numpy(f["Embeddings"][:]).float()
            gene_ids = [g.decode("utf-8") for g in f["Names"][:]]

        return x, gene_ids, label


class GenomeBagMultiTaskDataset(Dataset):
    def __init__(self, metadata, task_list):
        if isinstance(metadata, pd.DataFrame):
            self.df = metadata.reset_index(drop=True)
        elif isinstance(metadata, str):
            self.df = pd.read_csv(metadata)
        else:
            raise TypeError(
                f"GenomeBagMultiTaskDataset expects str or DataFrame, got {type(metadata)}"
            )
        self.task_list = task_list

        # ---------- genome-level aggregation ----------
        self.genomes = (
            self.df.groupby("genome_id")
            .agg({
                "h5_path": "first",
                "task": list,
                "label": list
            })
            .reset_index()
        )

        self.task2idx = {t: i for i, t in enumerate(task_list)}

    def __len__(self):
        return len(self.genomes)

    def __getitem__(self, idx):
        row = self.genomes.iloc[idx]
        h5_path = row["h5_path"]

        with h5py.File(h5_path, "r") as f:
            x = torch.from_numpy(f["Embeddings"][:]).float()
            gene_ids = [g.decode() for g in f["Names"][:]]

        y = torch.zeros(len(self.task_list))
        mask = torch.zeros(len(self.task_list), dtype=torch.bool)

        for t, lab in zip(row["task"], row["label"]):
            ti = self.task2idx[t]
            y[ti] = lab
            mask[ti] = True

        return x, gene_ids, y, mask

class GenomeEmbeddingDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with h5py.File(row["h5_path"], "r") as f:
            emb = f["embedding"][:]

        return torch.tensor(emb, dtype=torch.float32), torch.tensor(row["label"])

import torch
from torch.utils.data import Dataset
import pandas as pd
import h5py


class GenomeEmbeddingMultiTaskDataset(Dataset):

    def __init__(self, df, task_list):
        self.task_list = task_list
        self.task2idx = {t: i for i, t in enumerate(task_list)}
        T = len(task_list)

        grouped = df.groupby("genome_id")

        self.h5_paths = []
        self.y_list = []
        self.mask_list = []

        for genome_id, sub_df in grouped:
            h5_path = sub_df["h5_path"].iloc[0]

            y = torch.zeros(T, dtype=torch.float32)
            mask = torch.zeros(T, dtype=torch.bool)

            tasks = sub_df["task"].values
            labels = sub_df["label"].values

            for t, lab in zip(tasks, labels):
                ti = self.task2idx[t]
                y[ti] = float(lab)
                mask[ti] = True

            self.h5_paths.append(h5_path)
            self.y_list.append(y)
            self.mask_list.append(mask)

    def __len__(self):
        return len(self.h5_paths)

    def __getitem__(self, idx):
        h5_path = self.h5_paths[idx]

        with h5py.File(h5_path, "r") as f:
            emb = torch.from_numpy(f["embedding"][:]).float()

        return emb, self.y_list[idx], self.mask_list[idx]



def collate_fn(batch):
    batch_x, batch_gene_ids, batch_y = zip(*batch)

    max_len = max(x.shape[0] for x in batch_x)
    dim = batch_x[0].shape[1]
    B = len(batch_x)

    x = torch.zeros(B, max_len, dim, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    y = torch.tensor(batch_y, dtype=torch.long)

    gene_ids = [[None] * max_len for _ in range(B)]

    for i, (xx, gids) in enumerate(zip(batch_x, batch_gene_ids)):
        n = xx.shape[0]
        x[i, :n] = xx
        mask[i, :n] = True
        gene_ids[i][:n] = gids

    return x, mask, y, gene_ids

def collate_fn_multitask(batch):
    batch_x, batch_gene_ids, batch_y, batch_task_mask = zip(*batch)

    max_len = max(x.shape[0] for x in batch_x)
    dim = batch_x[0].shape[1]
    B = len(batch_x)
    T = batch_y[0].shape[0]

    # ---------- instance padding ----------
    x = torch.zeros(B, max_len, dim, dtype=torch.float32)
    inst_mask = torch.zeros(B, max_len, dtype=torch.bool)

    gene_ids = [[None] * max_len for _ in range(B)]

    for i, (xx, gids) in enumerate(zip(batch_x, batch_gene_ids)):
        n = xx.shape[0]
        x[i, :n] = xx
        inst_mask[i, :n] = True
        gene_ids[i][:n] = gids

    # ---------- task labels ----------
    y = torch.stack(batch_y, dim=0).float()          # (B, T)
    task_mask = torch.stack(batch_task_mask, dim=0) # (B, T)

    return x, inst_mask, y, task_mask, gene_ids


def get_cv_splits_from_group_PF(meta_csv):
    """
    Returns:
        folds: list of dict
            each dict has keys: train_df, val_df
        test_df: DataFrame (group == 5)
    """
    df = pd.read_csv(meta_csv)

    required_cols = {"gcf", "h5_path", "label", "group"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"metadata must contain columns {required_cols}")

    test_df = df[df["group"] == 5].reset_index(drop=True)

    folds = []
    for val_group in [1, 2, 3, 4]:
        train_df = df[df["group"].isin([g for g in [1, 2, 3, 4] if g != val_group])]
        val_df   = df[df["group"] == val_group]

        folds.append({
            "val_group": val_group,
            "train_df": train_df.reset_index(drop=True),
            "val_df": val_df.reset_index(drop=True)
        })

    return folds, test_df


def split_by_group_fold(
    metadata_csv,
    fold_id,
    val_ratio=0.2,
    seed=42,
    stratify_label=True
):
    df = pd.read_csv(metadata_csv)

    assert "group" in df.columns, "metadata must contain 'group' column"
    assert "label" in df.columns, "metadata must contain 'label' column"

    # ---------- outer test ----------
    df_test = df[df["group"] == fold_id].reset_index(drop=True)
    df_trainval = df[df["group"] != fold_id].reset_index(drop=True)

    if len(df_test) == 0:
        raise ValueError(f"No samples found for fold {fold_id}")

    # ---------- inner train / val ----------
    stratify = None
    if stratify_label and df_trainval["label"].nunique() > 1:
        stratify = df_trainval["label"]

    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify
    )

    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)

    return df_train, df_val, df_test
