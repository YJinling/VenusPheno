import torch
import torch.nn as nn
import torch.nn.functional as F
from venuspheno.models.aggregator import Aggregator

class GenomeClassifier(nn.Module):

    def __init__(self,
                 input_dim: int,
                 agg_mode: str = "isab_pma",
                 num_heads: int = 4,
                 num_inds: int = 128,
                 num_seeds: int = 1,
                 hidden_dim: int = 512,
                 dropout: float = 0.3,
                 num_classes: int = 1,
                 ln: bool = False):
        super().__init__()

        self.num_seeds = num_seeds
        self.input_dim = input_dim

        # ---------- 1. Aggregator ----------
        self.aggregator = Aggregator(
            input_dim=input_dim,
            mode=agg_mode,
            num_heads=num_heads,
            num_inds=num_inds,
            num_seeds=num_seeds,
            ln=ln
        )

        # ---------- 2. classifier 输入维度自动适配 ----------
        if num_seeds > 1 and agg_mode in ["pma", "isab_pma"]:
            fc_input_dim = input_dim * num_seeds
        else:
            fc_input_dim = input_dim

        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, x, mask=None, return_attn=False):

        if return_attn:
            bag_emb, attn = self.aggregator(x, mask, return_attn=True)
        else:
            bag_emb = self.aggregator(x, mask)

        # ---------- 多 seed flatten ----------
        if bag_emb.dim() == 3:     # (B, M, D)
            B, M, D = bag_emb.shape
            bag_emb = bag_emb.reshape(B, M * D)

        logits = self.classifier(bag_emb)

        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        if return_attn:
            return logits, bag_emb, attn

        return logits, bag_emb

# class GenomeClassifier(nn.Module):
#     """
#     Genome-level classifier with flexible aggregation module.
#     用于病原体预测任务的分类器。
#     输入: 一批bag (B, N, D)，每个bag是一个基因组中所有蛋白的embedding。
#     输出: (B, 1) 的致病性预测分数。
#     """
#     def __init__(self,
#                  input_dim: int,
#                  agg_mode: str = "isab_pma",
#                  num_heads: int = 4,
#                  num_inds: int = 128,
#                  num_seeds: int = 1,
#                  hidden_dim: int = 512,
#                  dropout: float = 0.3,
#                  num_classes: int = 1,
#                  ln: bool = False):
#         super().__init__()

#         # --- 1. Aggregation 层 ---
#         self.aggregator = Aggregator(
#             input_dim=input_dim,
#             mode=agg_mode,
#             num_heads=num_heads,
#             num_inds=num_inds,
#             num_seeds=num_seeds,
#             ln=ln
#         )

#         # --- 2. Bag-level classifier ---
#         # 这里是聚合后的全连接层，用于分类
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, num_classes)
#         )

#         # 如果是二分类任务（pathogenic vs non-pathogenic）
#         # 通常输出 raw logits, 再在 loss 中用 BCEWithLogitsLoss
#         self.num_classes = num_classes

#     def forward(self, x, mask=None, return_attn=False):
#         if return_attn:
#             bag_emb, attn = self.aggregator(x, mask, return_attn=True)
#             logits = self.classifier(bag_emb)
#             return logits.squeeze(-1), bag_emb, attn

#         bag_emb = self.aggregator(x, mask)
#         logits = self.classifier(bag_emb)
#         return logits.squeeze(-1), bag_emb



class GenomeMultiClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_list,
        agg_mode: str = "isab_pma",
        num_heads: int = 4,
        num_inds: int = 128,
        num_seeds: int = 1,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        ln: bool = False
    ):
        super().__init__()

        # ---------- 1. shared aggregator ----------
        self.aggregator = Aggregator(
            input_dim=input_dim,
            mode=agg_mode,
            num_heads=num_heads,
            num_inds=num_inds,
            num_seeds=num_seeds,
            ln=ln
        )

        # ---------- 2. shared projection ----------
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ---------- 3. task-specific heads ----------
        self.task_list = task_list
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(hidden_dim, 1)
            for task in task_list
        })

    def forward(self, x, mask=None, return_attn=False):
        if return_attn:
            bag_emb, attn = self.aggregator(x, mask, return_attn=True)
        else:
            bag_emb = self.aggregator(x, mask)

        z = self.shared_mlp(bag_emb)

        logits = {}
        for task, head in self.task_heads.items():
            logits[task] = head(z).squeeze(-1)   # (B,)

        if return_attn:
            return logits, bag_emb, attn
        return logits, bag_emb

class GenomeMultiClassifier2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_list,
        task_num_classes: dict,  
        agg_mode: str = "isab_pma",
        num_heads: int = 4,
        num_inds: int = 128,
        num_seeds: int = 1,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        ln: bool = False
    ):
        super().__init__()

        self.task_list = task_list
        self.task_num_classes = task_num_classes

        # ---------- 1. shared aggregator ----------
        self.aggregator = Aggregator(
            input_dim=input_dim,
            mode=agg_mode,
            num_heads=num_heads,
            num_inds=num_inds,
            num_seeds=num_seeds,
            ln=ln
        )

        # ---------- 2. shared projection ----------
        if num_seeds > 1 and agg_mode in ["pma", "isab_pma"]:
            fc_input_dim = input_dim * num_seeds
        else:
            fc_input_dim = input_dim

        self.shared_mlp = nn.Sequential(
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ---------- 3. task-specific heads ----------
        self.task_heads = nn.ModuleDict()

        for task in task_list:
            n_cls = task_num_classes[task]

            if n_cls == 2:
                # binary → 1 logit
                self.task_heads[task] = nn.Linear(hidden_dim, 1)
            else:
                # multi-class → C logits
                self.task_heads[task] = nn.Linear(hidden_dim, n_cls)

    def forward(self, x, mask=None, return_attn=False):
        if return_attn:
            bag_emb, attn = self.aggregator(x, mask, return_attn=True)
        else:
            bag_emb = self.aggregator(x, mask)
            attn = None

        if bag_emb.dim() == 3:
            B, M, D = bag_emb.shape
            bag_emb = bag_emb.reshape(B, M * D)

        z = self.shared_mlp(bag_emb)

        logits = {}
        for task in self.task_list:
            head = self.task_heads[task]
            out = head(z)

            if self.task_num_classes[task] == 2:
                logits[task] = out.squeeze(-1)
            else:
                logits[task] = out

        if return_attn:
            return logits, bag_emb, attn

        return logits, bag_emb



class GenomeClassifierNoAgg(nn.Module):
    """
    Genome-level classifier WITHOUT aggregation.
    输入: (B, D)
    输出:
        - binary: (B,)
        - multi-class: (B, C)
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 dropout: float = 0.3,
                 num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, mask=None, return_attn=False):
        """
        x: (B, D)
        """
        bag_emb = x
        logits = self.classifier(bag_emb)

        # -------- binary --------
        if self.num_classes == 1:
            logits = logits.squeeze(-1)  # (B,)

        # -------- multi-class --------
        # 保持 (B, C)
        if return_attn:
            return logits, bag_emb, None

        return logits, bag_emb


class GenomeMultiClassifierNoAgg2(nn.Module):
    """
    输入: x (B, D) genome-level embedding
    输出:
      - binary task: (B,)
      - multi-class task: (B, C)
    """
    def __init__(
        self,
        input_dim: int,
        task_list,
        task_num_classes: dict,
        hidden_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()

        self.task_list = task_list
        self.task_num_classes = task_num_classes

        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.task_heads = nn.ModuleDict()
        for task in task_list:
            n_cls = task_num_classes[task]
            if n_cls == 2:
                self.task_heads[task] = nn.Linear(hidden_dim, 1)
            else:
                self.task_heads[task] = nn.Linear(hidden_dim, n_cls)

    def forward(self, x):
        z = self.shared_mlp(x)

        logits = {}
        for task in self.task_list:
            out = self.task_heads[task](z)
            if self.task_num_classes[task] == 2:
                logits[task] = out.squeeze(-1)
            else:
                logits[task] = out
        return logits, z
