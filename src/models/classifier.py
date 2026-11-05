import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aggregator import Aggregator
import math

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
                 ln: bool = False,
                 head_type: str = "baseline",
                 input_proj: bool = False,
                 seed_mab: bool = False,
                 hsp_out_tokens: int = 4,
                 hsp_rank: int = 4):
        super().__init__()

        self.num_seeds = num_seeds
        self.input_dim = input_dim
        self.input_proj = (
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            if input_proj else nn.Identity()
        )

        # ---------- 1. Aggregator ----------
        self.aggregator = Aggregator(
            input_dim=input_dim,
            mode=agg_mode,
            num_heads=num_heads,
            num_inds=num_inds,
            num_seeds=num_seeds,
            ln=ln,
            seed_mab=seed_mab,
            hsp_out_tokens=hsp_out_tokens,
            hsp_rank=hsp_rank,
        )

        # ---------- 2. classifier 输入维度自动适配 ----------
        if agg_mode == "hsp":
            fc_input_dim = input_dim * hsp_out_tokens
        elif num_seeds > 1 and agg_mode in ["pma", "isab_pma"]:
            fc_input_dim = input_dim * num_seeds
        else:
            fc_input_dim = input_dim

        if head_type == "baseline":
            self.classifier = nn.Sequential(
                nn.Linear(fc_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        elif head_type == "norm_mlp":
            self.classifier = nn.Sequential(
                nn.LayerNorm(fc_input_dim),
                nn.Linear(fc_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        elif head_type == "res_mlp":
            self.classifier = ResidualMLPHead(
                fc_input_dim, hidden_dim, num_classes, dropout
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        self.num_classes = num_classes

    def forward(self, x, mask=None, return_attn=False, return_seed_emb=False):
        x = self.input_proj(x)

        if return_attn:
            if return_seed_emb:
                bag_emb, attn, seed_emb = self.aggregator(
                    x, mask, return_attn=True, return_seed_emb=True
                )
            else:
                bag_emb, attn = self.aggregator(x, mask, return_attn=True)
                seed_emb = bag_emb if bag_emb.dim() == 3 else None
        elif return_seed_emb:
            bag_emb, seed_emb = self.aggregator(x, mask, return_seed_emb=True)
        else:
            bag_emb = self.aggregator(x, mask)
            seed_emb = bag_emb if bag_emb.dim() == 3 else None

        # ---------- 多 seed flatten ----------
        if bag_emb.dim() == 3:     # (B, M, D)
            B, M, D = bag_emb.shape
            bag_emb = bag_emb.reshape(B, M * D)

        logits = self.classifier(bag_emb)

        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        if return_attn:
            return logits, bag_emb, attn

        if return_seed_emb:
            return logits, bag_emb, seed_emb

        return logits, bag_emb


class ResidualMLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block_norm = nn.LayerNorm(hidden_dim)
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        z = self.in_proj(x)
        z = z + self.block(self.block_norm(z))
        return self.out(z)


class SharedResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block_norm = nn.LayerNorm(hidden_dim)
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        z = self.in_proj(x)
        return z + self.block(self.block_norm(z))

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
        ln: bool = False,
        seed_mab: bool = False,
        hsp_out_tokens: int = 4,
        hsp_rank: int = 4,
    ):
        super().__init__()

        # ---------- 1. shared aggregator ----------
        self.aggregator = Aggregator(
            input_dim=input_dim,
            mode=agg_mode,
            num_heads=num_heads,
            num_inds=num_inds,
            num_seeds=num_seeds,
            ln=ln,
            seed_mab=seed_mab,
            hsp_out_tokens=hsp_out_tokens,
            hsp_rank=hsp_rank,
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
        ln: bool = False,
        seed_mab: bool = False,
        hsp_out_tokens: int = 4,
        hsp_rank: int = 4,
        shared_mlp_type: str = "linear",
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
            ln=ln,
            seed_mab=seed_mab,
            hsp_out_tokens=hsp_out_tokens,
            hsp_rank=hsp_rank,
        )

        # ---------- 2. shared projection ----------
        if agg_mode == "hsp":
            fc_input_dim = input_dim * hsp_out_tokens
        elif num_seeds > 1 and agg_mode in ["pma", "isab_pma"]:
            fc_input_dim = input_dim * num_seeds
        else:
            fc_input_dim = input_dim

        if shared_mlp_type == "res_mlp":
            self.shared_mlp = SharedResidualMLP(fc_input_dim, hidden_dim, dropout)
        elif shared_mlp_type == "linear":
            self.shared_mlp = nn.Sequential(
                nn.Linear(fc_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unknown shared_mlp_type: {shared_mlp_type}")

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

    def forward(self, x, mask=None, return_attn=False, return_seed_emb=False):
        if return_attn:
            if return_seed_emb:
                bag_emb, attn, seed_emb = self.aggregator(
                    x, mask, return_attn=True, return_seed_emb=True
                )
            else:
                bag_emb, attn = self.aggregator(x, mask, return_attn=True)
                seed_emb = bag_emb if bag_emb.dim() == 3 else None
        elif return_seed_emb:
            bag_emb, seed_emb = self.aggregator(x, mask, return_seed_emb=True)
        else:
            bag_emb = self.aggregator(x, mask)
            attn = None
            seed_emb = bag_emb if bag_emb.dim() == 3 else None

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
            return logits, z, attn

        if return_seed_emb:
            return logits, z, seed_emb

        return logits, z


class GenomeMultiClassifierSeedRouting(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_list,
        task_num_classes: dict,
        agg_mode: str = "isab_pma",
        num_heads: int = 4,
        num_inds: int = 64,
        num_seeds: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        ln: bool = False,
        seed_mab: bool = False,
        hsp_out_tokens: int = 4,
        hsp_rank: int = 4,
        shared_mlp_type: str = "res_mlp",
    ):
        super().__init__()
        if agg_mode not in ["pma", "isab_pma"]:
            raise ValueError("Seed routing requires agg_mode='pma' or 'isab_pma'")
        if num_seeds < 2:
            raise ValueError("Seed routing requires num_seeds >= 2")

        self.task_list = list(task_list)
        self.task_num_classes = task_num_classes
        self.num_seeds = num_seeds

        self.aggregator = Aggregator(
            input_dim=input_dim,
            mode=agg_mode,
            num_heads=num_heads,
            num_inds=num_inds,
            num_seeds=num_seeds,
            ln=ln,
            seed_mab=seed_mab,
            hsp_out_tokens=hsp_out_tokens,
            hsp_rank=hsp_rank,
        )

        fc_input_dim = input_dim * num_seeds
        if shared_mlp_type == "res_mlp":
            self.shared_mlp = SharedResidualMLP(fc_input_dim, hidden_dim, dropout)
        elif shared_mlp_type == "linear":
            self.shared_mlp = nn.Sequential(
                nn.Linear(fc_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unknown shared_mlp_type: {shared_mlp_type}")

        self.route_logits = nn.Parameter(torch.zeros(len(self.task_list), num_seeds))
        self.task_heads = nn.ModuleDict()
        for task in self.task_list:
            n_cls = task_num_classes[task]
            self.task_heads[task] = nn.Linear(hidden_dim, 1 if n_cls == 2 else n_cls)

    def routing_weights(self):
        return torch.softmax(self.route_logits, dim=1)

    def get_routing_matrix(self):
        return self.routing_weights().detach().cpu()

    def forward(self, x, mask=None, return_attn=False, return_seed_emb=False):
        if return_attn:
            if return_seed_emb:
                seed_emb, attn, raw_seed_emb = self.aggregator(
                    x, mask, return_attn=True, return_seed_emb=True
                )
            else:
                seed_emb, attn = self.aggregator(x, mask, return_attn=True)
                raw_seed_emb = seed_emb
        elif return_seed_emb:
            seed_emb, raw_seed_emb = self.aggregator(x, mask, return_seed_emb=True)
            attn = None
        else:
            seed_emb = self.aggregator(x, mask)
            raw_seed_emb = seed_emb
            attn = None

        if seed_emb.dim() != 3:
            raise ValueError(f"Seed routing expected (B, S, D), got {tuple(seed_emb.shape)}")

        weights = self.routing_weights()
        logits = {}
        task_z = {}
        for t_idx, task in enumerate(self.task_list):
            scale = (weights[t_idx] * self.num_seeds).view(1, self.num_seeds, 1)
            routed = (seed_emb * scale).reshape(seed_emb.size(0), -1)
            z = self.shared_mlp(routed)
            out = self.task_heads[task](z)
            logits[task] = out.squeeze(-1) if self.task_num_classes[task] == 2 else out
            task_z[task] = z

        z_global = torch.stack(list(task_z.values()), dim=0).mean(dim=0)
        if return_attn:
            aux = {
                "attn": attn,
                "routing_weights": weights.detach(),
                "task_embeddings": task_z,
            }
            return logits, z_global, aux
        if return_seed_emb:
            return logits, z_global, raw_seed_emb
        return logits, z_global


class TaskAdapter(nn.Module):
    def __init__(self, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z):
        return self.up(F.gelu(self.down(z)))


class GenomeMultiClassifierTaskAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_list,
        task_num_classes: dict,
        agg_mode: str = "isab_pma",
        num_heads: int = 4,
        num_inds: int = 64,
        num_seeds: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        ln: bool = False,
        seed_mab: bool = False,
        hsp_out_tokens: int = 4,
        hsp_rank: int = 4,
        shared_mlp_type: str = "res_mlp",
        adapter_bottleneck_dim: int = 64,
        adapter_gamma_init: float = 0.0,
    ):
        super().__init__()
        self.task_list = list(task_list)
        self.task_num_classes = task_num_classes

        self.aggregator = Aggregator(
            input_dim=input_dim,
            mode=agg_mode,
            num_heads=num_heads,
            num_inds=num_inds,
            num_seeds=num_seeds,
            ln=ln,
            seed_mab=seed_mab,
            hsp_out_tokens=hsp_out_tokens,
            hsp_rank=hsp_rank,
        )

        if agg_mode == "hsp":
            fc_input_dim = input_dim * hsp_out_tokens
        elif num_seeds > 1 and agg_mode in ["pma", "isab_pma"]:
            fc_input_dim = input_dim * num_seeds
        else:
            fc_input_dim = input_dim

        if shared_mlp_type == "res_mlp":
            self.shared_mlp = SharedResidualMLP(fc_input_dim, hidden_dim, dropout)
        elif shared_mlp_type == "linear":
            self.shared_mlp = nn.Sequential(
                nn.Linear(fc_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            raise ValueError(f"Unknown shared_mlp_type: {shared_mlp_type}")

        # Construct heads before private adapters so, under the same seed, the
        # shared backbone and heads match GenomeMultiClassifier2 exactly.
        self.task_heads = nn.ModuleDict()
        for task in self.task_list:
            n_cls = task_num_classes[task]
            self.task_heads[task] = nn.Linear(hidden_dim, 1 if n_cls == 2 else n_cls)
        self.adapters = nn.ModuleDict({
            task: TaskAdapter(hidden_dim, adapter_bottleneck_dim)
            for task in self.task_list
        })
        self.adapter_gammas = nn.ParameterDict({
            task: nn.Parameter(torch.full((1,), adapter_gamma_init))
            for task in self.task_list
        })

    def forward(self, x, mask=None, return_attn=False, return_seed_emb=False,
                return_adapter_info=False):
        if return_attn:
            if return_seed_emb:
                bag_emb, attn, seed_emb = self.aggregator(
                    x, mask, return_attn=True, return_seed_emb=True
                )
            else:
                bag_emb, attn = self.aggregator(x, mask, return_attn=True)
                seed_emb = bag_emb if bag_emb.dim() == 3 else None
        elif return_seed_emb:
            bag_emb, seed_emb = self.aggregator(x, mask, return_seed_emb=True)
            attn = None
        else:
            bag_emb = self.aggregator(x, mask)
            attn = None
            seed_emb = bag_emb if bag_emb.dim() == 3 else None

        if bag_emb.dim() == 3:
            bag_emb = bag_emb.reshape(bag_emb.size(0), -1)

        z = self.shared_mlp(bag_emb)
        logits = {}
        task_z = {}
        adapter_contributions = {}
        for task in self.task_list:
            adapter_contributions[task] = self.adapter_gammas[task] * self.adapters[task](z)
            z_t = z + adapter_contributions[task]
            out = self.task_heads[task](z_t)
            logits[task] = out.squeeze(-1) if self.task_num_classes[task] == 2 else out
            task_z[task] = z_t

        z_global = torch.stack(list(task_z.values()), dim=0).mean(dim=0)
        if return_adapter_info:
            return logits, z_global, {
                "shared_embedding": z,
                "adapter_contributions": adapter_contributions,
                "seed_embeddings": seed_emb,
                "attn": attn,
            }
        if return_attn:
            return logits, z_global, attn
        if return_seed_emb:
            return logits, z_global, seed_emb
        return logits, z_global


class GenomeMultiClassifierMMoE(nn.Module):
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
        ln: bool = False,
        num_experts: int = 4,
        seed_mab: bool = False,
        hsp_out_tokens: int = 4,
        hsp_rank: int = 4,
    ):
        super().__init__()
        self.task_list = list(task_list)
        self.task_num_classes = task_num_classes
        self.num_experts = num_experts

        self.aggregator = Aggregator(
            input_dim=input_dim,
            mode=agg_mode,
            num_heads=num_heads,
            num_inds=num_inds,
            num_seeds=num_seeds,
            ln=ln,
            seed_mab=seed_mab,
            hsp_out_tokens=hsp_out_tokens,
            hsp_rank=hsp_rank,
        )

        if agg_mode == "hsp":
            fc_input_dim = input_dim * hsp_out_tokens
        elif num_seeds > 1 and agg_mode in ["pma", "isab_pma"]:
            fc_input_dim = input_dim * num_seeds
        else:
            fc_input_dim = input_dim

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fc_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])
        self.gates = nn.ModuleDict({
            task: nn.Linear(fc_input_dim, num_experts)
            for task in self.task_list
        })
        self.task_heads = nn.ModuleDict()
        for task in self.task_list:
            n_cls = task_num_classes[task]
            self.task_heads[task] = nn.Linear(hidden_dim, 1 if n_cls == 2 else n_cls)

    def forward(self, x, mask=None, return_attn=False):
        if return_attn:
            bag_emb, attn = self.aggregator(x, mask, return_attn=True)
        else:
            bag_emb = self.aggregator(x, mask)
            attn = None

        if bag_emb.dim() == 3:
            B, M, D = bag_emb.shape
            bag_emb = bag_emb.reshape(B, M * D)

        expert_z = torch.stack([expert(bag_emb) for expert in self.experts], dim=1)

        logits = {}
        gate_weights = {}
        for task in self.task_list:
            gate = torch.softmax(self.gates[task](bag_emb), dim=1)
            task_z = torch.sum(expert_z * gate.unsqueeze(-1), dim=1)
            out = self.task_heads[task](task_z)
            logits[task] = out.squeeze(-1) if self.task_num_classes[task] == 2 else out
            if return_attn:
                gate_weights[task] = gate.detach()

        z_global = expert_z.mean(dim=1)
        if return_attn:
            return logits, z_global, {"attn": attn, "mmoe_gates": gate_weights}
        return logits, z_global



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

class GenomeMultiClassifierTaskRouter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_list,
        task_num_classes: dict,
        agg_mode: str = "pma",
        num_heads: int = 4,
        num_inds: int = 64,
        num_seeds: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        ln: bool = False
    ):
        super().__init__()

        assert agg_mode in ["pma", "isab_pma"], \
            "E6 requires agg_mode='pma' or 'isab_pma'"
        assert num_seeds in [2, 4], \
            "E6 is designed for PMA num_seeds=2 or 4"

        self.task_list = list(task_list)
        self.task_num_classes = task_num_classes
        self.num_tasks = len(task_list)
        self.num_seeds = num_seeds
        self.hidden_dim = hidden_dim

        self.aggregator = Aggregator(
            input_dim=input_dim,
            mode=agg_mode,
            num_heads=num_heads,
            num_inds=num_inds,
            num_seeds=num_seeds,
            ln=ln
        )

        self.seed_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.task_queries = nn.Parameter(
            torch.empty(self.num_tasks, hidden_dim)
        )
        nn.init.xavier_uniform_(self.task_queries)

        self.router_norm = nn.LayerNorm(hidden_dim)

        self.task_heads = nn.ModuleDict()
        for task in self.task_list:
            n_cls = task_num_classes[task]
            if n_cls == 2:
                self.task_heads[task] = nn.Linear(hidden_dim, 1)
            else:
                self.task_heads[task] = nn.Linear(hidden_dim, n_cls)

    def forward(self, x, mask=None, return_attn=False):
        seed_emb = self.aggregator(x, mask)

        if seed_emb.dim() == 2:
            seed_emb = seed_emb.unsqueeze(1)

        seed_z = self.seed_projector(seed_emb)

        logits = {}
        router_attn = {}

        for t_idx, task in enumerate(self.task_list):
            q = self.task_queries[t_idx].view(1, 1, -1)

            score = (seed_z * q).sum(dim=-1) / math.sqrt(self.hidden_dim)
            alpha = torch.softmax(score, dim=1)

            task_z = torch.sum(seed_z * alpha.unsqueeze(-1), dim=1)
            task_z = self.router_norm(task_z)

            out = self.task_heads[task](task_z)

            if self.task_num_classes[task] == 2:
                logits[task] = out.squeeze(-1)
            else:
                logits[task] = out

            if return_attn:
                router_attn[task] = alpha

        z_global = seed_z.mean(dim=1)

        if return_attn:
            return logits, z_global, router_attn

        return logits, z_global

class GenomeMultiClassifierGatedTaskRouter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_list,
        task_num_classes: dict,
        agg_mode: str = "pma",
        num_heads: int = 4,
        num_inds: int = 64,
        num_seeds: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        ln: bool = False,
        router_temperature: float = 1.5
    ):
        super().__init__()

        if agg_mode not in ["pma", "isab_pma"]:
            raise ValueError("Gated task router requires agg_mode='pma' or 'isab_pma'")

        if num_seeds < 2:
            raise ValueError("Gated task router requires num_seeds >= 2")

        self.task_list = list(task_list)
        self.task_num_classes = task_num_classes
        self.num_tasks = len(task_list)
        self.num_seeds = num_seeds
        self.hidden_dim = hidden_dim
        self.router_temperature = router_temperature

        self.aggregator = Aggregator(
            input_dim=input_dim,
            mode=agg_mode,
            num_heads=num_heads,
            num_inds=num_inds,
            num_seeds=num_seeds,
            ln=ln
        )

        self.seed_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.task_queries = nn.Parameter(torch.empty(self.num_tasks, hidden_dim))
        nn.init.xavier_uniform_(self.task_queries)

        self.task_gates = nn.Parameter(torch.zeros(self.num_tasks))

        self.router_norm = nn.LayerNorm(hidden_dim)

        self.task_heads = nn.ModuleDict()
        for task in self.task_list:
            n_cls = task_num_classes[task]
            if n_cls == 2:
                self.task_heads[task] = nn.Linear(hidden_dim, 1)
            else:
                self.task_heads[task] = nn.Linear(hidden_dim, n_cls)

    def forward(self, x, mask=None, return_attn=False):
        seed_emb = self.aggregator(x, mask)

        if seed_emb.dim() == 2:
            seed_emb = seed_emb.unsqueeze(1)

        seed_z = self.seed_projector(seed_emb)
        shared_z = seed_z.mean(dim=1)

        logits = {}
        router_attn = {}
        router_gates = {}
        task_embeddings = {}

        for t_idx, task in enumerate(self.task_list):
            q = self.task_queries[t_idx].view(1, 1, -1)

            score = (seed_z * q).sum(dim=-1) / math.sqrt(self.hidden_dim)
            alpha = torch.softmax(score / self.router_temperature, dim=1)

            routed_z = torch.sum(seed_z * alpha.unsqueeze(-1), dim=1)

            gate = torch.sigmoid(self.task_gates[t_idx])
            task_z = gate * routed_z + (1.0 - gate) * shared_z
            task_z = self.router_norm(task_z)

            out = self.task_heads[task](task_z)

            if self.task_num_classes[task] == 2:
                logits[task] = out.squeeze(-1)
            else:
                logits[task] = out

            if return_attn:
                router_attn[task] = alpha
                router_gates[task] = gate.detach()
                task_embeddings[task] = task_z

        z_global = shared_z

        if return_attn:
            aux = {
                "router_attn": router_attn,
                "router_gates": router_gates,
                "task_embeddings": task_embeddings,
                "seed_z": seed_z,
                "shared_z": shared_z
            }
            return logits, z_global, aux

        return logits, z_global

class GenomeMultiClassifierGatedTaskRouter2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_list,
        task_num_classes: dict,
        agg_mode: str = "pma",
        num_heads: int = 4,
        num_inds: int = 64,
        num_seeds: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        ln: bool = False,
        router_temperature: float = 0.5,
        seed_noise_std: float = 0.01,
        lambda_init: float = 0.8
    ):
        super().__init__()

        if agg_mode not in ["pma", "isab_pma"]:
            raise ValueError("Gated task router requires agg_mode='pma' or 'isab_pma'")

        if num_seeds < 2:
            raise ValueError("Gated task router requires num_seeds >= 2")

        self.task_list = list(task_list)
        self.task_num_classes = task_num_classes
        self.num_tasks = len(task_list)
        self.num_seeds = num_seeds
        self.hidden_dim = hidden_dim
        self.router_temperature = router_temperature
        self.seed_noise_std = seed_noise_std

        self.aggregator = Aggregator(
            input_dim=input_dim,
            mode=agg_mode,
            num_heads=num_heads,
            num_inds=num_inds,
            num_seeds=num_seeds,
            ln=ln
        )

        self.seed_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.router_norm = nn.LayerNorm(hidden_dim)

        self.task_queries = nn.Parameter(torch.empty(self.num_tasks, hidden_dim))
        nn.init.orthogonal_(self.task_queries)

        lambda_init = float(lambda_init)
        lambda_init = min(max(lambda_init, 1e-4), 1.0 - 1e-4)
        gate_logit = math.log(lambda_init / (1.0 - lambda_init))
        self.task_gates = nn.Parameter(torch.full((self.num_tasks,), gate_logit))

        self.task_heads = nn.ModuleDict()
        for task in self.task_list:
            n_cls = task_num_classes[task]
            if n_cls == 2:
                self.task_heads[task] = nn.Linear(hidden_dim, 1)
            else:
                self.task_heads[task] = nn.Linear(hidden_dim, n_cls)

    def compute_query_orth_loss(self):
        q = F.normalize(self.task_queries, dim=1)
        gram = q @ q.T
        eye = torch.eye(gram.size(0), device=gram.device)
        return ((gram - eye) ** 2).mean()

    def forward(self, x, mask=None, return_attn=False):
        seed_emb = self.aggregator(x, mask)

        if seed_emb.dim() == 2:
            seed_emb = seed_emb.unsqueeze(1)

        seed_z = self.seed_projector(seed_emb)

        if self.training and self.seed_noise_std > 0:
            seed_z = seed_z + torch.randn_like(seed_z) * self.seed_noise_std

        shared_z = seed_z.mean(dim=1)

        logits = {}
        router_attn = {}
        router_gates = {}
        task_embeddings = {}

        entropy_terms = []

        seed_z_norm = F.normalize(seed_z, dim=-1)
        task_queries_norm = F.normalize(self.task_queries, dim=-1)

        for t_idx, task in enumerate(self.task_list):
            q = task_queries_norm[t_idx].view(1, 1, -1)

            score = (seed_z_norm * q).sum(dim=-1)
            score = score / max(self.router_temperature, 1e-6)

            alpha = torch.softmax(score, dim=1)

            routed_z = torch.sum(seed_z * alpha.unsqueeze(-1), dim=1)

            gate = torch.sigmoid(self.task_gates[t_idx])
            task_z = gate * routed_z + (1.0 - gate) * shared_z
            task_z = self.router_norm(task_z)

            out = self.task_heads[task](task_z)

            if self.task_num_classes[task] == 2:
                logits[task] = out.squeeze(-1)
            else:
                logits[task] = out

            if return_attn:
                router_attn[task] = alpha
                router_gates[task] = gate.detach()
                task_embeddings[task] = task_z

                entropy = -(alpha * torch.log(alpha.clamp(min=1e-8))).sum(dim=1).mean()
                entropy_terms.append(entropy)

        z_global = shared_z

        if return_attn:
            if len(entropy_terms) > 0:
                router_entropy = torch.stack(entropy_terms).mean()
            else:
                router_entropy = torch.tensor(0.0, device=x.device)

            aux = {
                "router_attn": router_attn,
                "router_gates": router_gates,
                "task_embeddings": task_embeddings,
                "seed_z": seed_z,
                "shared_z": shared_z,
                "router_entropy": router_entropy,
                "query_orth_loss": self.compute_query_orth_loss(),
            }
            return logits, z_global, aux

        return logits, z_global
