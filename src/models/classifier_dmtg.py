"""DMTG head for VenusPheno's shared Set Transformer backbone."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aggregator import Aggregator
from models.classifier import SharedResidualMLP


class GenomeMultiClassifierDMTG(nn.Module):
    """One-shot differentiable task grouping after a shared PMA representation.

    Each task owns one prediction head per group.  Training optimizes the
    Gumbel-Softmax-weighted branch losses; evaluation selects one group per
    task with the learned categorical argmax.
    """

    def __init__(
        self,
        input_dim,
        task_list,
        task_num_classes,
        agg_mode="isab_pma",
        num_heads=4,
        num_inds=64,
        num_seeds=4,
        hidden_dim=512,
        dropout=0.3,
        ln=False,
        num_groups=3,
        temperature=1.0,
        seed_mab=False,
        hsp_out_tokens=4,
        hsp_rank=4,
    ):
        super().__init__()
        if agg_mode not in {"pma", "isab_pma"} or num_seeds < 2:
            raise ValueError("DMTG requires multi-seed PMA or ISAB+PMA")
        if num_groups < 2:
            raise ValueError("DMTG requires at least two groups")

        self.task_list = list(task_list)
        self.task_num_classes = task_num_classes
        self.num_groups = num_groups
        self.temperature = temperature
        self.train_group_mode = "learned"
        self.eval_group_mode = "hard"
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
        flattened_dim = input_dim * num_seeds
        self.group_branches = nn.ModuleList(
            [SharedResidualMLP(flattened_dim, hidden_dim, dropout) for _ in range(num_groups)]
        )
        self.group_heads = nn.ModuleList(
            [nn.ModuleDict({
                task: nn.Linear(hidden_dim, 1 if task_num_classes[task] == 2 else task_num_classes[task])
                for task in self.task_list
            }) for _ in range(num_groups)]
        )
        # Small noise breaks otherwise exact symmetry between initially equal groups.
        self.group_logits = nn.Parameter(torch.ones(len(self.task_list), num_groups) + 1e-3 * torch.randn(len(self.task_list), num_groups))

    def group_weights(self):
        if self.training:
            if self.train_group_mode == "uniform":
                return torch.full_like(self.group_logits, 1.0 / self.num_groups)
            return F.gumbel_softmax(self.group_logits, tau=self.temperature, hard=False, dim=-1)
        if self.eval_group_mode == "soft":
            return torch.softmax(self.group_logits, dim=-1)
        if self.eval_group_mode == "uniform":
            return torch.full_like(self.group_logits, 1.0 / self.num_groups)
        assignments = self.group_logits.argmax(dim=-1)
        return F.one_hot(assignments, num_classes=self.num_groups).to(self.group_logits.dtype)

    def set_eval_group_mode(self, mode):
        if mode not in {"hard", "soft", "uniform"}:
            raise ValueError(f"Unknown DMTG eval group mode: {mode}")
        self.eval_group_mode = mode

    def set_train_group_mode(self, mode):
        if mode not in {"learned", "uniform"}:
            raise ValueError(f"Unknown DMTG train group mode: {mode}")
        self.train_group_mode = mode

    def group_assignments(self):
        return self.group_logits.argmax(dim=-1)

    def selected_head_weight(self, task):
        task_idx = self.task_list.index(task)
        group_idx = int(self.group_assignments()[task_idx])
        return self.group_heads[group_idx][task].weight

    def forward(self, x, mask=None, return_branch_logits=False, return_seed_emb=False):
        if return_seed_emb:
            seed_emb, raw_seed_emb = self.aggregator(x, mask, return_seed_emb=True)
        else:
            seed_emb = self.aggregator(x, mask)
            raw_seed_emb = seed_emb
        if seed_emb.dim() != 3:
            raise ValueError(f"DMTG expected PMA output (B, S, D), got {tuple(seed_emb.shape)}")

        flat_emb = seed_emb.reshape(seed_emb.size(0), -1)
        branch_features = [branch(flat_emb) for branch in self.group_branches]
        weights = self.group_weights()
        branch_logits = {}
        logits = {}
        for task_idx, task in enumerate(self.task_list):
            outputs = [heads[task](features) for heads, features in zip(self.group_heads, branch_features)]
            if self.task_num_classes[task] == 2:
                outputs = [output.squeeze(-1) for output in outputs]
            branch_logits[task] = outputs
            logits[task] = sum(weights[task_idx, group_idx] * output for group_idx, output in enumerate(outputs))

        global_feature = torch.stack(branch_features, dim=0).mean(dim=0)
        if return_branch_logits:
            return logits, global_feature, {
                "branch_logits": branch_logits,
                "group_weights": weights,
                "seed_emb": raw_seed_emb,
            }
        if return_seed_emb:
            return logits, global_feature, raw_seed_emb
        return logits, global_feature
