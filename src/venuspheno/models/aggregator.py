import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import math

# Set transformer modules
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, return_attn=False):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if return_attn:
            B = Q.size(0)
            A = A.view(self.num_heads, B, A.size(1), A.size(2))
            A = A.permute(1, 0, 2, 3)  # (B, heads, Q_len, K_len)
            return O, A

        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, return_attn=False):
        if return_attn:
            H, attn0 = self.mab0(
                self.I.repeat(X.size(0), 1, 1),
                X,
                return_attn=True
            )
            Z, attn1 = self.mab1(
                X,
                H,
                return_attn=True
            )
            return Z, {
                "induce_to_gene": attn0,
                "gene_to_induce": attn1
            }

        # 默认路径（训练 / 普通推理）
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        Z = self.mab1(X, H)
        return Z


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    # def forward(self, X):
    #     return self.mab(self.S.repeat(X.size(0), 1, 1), X)
    def forward(self, X, return_attn=False):
        Q = self.S.repeat(X.size(0), 1, 1)
        if return_attn:
            Z, attn = self.mab(Q, X, return_attn=True)
            return Z, attn
        Z = self.mab(Q, X, return_attn=False)
        return Z



class Aggregator(nn.Module):
    """
    通用 Aggregator 模块，支持多种聚合方式：
    mean / max / attention / PMA / ISAB_PMA / ISAB_mean
    """
    def __init__(self, input_dim, mode="mean", num_heads=4, 
                 num_inds=128, num_seeds=1, ln=False):
        super().__init__()
        self.mode = mode.lower()
        self.input_dim = input_dim

        if self.mode == "attention":
            self.V = nn.Linear(input_dim, num_inds)
            self.w = nn.Linear(num_inds, 1)

        elif self.mode == "pma":
            self.num_seeds = num_seeds
            self.pma = PMA(input_dim, num_heads=num_heads, num_seeds=num_seeds, ln=ln)
        
        elif self.mode == "isab_mean":
            self.isab = ISAB(input_dim, input_dim, num_heads=num_heads, num_inds=num_inds, ln=ln)

        elif self.mode == "isab_pma":
            self.num_seeds = num_seeds
            self.isab = ISAB(input_dim, input_dim, num_heads=num_heads, num_inds=num_inds, ln=ln)
            self.pma = PMA(input_dim, num_heads=num_heads, num_seeds=num_seeds, ln=ln)

    def forward(self, x, mask=None, return_attn=False):
        mode = self.mode

        if mode == "mean":
            if mask is not None:
                mask = mask.unsqueeze(-1)
                x = x * mask
                out = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            else:
                out = x.mean(dim=1)
            return out

        elif mode == "max":
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
            out, _ = x.max(dim=1)
            return out

        elif mode == "attention":
            attn = torch.tanh(self.V(x))
            attn = self.w(attn).squeeze(-1)
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float("-inf"))
            alpha = F.softmax(attn, dim=1)
            out = torch.bmm(alpha.unsqueeze(1), x).squeeze(1)
            return out

        elif mode == "pma":
            if return_attn:
                out, attn = self.pma(x, return_attn=True)
                if self.num_seeds == 1:
                    out = out.squeeze(1)
                return out, attn

            out = self.pma(x)
            if self.num_seeds == 1:
                return out.squeeze(1)
            return out

        elif mode == "isab_mean":
            z = self.isab(x)
            if mask is not None:
                mask = mask.unsqueeze(-1)
                z = z * mask
                out = z.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            else:
                out = z.mean(dim=1)
            return out

        elif mode == "isab_pma":
            if return_attn:
                z, isab_attn = self.isab(x, return_attn=True)
                out, pma_attn = self.pma(z, return_attn=True)

                if self.num_seeds == 1:
                    out = out.squeeze(1)

                return out, {
                    "isab": isab_attn,
                    "pma": pma_attn
                }

            z = self.isab(x)
            out = self.pma(z)
            if self.num_seeds == 1:
                return out.squeeze(1)
            return out

        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")
