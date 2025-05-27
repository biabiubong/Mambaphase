import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_ssm, dt_rank, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_inner)
        self.conv1d = nn.Conv1d(d_inner, d_inner, 3, padding=1, groups=d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.A_log = nn.Parameter(torch.randn(d_inner, n_ssm // 2))
        self.D = nn.Parameter(torch.randn(d_inner))
        self.x_proj = nn.Linear(d_inner, dt_rank + n_ssm // 2)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        self.dropout = nn.Dropout(dropout)
    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        total_size = self.x_proj.out_features
        expected_size = total_size - n
        (delta, B) = x_dbl.split(split_size=[expected_size, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, D)
        return y
    def selective_scan(self, u, delta, A, B, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            ys.append(x.sum(dim=-1))
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y
    def forward(self, x):
        (b, l, d) = x.shape
        x = self.in_proj(x)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        x = self.dropout(x)
        y = self.ssm(x)
        y = self.dropout(y)
        output = self.out_proj(y)
        return output

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_ssm, dt_rank, dropout=0.1):
        super().__init__()
        # 只留一个MambaBlock
        self.mamba_block = MambaBlock(d_model, d_inner, n_ssm, dt_rank, dropout)
        self.norm1 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.mamba_block(self.norm1(x))
        x = self.dropout(x)
        return x

class SequenceEmbedding(nn.Module):
    def __init__(self, d_model, d_inner, n_ssm, dt_rank, vocab_size, n_layer, dropout=0.1, output_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            ResidualBlock(d_model, d_inner, n_ssm, dt_rank, dropout)
            for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(d_model, output_dim)
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        x = self.dropout(x)
        x = x.mean(dim=1)
        x = self.output_proj(x)
        return x

class SequenceToVectorModel(nn.Module):
    def __init__(self, d_model, d_inner, n_ssm, dt_rank, vocab_size, n_layer, output_dim=128, dropout=0.1):
        super().__init__()
        self.embedding_model = SequenceEmbedding(d_model, d_inner, n_ssm, dt_rank, vocab_size, n_layer, dropout, output_dim)
    def forward(self, input_ids):
        embeddings = self.embedding_model(input_ids)
        return embeddings