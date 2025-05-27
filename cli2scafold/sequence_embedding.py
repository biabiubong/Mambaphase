import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_ssm, dt_rank, dropout=0.2):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_inner)
        self.conv1d = nn.Conv1d(d_inner, d_inner, 3, padding=1, groups=d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.A_log = nn.Parameter(torch.randn(d_inner, n_ssm))
        self.D = nn.Parameter(torch.randn(d_inner))
        self.x_proj = nn.Linear(d_inner, dt_rank + n_ssm)
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

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.query_key_value = nn.Linear(d_model, 3 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, l, d = x.shape
        qkv = self.query_key_value(x).view(b, l, self.n_heads, 3 * self.d_k).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, l, d)
        output = self.out_proj(attn_output)
        return output

class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_inner, n_ssm, dt_rank, n_heads, dropout=0.2):
        super().__init__()
        self.mamba_blocks = nn.ModuleList([MambaBlock(d_model, d_inner, n_ssm, dt_rank, dropout) for _ in range(2)])
        self.attention = Attention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mamba_outputs = []
        for mamba_block in self.mamba_blocks:
            mamba_outputs.append(mamba_block(self.norm1(x)))
        x = torch.stack(mamba_outputs, dim=0).sum(dim=0) / len(self.mamba_blocks)
        x = self.dropout(x)
        x = self.dropout(self.attention(self.norm2(x)) + x)
        return x

class SequenceEmbedding(nn.Module):
    def __init__(self, d_model, d_inner, n_ssm, dt_rank, vocab_size, n_layer, n_heads, dropout=0.2, output_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([ResidualBlock(d_model, d_inner, n_ssm, dt_rank, n_heads, dropout) for _ in range(n_layer)])
        self.norm_f = nn.LayerNorm(d_model)
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
    def __init__(self, d_model, d_inner, n_ssm, dt_rank, vocab_size, n_layer, n_heads, output_dim=512, dropout=0.2):
        super().__init__()
        self.embedding_model = SequenceEmbedding(d_model, d_inner, n_ssm, dt_rank, vocab_size, n_layer, n_heads, dropout, output_dim)

    def forward(self, input_ids):
        embeddings = self.embedding_model(input_ids)
        return embeddings

def create_sequence_to_vector_model(vocab_size, d_model=64, d_inner=64, n_ssm=4, dt_rank=1, n_layer=2, n_heads=2, dropout=0.2, output_dim=512):
    """
    初始化一个 SequenceToVector 模型，降低了参数量。
    
    参数:
    - vocab_size: 词汇大小
    - d_model: 输入和输出的嵌入维度
    - d_inner: MambaBlock 中间层的维度，已降低
    - n_ssm: MambaBlock 中 SSM 参数的数量，已降低
    - dt_rank: MambaBlock 中 delta 投影的秩，已降低
    - n_layer: 模型中的残差块数量
    - n_heads: 注意力头的数量，已降低
    - dropout: Dropout 比率
    - output_dim: 模型的最终输出维度
    
    返回:
    - 一个 SequenceToVectorModel 实例
    """
    return SequenceToVectorModel(d_model, d_inner, n_ssm, dt_rank, vocab_size, n_layer, n_heads, output_dim, dropout)