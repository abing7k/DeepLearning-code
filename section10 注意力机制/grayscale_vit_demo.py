
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_2d_sincos_pos_embed(h, w, dim, device):
    grid_y, grid_x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    dim_half = dim // 2
    assert dim % 2 == 0
    dim_y = dim_half
    dim_x = dim - dim_y
    def pe_1d(pos, d):
        omega = torch.arange(d // 2, device=device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (d // 2)))
        out = torch.einsum('hw, d -> hwd', pos, omega)
        sin = torch.sin(out)
        cos = torch.cos(out)
        return torch.cat([sin, cos], dim=-1)
    grid_y, grid_x = grid_y.float(), grid_x.float()
    pe_y = pe_1d(grid_y, dim_y)
    pe_x = pe_1d(grid_x, dim_x)
    pe = torch.cat([pe_y, pe_x], dim=-1).view(1, h*w, dim)
    return pe

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=1, patch=8, d_model=64):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, d_model, kernel_size=patch, stride=patch)
    def forward(self, x):
        x = self.proj(x)
        B, C, H_, W_ = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, (H_, W_)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.scale = head_dim ** -0.5
    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return out, attn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(self.head_dim)
        self.out = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, N, D = x.shape
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        def split_heads(t):
            return t.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split_heads(Q), split_heads(K), split_heads(V)
        out, attn = self.attn(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out(out)
        return out, attn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )
    def forward(self, x):
        y, attn = self.mha(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

@dataclass
class TinyGrayViTConfig:
    img_size: int = 32
    patch: int = 8
    d_model: int = 64
    n_heads: int = 4
    depth: int = 2

class TinyGrayViT(nn.Module):
    def __init__(self, cfg: TinyGrayViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(in_ch=1, patch=cfg.patch, d_model=cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg.d_model, cfg.n_heads) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.d_model)
    def forward(self, x):
        tok, (H_, W_) = self.patch_embed(x)
        pos = get_2d_sincos_pos_embed(H_, W_, self.cfg.d_model, x.device)
        tok = tok + pos
        attn_maps = []
        for blk in self.blocks:
            tok, attn = blk(tok)
            attn_maps.append(attn)
        tok = self.norm(tok)
        return tok, attn_maps, (H_, W_)

def demo():
    device = 'cpu'
    cfg = TinyGrayViTConfig()
    model = TinyGrayViT(cfg).to(device)
    x = torch.randn(1, 1, cfg.img_size, cfg.img_size, device=device)
    with torch.no_grad():
        out, attn_maps, (Hp, Wp) = model(x)
    print("Input:", list(x.shape), "Grid:", Hp, "x", Wp, "Tokens:", Hp*Wp)
    print("Final tokens:", list(out.shape))
    for i, a in enumerate(attn_maps):
        print(f"Block {i} attn:", list(a.shape))

if __name__ == "__main__":
    demo()
