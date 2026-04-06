import math
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class FastGQA(nn.Module):
    """
    Vectorized Grouped Multi-Query Attention.

    - d_model: input/output dim
    - d_V: head dim
    - n_groups * n_queries = total heads
    - K,V shared across all heads (multi-query flavor)
    """
    def __init__(self,
                 d_model: int = 512,
                 d_V: int = 64,
                 n_groups: int = 4,
                 n_queries: int = 8,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 use_sdpa: bool = True) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_V = d_V
        self.n_groups = n_groups
        self.n_queries = n_queries

        self.n_heads = n_groups * n_queries
        self.head_dim = d_V
        self.inner_dim = self.n_heads * self.head_dim

        # One Q projection for all heads, one shared K,V projection
        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=True)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=True)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=True)

        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

        self.attn_dropout_p = float(attn_dropout) if attn_dropout else 0.0
        self.attn_dropout = nn.Dropout(self.attn_dropout_p) if self.attn_dropout_p > 0 else nn.Identity()
        self.proj_dropout = nn.Dropout(proj_dropout) if proj_dropout and proj_dropout > 0 else nn.Identity()

        self.use_sdpa = bool(use_sdpa)

        # scalar, not a Tensor, so no per-forward tensor creation
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: t.Tensor, mask: t.Tensor | None = None) -> t.Tensor:
        """
        x: [B, N, d_model]
        mask: optional [B, N, N] or [B, 1, N, N] (1 = keep, 0 = mask)
        returns: [B, N, d_model]
        """
        B, N, D = x.shape
        assert D == self.d_model

        # Q: [B, N, H*dv] -> [B, H, N, dv]
        q = self.q_proj(x)
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, N, dv]

        # K,V shared across heads: [B, N, dv] -> [B, 1, N, dv]
        k = self.k_proj(x).unsqueeze(1)  # [B, 1, N, dv]
        v = self.v_proj(x).unsqueeze(1)  # [B, 1, N, dv]

        # Prefer SDPA (FlashAttention / memory-efficient attention) when available.
        # We expand shared K,V across heads with a view (no copy) for SDPA compatibility.
        if self.use_sdpa and hasattr(F, 'scaled_dot_product_attention'):
            k_sdpa = k.expand(B, self.n_heads, N, self.head_dim)
            v_sdpa = v.expand(B, self.n_heads, N, self.head_dim)

            attn_mask = None
            if mask is not None:
                # allow [B, N, N] or [B, 1, N, N]; interpret 1=keep, 0=mask
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                if mask.dtype != t.bool:
                    mask = mask != 0
                # SDPA boolean mask: True means "mask out".
                attn_mask = ~mask

            dropout_p = self.attn_dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            # scores: [B, H, N, N]
            scores = t.matmul(q, k.transpose(-1, -2)) * self.scale

            if mask is not None:
                # allow [B, N, N] or [B, 1, N, N]
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)  # [B, 1, N, N]
                keep = mask if mask.dtype == t.bool else (mask != 0)
                scores = scores.masked_fill(~keep, t.finfo(scores.dtype).min)

            attn = scores.softmax(dim=-1)     # [B, H, N, N]
            attn = self.attn_dropout(attn)
            out = t.matmul(attn, v)          # [B, H, N, dv]

        # back to [B, N, H*dv]
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_dim)
        out = self.out_proj(out)         # [B, N, d_model]
        out = self.proj_dropout(out)
        return out


class GQASubnet1D(nn.Module):
    """
    Faster GQA-based subnet for AllInOneBlock in 1D (vector) mode.

    Expects:
        x: [B, dims_in]
    Returns:
        y: [B, dims_out]
    """
    def __init__(self, dims_in: int, dims_out: int,
                 num_tokens: int = 8,
                 d_V: int = 64,
                 n_groups: int = 4,
                 n_queries: int = 8,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 mlp_ratio: float = 2.0,
                 zero_init: bool = True):
        super().__init__()

        # Hidden size for internal representation
        hidden_dim = max(dims_in, dims_out)
        if hidden_dim % num_tokens != 0:
            hidden_dim = ((hidden_dim + num_tokens - 1) // num_tokens) * num_tokens

        self.dims_in = dims_in
        self.dims_out = dims_out
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.d_model = hidden_dim // num_tokens  # per-token dim

        # Project from flow space -> attention hidden
        self.in_proj = nn.Linear(dims_in, hidden_dim)

        # Pre-norm improves stability when used inside many flow coupling blocks
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        # Fast grouped multi-query attention over [B, num_tokens, d_model]
        self.gqa = FastGQA(
            d_model=self.d_model,
            d_V=d_V,
            n_groups=n_groups,
            n_queries=n_queries,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            use_sdpa=True,
        )

        self.drop = nn.Dropout(proj_dropout) if proj_dropout and proj_dropout > 0 else nn.Identity()

        # Small per-token MLP (Transformer-style)
        mlp_hidden = max(1, int(self.d_model * float(mlp_ratio)))
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(proj_dropout) if proj_dropout and proj_dropout > 0 else nn.Identity(),
            nn.Linear(mlp_hidden, self.d_model),
            nn.Dropout(proj_dropout) if proj_dropout and proj_dropout > 0 else nn.Identity(),
        )

        # Back to flow dim
        self.out_proj = nn.Linear(hidden_dim, dims_out)

        # For flow coupling stability: start near-identity by making subnet output ~0 initially.
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: [B, dims_in]
        """
        B, D = x.shape
        assert D == self.dims_in, f"Expected {self.dims_in}, got {D}"

        # [B, dims_in] -> [B, hidden_dim]
        h = self.in_proj(x)                    # [B, hidden_dim]

        # -> [B, num_tokens, d_model]
        h = h.view(B, self.num_tokens, self.d_model)

        # Attention block (pre-norm + residual)
        h = h + self.drop(self.gqa(self.norm1(h)))

        # MLP block (pre-norm + residual)
        h = h + self.mlp(self.norm2(h))

        # Flatten back
        h = h.reshape(B, self.hidden_dim)      # [B, hidden_dim]

        # Final projection
        y = self.out_proj(h)                   # [B, dims_out]
        return y

def subnet_gqa_1d(dims_in, dims_out):
    # dims_in and dims_out are ints in your AllInOneBlock config
    return GQASubnet1D(
        dims_in=dims_in,
        dims_out=dims_out,
        num_tokens=8,     # tune this: 4, 8, 16 (smaller = faster)
        d_V=64,
        n_groups=4,
        n_queries=8,
        attn_dropout=0.0,
        proj_dropout=0.0,
        mlp_ratio=2.0,
        zero_init=True,
    )
