import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Generates sinusoidal positional embeddings for the diffusion timestep 't'.
    This allows the model to know exactly at which noise level it currently operates.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ModulatedLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (adaLN).
    Dynamically predicts scale and shift based on the timestep conditioning.
    """
    def __init__(self, hidden_size: int, condition_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 2 * hidden_size)
        )
        # Zero-initialize the last layer for stable training at the start
        nn.init.zeros_(self.mlp[1].weight)
        nn.init.zeros_(self.mlp[1].bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        emb_out = self.mlp(condition)
        shift, scale = emb_out.chunk(2, dim=-1)
        return self.ln(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding for Transformer.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Seq_Len, Batch, Dim]
        x = x + self.pe[:x.size(0)]
        return x


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (max_period ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype = None):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


class PhonemeEncoderBlock(nn.Module):
    """
    A single Transformer Encoder block for phonemes, utilizing RoPE and FlashAttention.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor = None):
        h = self.norm1(x)
        B, L, C = h.shape

        qkv = self.qkv_proj(h).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to Text Queries and Keys
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        attn_mask = None
        if mask is not None:
            # Invert padding mask (True -> False) for SDPA
            attn_mask = ~mask.unsqueeze(1).unsqueeze(2)

        # FlashAttention-2
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.1 if self.training else 0.0
        )

        x = x + self.out_proj(attn_out.transpose(1, 2).reshape(B, L, C))
        x = x + self.ffn(self.norm2(x))
        return x


class PhonemeEncoder(nn.Module):
    """
    A lightweight Transformer Encoder to convert phoneme IDs into dense context embeddings.
    Uses Rotary Positional Embeddings (RoPE) to support arbitrary sequence lengths.
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 768, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim, padding_idx=0)
        self.rope = RotaryEmbedding(hidden_dim // num_heads)

        self.layers = nn.ModuleList([
            PhonemeEncoderBlock(hidden_dim=hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, phoneme_ids: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        :param phoneme_ids: [Batch, Seq_Len] of integers
        :param src_key_padding_mask: [Batch, Seq_Len] boolean mask (True for padding)
        :return: [Batch, Seq_Len, Hidden_Dim] dense embeddings
        """
        x = self.embedding(phoneme_ids)  # [B, S, H]

        # Generate RoPE frequencies for the text sequence length
        cos, sin = self.rope(x.shape[1], x.device)

        for layer in self.layers:
            x = layer(x, cos, sin, mask=src_key_padding_mask)

        x = self.final_norm(x)
        return x