import math

import torch
import torch.nn as nn


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


class PhonemeEncoder(nn.Module):
    """
    A lightweight Transformer Encoder to convert phoneme IDs into dense context embeddings.
    Matches the dimension expected by DiT Cross-Attention (e.g., 768).
    """
    def __init__(self, vocab_size: int, hidden_dim: int = 768, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, phoneme_ids: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        :param phoneme_ids: [Batch, Seq_Len] of integers
        :param src_key_padding_mask: [Batch, Seq_Len] boolean mask (True for padding)
        :return: [Batch, Seq_Len, Hidden_Dim] dense embeddings
        """
        x = self.embedding(phoneme_ids)  # [B, S, H]

        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        encoded_phonemes = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        return encoded_phonemes
