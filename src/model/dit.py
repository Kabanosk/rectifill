import math

import torch
import torch.nn as nn

from src.config.config import ModelConfig
from src.model.modules import SinusoidalPositionEmbeddings, ModulatedLayerNorm


class DiTBlock(nn.Module):
    """
    A single block of the Diffusion Transformer.
    Incorporates Self-Attention for audio context and Cross-Attention for text conditioning.
    """

    def __init__(self, hidden_size: int, num_heads: int, text_dim: int, cond_dim: int, dropout: float):
        """
        :param hidden_size: The feature dimension of the audio patches.
        :param num_heads: Number of attention heads for MultiheadAttention.
        :param text_dim: The feature dimension of the incoming text embeddings.
        :param cond_dim: The dimension of the conditioning tensor (timestep embedding).
        :param dropout: Dropout probability for regularization.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.norm1 = ModulatedLayerNorm(hidden_size, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout)

        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True,
                                                dropout=dropout)

        self.norm3 = ModulatedLayerNorm(hidden_size, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, text_emb: torch.Tensor,
                text_mask: torch.Tensor = None, mel_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Processes the input through self-attention, cross-attention, and FFN.

        :param x: Audio patches tensor of shape [Batch, Seq_Len, Hidden_Size].
        :param cond: Timestep condition tensor of shape [Batch, Cond_Dim].
        :param text_emb: Text embeddings tensor of shape [Batch, Text_Seq_Len, Text_Dim].
        :param text_mask: Optional boolean mask for text embeddings.
        :param mel_pad_mask: Optional boolean mask for mel-spectrogram padding.

        :return: Processed tensor of shape [Batch, Seq_Len, Hidden_Size].
        """
        # --- Self Attention ---
        h = self.norm1(x, cond)
        attn_out, _ = self.attn(query=h, key=h, value=h, key_padding_mask=mel_pad_mask)
        x = x + attn_out

        # --- Cross Attention ---
        text_emb_proj = self.text_proj(text_emb)
        h = self.norm2(x)
        cross_out, _ = self.cross_attn(query=h, key=text_emb_proj, value=text_emb_proj, key_padding_mask=text_mask)
        x = x + cross_out

        # --- Feed Forward ---
        h = self.norm3(x, cond)
        ffn_out = self.ffn(h)
        x = x + ffn_out

        return x


class DiTModel(nn.Module):
    """
    Main Diffusion Transformer architecture for Audio Inpainting.
    Predicts the velocity field for Rectified Flow Matching.
    """

    def __init__(self, config: ModelConfig):
        """
        :param config: ModelConfig object containing all hyperparameters.
        """
        super().__init__()
        self.config = config

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )

        in_channels = config.mel_bins + 1
        self.input_proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=5,
            padding=2
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.input_dropout = nn.Dropout(config.dropout)
        self.null_text_embed = nn.Parameter(torch.randn(1, 1, config.text_dim) * 0.02)

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                text_dim=config.text_dim,
                cond_dim=config.hidden_size,
                dropout=config.dropout
            ) for _ in range(config.depth)
        ])

        self.output_proj = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.mel_bins, kernel_size=1)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, xt: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass predicting the velocity field v = x1 - x0.

        :param xt: Noisy mel-spectrogram of shape [Batch, Mel_Bins, Time].
        :param mask: Inpainting mask of shape (1 indicates missing regions) [Batch, 1, Time].
        :param kwargs: kwargs takes other variables that are not presented in BaseModel class like:
            - t: Diffusion timestep of shape [Batch].
            - text_emb: Text embeddings of shape [Batch, Text_Seq_Len, Text_Dim].
            - text_mask: Optional boolean mask for text embeddings of shape [Batch, Text_Seq_Len].
        :return: Predicted velocity field of shape [Batch, Mel_Bins, Time].
        """
        t = kwargs["t"]
        text_emb = kwargs.get("text_emb", None)
        text_mask = kwargs.get("text_mask", None)
        mel_pad_mask = kwargs.get("mel_pad_mask", None)
        cfg_drop_mask = kwargs.get("cfg_drop_mask", None)

        if text_emb is not None and cfg_drop_mask is not None:
            null_emb = self.null_text_embed.expand(text_emb.shape[0], text_emb.shape[1], -1)
            text_emb = torch.where(cfg_drop_mask, null_emb, text_emb)

        t_emb = self.time_mlp(t * 1000.0)  # [Batch, Hidden_Size]
        x = torch.cat([xt, mask], dim=1)  # [Batch, Mel_Bins + 1, Time]

        # Project to hidden dimensions
        x = self.input_proj(x)  # [Batch, Hidden_Size, Time]
        x = x.transpose(1, 2)  # [Batch, Time, Hidden_Size]

        # Add positional embeddings
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.input_dropout(x)

        for block in self.blocks:
            x = block(x, cond=t_emb, text_emb=text_emb, text_mask=text_mask, mel_pad_mask=mel_pad_mask)

        x = x.transpose(1, 2)  # [Batch, Hidden_Size, Time]

        velocity = self.output_proj(x)  # [Batch, Mel_Bins, Time]
        return velocity
