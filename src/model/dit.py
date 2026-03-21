import math
import torch
import torch.nn as nn

from src.config.config import ModelConfig
from src.model.base import BaseModel

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Generates sinusoidal positional embeddings for the diffusion timestep 't'.
    This allows the model to know exactly at which noise level it currently operates.
    """

    def __init__(self, dim: int):
        """
        :param dim: The hidden dimension size of the embeddings.
        """
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Computes the sinusoidal embeddings for the given timesteps.

        :param time: Tensor of shape [Batch_Size] containing diffusion timesteps.
        :return: Tensor of shape [Batch_Size, Dim] with sinusoidal embeddings.
        """
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
    Instead of learning static affine parameters, it dynamically predicts
    the scale and shift based on the diffusion timestep embedding.
    """

    def __init__(self, hidden_size: int, condition_dim: int):
        """
        :param hidden_size: The feature dimension of the input tensor (audio patches).
        :param condition_dim: The dimension of the conditioning tensor (timestep embedding).
        """
        super().__init__()
        self.ln = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 2 * hidden_size)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Applies modulated layer normalization to the input tensor.

        :param x: Input tensor of shape [Batch, Seq_Len, Hidden_Size].
        :param condition: Timestep embedding of shape [Batch, Condition_Dim].
        :return: Normalized and modulated tensor of shape [Batch, Seq_Len, Hidden_Size].
        """
        emb_out = self.mlp(condition)
        shift, scale = emb_out.chunk(2, dim=-1)

        return self.ln(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
                text_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Processes the input through self-attention, cross-attention, and FFN.

        :param x: Audio patches tensor of shape [Batch, Seq_Len, Hidden_Size].
        :param cond: Timestep condition tensor of shape [Batch, Cond_Dim].
        :param text_emb: Text embeddings tensor of shape [Batch, Text_Seq_Len, Text_Dim].
        :param text_mask: Optional boolean mask for text embeddings.
        :return: Processed tensor of shape [Batch, Seq_Len, Hidden_Size].
        """
        # --- Self Attention ---
        h = self.norm1(x, cond)
        attn_out, _ = self.attn(query=h, key=h, value=h)
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


class DiTModel(BaseModel):
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
        self.input_proj = nn.Conv1d(in_channels=in_channels, out_channels=config.hidden_size, kernel_size=1)

        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.hidden_size))

        self.input_dropout = nn.Dropout(config.dropout)

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
        t = kwargs.get("t", 0)
        text_emb = kwargs.get("text_emb", None)
        text_mask = kwargs.get("text_mask", None)

        # --- Timestep Embedding ---
        t_emb = self.time_mlp(t)  # [Batch, Hidden_Size]

        # --- Input Projection ---
        x = torch.cat([xt, mask], dim=1)  # [Batch, Mel_Bins + 1, Time]

        # Project to hidden dimensions
        x = self.input_proj(x)  # [Batch, Hidden_Size, Time]
        x = x.transpose(1, 2)  # [Batch, Time, Hidden_Size]

        # Add positional embeddings
        seq_len = x.shape[1]
        x = x + self.pos_embed[:, :seq_len, :]
        x = self.input_dropout(x)

        # --- Process through DiT Blocks ---
        for block in self.blocks:
            x = block(x, cond=t_emb, text_emb=text_emb, text_mask=text_mask)

        # --- Output Projection ---
        x = x.transpose(1, 2)  # [Batch, Hidden_Size, Time]

        # Predict velocity field v
        velocity = self.output_proj(x)  # [Batch, Mel_Bins, Time]

        return velocity

    def configure_optimizers(self, **kwargs) -> torch.optim.Optimizer:
        """
        Implementation of the abstract method from BaseModel.
        """
        return torch.optim.AdamW(self.parameters(), **kwargs)
