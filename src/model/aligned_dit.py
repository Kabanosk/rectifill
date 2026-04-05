import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from loguru import logger

from src.config.config import ModelConfig
from src.model.modules import SinusoidalPositionEmbeddings, ModulatedLayerNorm


class LengthRegulator(nn.Module):
    """
    Expands text embeddings according to target durations.
    """

    def forward(self, text_emb: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """
        :param text_emb: Text embeddings of shape [Batch, Seq_Len, Dim].
        :param durations: Durations of shape [Batch, Seq_Len].
        :return: Expanded embeddings of shape [Batch, Sum(Durations), Dim].
        """
        expanded_embeddings = []
        for i in range(text_emb.shape[0]):
            emb_i = text_emb[i]  # [Seq_Len, Dim]
            dur_i = durations[i]  # [Seq_Len]

            expanded_i = torch.repeat_interleave(emb_i, dur_i, dim=0)  # [Total_Mel_Frames, Dim], repeat each token dur times
            expanded_embeddings.append(expanded_i)

        expanded_embeddings = pad_sequence(expanded_embeddings, batch_first=True, padding_value=0.0)
        return expanded_embeddings


class AlignedDiTBlock(nn.Module):
    """
    Simplified DiT Block. Since Text and Audio are already combined (Early Fusion), we only need Self-Attention.
    """

    def __init__(self, hidden_size: int, num_heads: int, cond_dim: int, dropout: float):
        super().__init__()
        self.norm1 = ModulatedLayerNorm(hidden_size, cond_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout)

        self.norm2 = ModulatedLayerNorm(hidden_size, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mel_pad_mask: torch.Tensor = None) -> torch.Tensor:
        h = self.norm1(x, cond)
        attn_out, _ = self.attn(query=h, key=h, value=h, key_padding_mask=mel_pad_mask)
        x = x + attn_out

        h = self.norm2(x, cond)
        ffn_out = self.ffn(h)
        x = x + ffn_out

        return x


class AlignedDiTModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        logger.info("Initializing AlignedDiTModel with Early Fusion...")

        # Timestep conditioning
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )

        # Audio processing
        in_channels = config.mel_bins + 1
        self.input_proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=5,
            padding=2
        )

        # Text Processing
        self.text_proj = nn.Linear(config.text_dim, config.hidden_size)
        self.length_regulator = LengthRegulator()
        self.fusion_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.null_text_embed = nn.Parameter(torch.randn(1, 1, config.text_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.input_dropout = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            AlignedDiTBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                cond_dim=config.hidden_size,
                dropout=config.dropout
            ) for _ in range(config.depth)
        ])

        self.output_proj = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.mel_bins, kernel_size=1)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, xt: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.
        xt: [B, 128, T]
        mask: [B, 1, T]
        kwargs MUST contain 't', 'text_emb', and 'durations'
        """
        t = kwargs["t"]
        text_emb = kwargs.get("text_emb")  # [B, Text_Seq_Len, 768]
        durations = kwargs.get("durations")  # [B, Text_Seq_Len]
        mel_pad_mask = kwargs.get("mel_pad_mask")  # [B, T]
        cfg_drop_mask = kwargs.get("cfg_drop_mask", None)

        if text_emb is not None and durations is not None:
            raise ValueError("AlignedDiT requires text_emb and durations!")

        if cfg_drop_mask is not None:
            null_emb = self.null_text_embed.expand(text_emb.shape[0], text_emb.shape[1], -1)
            text_emb = torch.where(cfg_drop_mask, null_emb, text_emb)

        t_emb = self.time_mlp(t * 1000.0)
        x_audio = torch.cat([xt, mask], dim=1)
        x_audio = self.input_proj(x_audio)  # [B, Hidden_Size, T]
        x_audio = x_audio.transpose(1, 2)  # [B, T, Hidden_Size]
        audio_time = x_audio.shape[1]

        # --- TEXT & ALIGNMENT ---
        x_text = self.text_proj(text_emb)  # [B, Text_Seq_Len, Hidden_Size]
        x_text_aligned = self.length_regulator(x_text, durations)  # [B, T_text, Hidden]

        if x_text_aligned.shape[1] > audio_time:
            x_text_aligned = x_text_aligned[:, :audio_time, :]
        elif x_text_aligned.shape[1] < audio_time:
            pad_amount = audio_time - x_text_aligned.shape[1]
            x_text_aligned = torch.nn.functional.pad(x_text_aligned, (0, 0, 0, pad_amount))

        # --- EARLY FUSION ---
        x_mixed = torch.cat([x_audio, x_text_aligned], dim=-1)
        x = self.fusion_proj(x_mixed)  # [B, T, Hidden_Size]

        # --- TRANSFORMER ---
        x = x + self.pos_embed[:, :audio_time, :]
        x = self.input_dropout(x)

        for block in self.blocks:
            x = block(x, cond=t_emb, mel_pad_mask=mel_pad_mask)

        x = x.transpose(1, 2)
        velocity = self.output_proj(x)

        return velocity
