import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.config.config import ModelConfig
from src.model.modules import apply_rotary_pos_emb, ModulatedLayerNorm, PhonemeEncoder, SinusoidalPositionEmbeddings, RotaryEmbedding


class DiTBlock(nn.Module):
    """
    A single block of the Diffusion Transformer.
    Incorporates Self-Attention for audio context and Cross-Attention for text/phonemes conditioning.
    """

    def __init__(self, hidden_size: int, num_heads: int, text_dim: int, cond_dim: int, dropout: float):
        """
        :param hidden_size: The feature dimension of the audio patches.
        :param num_heads: Number of attention heads for self and cross attention.
        :param text_dim: The feature dimension of the incoming text embeddings.
        :param cond_dim: The dimension of the conditioning tensor (timestep embedding).
        :param dropout: Dropout probability for regularization.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout

        self.norm1 = ModulatedLayerNorm(hidden_size, cond_dim)

        # Self-Attention projections
        self.qkv_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.attn_out_proj = nn.Linear(hidden_size, hidden_size)

        self.norm2 = ModulatedLayerNorm(hidden_size, cond_dim)

        # Cross-Attention projections
        self.q_cross_proj = nn.Linear(hidden_size, hidden_size)
        self.k_cross_proj = nn.Linear(text_dim, hidden_size)
        self.v_cross_proj = nn.Linear(text_dim, hidden_size)
        self.cross_out_proj = nn.Linear(hidden_size, hidden_size)

        self.norm3 = ModulatedLayerNorm(hidden_size, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor, text_emb: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor,
                text_mask: torch.Tensor = None, mel_pad_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Processes the input through self-attention, cross-attention, and FFN.

        :param x: Audio patches tensor of shape [Batch, Seq_Len, Hidden_Size].
        :param cond: Timestep condition tensor of shape [Batch, Cond_Dim].
        :param text_emb: Text embeddings tensor of shape [Batch, Text_Seq_Len, Text_Dim].
        :param rope_cos: Cosine component of Rotary Positional Embeddings.
        :param rope_sin: Sine component of Rotary Positional Embeddings.
        :param text_mask: Optional boolean mask for text embeddings. True means padding.
        :param mel_pad_mask: Optional boolean mask for mel-spectrogram padding. True means padding.

        :return: Processed tensor of shape [Batch, Seq_Len, Hidden_Size].
        """
        # --- Self Attention ---
        h = self.norm1(x, cond)
        B, L, C = h.shape

        qkv = self.qkv_proj(h).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [Batch, Heads, Seq_Len, HeadDim]

        # Apply Rotary Positional Embeddings to Q and K
        q = apply_rotary_pos_emb(q, rope_cos, rope_sin)
        k = apply_rotary_pos_emb(k, rope_cos, rope_sin)

        attn_mask = None
        if mel_pad_mask is not None:
            attn_mask = ~mel_pad_mask.unsqueeze(1).unsqueeze(2)  # [Batch, 1, 1, Seq_Len]

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0
        )

        attn_out = attn_out.transpose(1, 2).reshape(B, L, C)
        x = x + self.attn_out_proj(attn_out)

        # --- Cross Attention ---
        h = self.norm2(x, cond)

        q_cross = self.q_cross_proj(h).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_cross = self.k_cross_proj(text_emb).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_cross = self.v_cross_proj(text_emb).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        cross_mask = None
        if text_mask is not None:
            cross_mask = ~text_mask.unsqueeze(1).unsqueeze(2)  # [Batch, 1, 1, Text_Seq_Len]

        cross_out = torch.nn.functional.scaled_dot_product_attention(
            q_cross, k_cross, v_cross,
            attn_mask=cross_mask,
            dropout_p=self.dropout if self.training else 0.0
        )

        cross_out = cross_out.transpose(1, 2).reshape(B, L, C)
        x = x + self.cross_out_proj(cross_out)

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
        self.gradient_checkpointing = False  # turned off by default, changed in main script

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )

        # Channel-wise concatenation: xt (noise) + x_context (clean) + mask
        in_channels = (config.mel_bins * 2) + 1
        self.input_proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=config.hidden_size,
            kernel_size=5,
            padding=2
        )

        if self.config.context_type == "phonemes":
            self.phoneme_encoder = PhonemeEncoder(
                vocab_size=config.phoneme_vocab_size,
                hidden_dim=config.text_dim,
                num_layers=config.phoneme_layers,
                num_heads=config.phoneme_heads,
            )

        # Rotary Positional Embeddings (RoPE) replacing absolute positional embeddings
        self.rope = RotaryEmbedding(config.hidden_size // config.num_heads)

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

        self.final_norm = ModulatedLayerNorm(config.hidden_size, config.hidden_size)
        self.output_proj = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.mel_bins, kernel_size=1)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, xt: torch.Tensor, x_context: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass predicting the velocity field v = x1 - x0.

        :param xt: Noisy mel-spectrogram of shape [Batch, Mel_Bins, Time].
        :param x_context: Mel-spectrogram with hole [Batch, Mel_Bins, Time].
        :param mask: Inpainting mask of shape (1 indicates missing regions) [Batch, 1, Time].
        :param kwargs: kwargs takes other variables that are not presented in BaseModel class like:
            - t: Diffusion timestep of shape [Batch].
            - text_emb: Text embeddings of shape [Batch, Text_Seq_Len, Text_Dim].
            - text_mask: Optional boolean mask for text embeddings of shape [Batch, Text_Seq_Len].
            - phonemes_ids: Optional phoneme IDs of shape [Batch, Text_Seq_Len] (used if context_type is "phonemes").
            - mel_pad_mask (torch.Tensor): Padding mask for audio [Batch, Time].
            - cfg_drop_mask (torch.Tensor): Boolean mask for context dropping during training [Batch, 1, 1].

        :return: Predicted velocity field of shape [Batch, Mel_Bins, Time].
        """
        t = kwargs["t"]
        text_mask = kwargs.get("text_mask", None)
        mel_pad_mask = kwargs.get("mel_pad_mask", None)
        cfg_drop_mask = kwargs.get("cfg_drop_mask", None)

        is_fully_unconditional = cfg_drop_mask is not None and torch.all(cfg_drop_mask)

        if self.config.context_type == "phonemes":
            phoneme_ids = kwargs.get("phoneme_ids", torch.tensor([], device=xt.device))

            if is_fully_unconditional:
                # Bypass the heavy transformer encoder and use pre-learned null embeddings
                text_emb = self.null_text_embed.expand(phoneme_ids.shape[0], phoneme_ids.shape[1], -1)
            else:
                text_emb = self.phoneme_encoder(phoneme_ids, src_key_padding_mask=text_mask)
                # Handle partial dropping within the batch (during training)
                if cfg_drop_mask is not None:
                    null_emb = self.null_text_embed.expand(text_emb.shape[0], text_emb.shape[1], -1)
                    text_emb = torch.where(cfg_drop_mask, null_emb, text_emb)
        else:
            # T5 processing branch
            text_emb = kwargs.get("text_emb", None)
            if text_emb is not None and cfg_drop_mask is not None:
                null_emb = self.null_text_embed.expand(text_emb.shape[0], text_emb.shape[1], -1)
                text_emb = torch.where(cfg_drop_mask, null_emb, text_emb)

        t_emb = self.time_mlp(t * 1000.0)  # [Batch, Hidden_Size]
        x = torch.cat([xt, x_context, mask], dim=1)  # [Batch, 2 * Mel_Bins + 1, Time]

        if mel_pad_mask is not None:
            x = x.masked_fill(mel_pad_mask.unsqueeze(1), 0.0)

        x = self.input_proj(x)  # [Batch, Hidden_Size, Time]
        x = x.transpose(1, 2)  # [Batch, Time, Hidden_Size]

        x = self.input_dropout(x)

        seq_len = x.shape[1]
        rope_cos, rope_sin = self.rope(seq_len, x.device)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                def checkpointed_block(x_in):
                    return block(x_in, cond=t_emb, text_emb=text_emb, rope_cos=rope_cos, rope_sin=rope_sin,
                                 text_mask=text_mask, mel_pad_mask=mel_pad_mask)

                x = checkpoint(checkpointed_block, x, use_reentrant=False)
            else:
                x = block(x, cond=t_emb, text_emb=text_emb, rope_cos=rope_cos, rope_sin=rope_sin, text_mask=text_mask,
                          mel_pad_mask=mel_pad_mask)

        x = self.final_norm(x, t_emb)

        x = x.transpose(1, 2)  # [Batch, Hidden_Size, Time]

        velocity = self.output_proj(x)  # [Batch, Mel_Bins, Time]
        return velocity
