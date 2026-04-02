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
