from typing import Optional

import torch
from torch.optim import Optimizer

from src.model import BaseModel


class InterpolationBaseline(BaseModel):
    """
    Baseline model that performs linear interpolation in the Mel-spectrogram domain
    to fill in the missing audio frames (inpainting).
    Since it uses classical DSP logic, it has no learnable parameters.
    """
    def __init__(self):
        super().__init__()

    def forward(self, mel: torch.Tensor, inpainting_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Fills the holes in the Mel-spectrogram using linear interpolation.

        :param mel: Tensor of shape [batch_size, num_channels, num_mels, time_frames]
        :param inpainting_mask: Boolean tensor of shape (True for holes) [batch_size, num_channels, time_frames]

        :return: Reconstructed Mel-spectrogram of shape [batch_size, num_channels, num_mels, time_frames]
        """
        batch_size, num_channels, num_mels, time_frames = mel.shape
        output = mel.clone()

        for batch_idx in range(batch_size):
            mask_1d = inpainting_mask[batch_idx, 0, :]  # Shape: [time_frames]

            hole_indices = torch.nonzero(mask_1d, as_tuple=True)[0]

            if len(hole_indices) == 0:
                continue  # No hole to fix

            start_idx = hole_indices[0].item()
            end_idx = hole_indices[-1].item()
            hole_length = end_idx - start_idx + 1

            frame_before = mel[batch_idx, :, :, max(0, start_idx - 1)]  # [num_channels, num_mels]
            frame_after = mel[batch_idx, :, :, min(time_frames - 1, end_idx + 1)]  # [num_channels, num_mels]

            steps = torch.linspace(0, 1, steps=hole_length, device=mel.device)
            steps = steps.view(hole_length, 1, 1)

            frame_before = frame_before.unsqueeze(0)
            frame_after = frame_after.unsqueeze(0)

            interpolated_frames = (1.0 - steps) * frame_before + steps * frame_after
            interpolated_frames = interpolated_frames.permute(1, 2, 0)
            output[batch_idx, :, :, start_idx: end_idx + 1] = interpolated_frames

        return output

    def configure_optimizers(self, **kwargs) -> Optional[Optimizer]:
        """ For interpolation model there is no need for optimizer. """
        return None
