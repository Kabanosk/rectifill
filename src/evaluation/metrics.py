import torch


def calculate_lsd(pred_mel: torch.Tensor, target_mel: torch.Tensor, mask_bool: torch.Tensor) -> float:
    """
    Calculates the Log-Spectral Distance (LSD) between two mel-spectrograms
    ONLY for the masked (inpainted) regions.

    Inputs are assumed to be already in decibels (dB).

    :param pred_mel: Predicted mel-spectrogram in dB [Batch, Mel_Bins, Time].
    :param target_mel: Target mel-spectrogram in dB [Batch, Mel_Bins, Time].
    :param mask_bool: Boolean mask where True indicates the generated hole [Batch, 1, Time].
    :return: Mean LSD for the masked frames in dB.
    """
    diff_squared = (target_mel - pred_mel) ** 2
    lsd_per_frame = torch.sqrt(torch.mean(diff_squared, dim=1, keepdim=True))

    masked_lsd = lsd_per_frame[mask_bool]

    if masked_lsd.numel() == 0:
        return 0.0

    return masked_lsd.mean().item()