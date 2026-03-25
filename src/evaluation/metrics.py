import torch


def calculate_lsd(pred_mel: torch.Tensor, target_mel: torch.Tensor) -> float:
    """
    Calculates the Log-Spectral Distance (LSD) between two mel-spectrograms.
    Lower is better.

    :param pred_mel: Predicted mel-spectrogram.
    :param target_mel: Target mel-spectrogram.
    :return: Log-Spectral Distance between them.
    """
    eps = 1e-10

    log_pred = torch.log10(torch.clamp(pred_mel, min=eps))
    log_target = torch.log10(torch.clamp(target_mel, min=eps))

    diff_squared = (log_target - log_pred) ** 2
    lsd_per_frame = torch.sqrt(torch.mean(diff_squared, dim=1))

    return lsd_per_frame.mean().item()
