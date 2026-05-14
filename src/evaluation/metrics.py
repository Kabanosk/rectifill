import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


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


def calculate_speech_metrics(pred_wav: torch.Tensor, target_wav: torch.Tensor, sample_rate: int = 16000) -> dict:
    """
    Calculates PESQ and STOI metrics for reconstructed waveforms.

    :param pred_wav: Predicted waveform tensor [1, Time]
    :param target_wav: Ground truth waveform tensor [1, Time]
    :param sample_rate: Audio sample rate (default 16000)
    :return: dict with pesq and stoi scores
    """
    pred_wav = pred_wav.detach().cpu()
    target_wav = target_wav.detach().cpu()

    pesq_metric = PerceptualEvaluationSpeechQuality(sample_rate, 'wb')
    stoi_metric = ShortTimeObjectiveIntelligibility(sample_rate)

    pesq_val = pesq_metric(pred_wav, target_wav).item()
    stoi_val = stoi_metric(pred_wav, target_wav).item()

    return {
        "pesq": pesq_val,
        "stoi": stoi_val
    }
