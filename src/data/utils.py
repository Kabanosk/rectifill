import librosa
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


def load_wav(wav_path, sample_rate=16000) -> torch.Tensor:
    """Load a WAV file and resample to the target sample rate if needed.

    :param wav_path: Path to the input WAV file.
    :param sample_rate: Target sample rate (default: 16000).
    :return: A tensor containing the audio data, resampled to the target sample rate.
    """
    wav, sr = torchaudio.load(wav_path, backend="soundfile")
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)
    return wav


def save_wav(wav_path, wav, sample_rate=16000) -> None:
    """Save a tensor as a WAV file with the specified sample rate.

    :param wav_path: Path to save the output WAV file.
    :param wav: A tensor containing the audio data to save.
    :param sample_rate: Sample rate for the output WAV file (default: 16000).
    """
    torchaudio.save(wav_path, wav, sample_rate)


def get_mel_transform(sample_rate=16000, n_mels=128) -> torch.nn.Sequential:
    """Creates a transform to convert audio waveforms into Log-Mel-Spectrograms.

    Uses AmplitudeToDB to compress the dynamic range, which is crucial for
    generative models and stable training.

    :param sample_rate: Expected sample rate of the audio (default: 16000).
    :param n_mels: Number of mel filterbanks (default: 128).
    :return: A Sequential PyTorch module applying MelSpectrogram then AmplitudeToDB.
    """
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=n_mels,
        normalized=True
    )
    amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=100.0)

    return torch.nn.Sequential(mel_spectrogram, amplitude_to_db)


def mel_to_waveform(mel: np.ndarray, sr: int = 16000, n_fft: int = 1024, hop_length: int = 512) -> np.ndarray:
    """
    Converts a log-mel spectrogram back to audio using the Griffin-Lim algorithm.

    :param mel: Log-mel spectrogram to convert.
    :param sr: Expected sample rate of the audio (default: 16000).
    :param n_fft: Number of FFT bins (default: 1024).
    :param hop_length: Hop length (default: 512).
    :return: A numpy array representing the audio waveform.
    """
    mel_power = 10.0 ** (mel / 10.0)
    wav = librosa.feature.inverse.mel_to_audio(
        M=mel_power, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    return wav


def normalize_mel(mel: torch.Tensor, min_db: float = -100.0, max_db: float = 20.0) -> torch.Tensor:
    """
    Scales Mel-spectrogram values from the [min_db, max_db] range to [-1, 1].
    Ideal for diffusion models and Flow Matching.

    :param mel: Mel-spectrogram tensor in decibels (dB).
    :param min_db: Minimum expected dB value (default: -100.0).
    :param max_db: Maximum expected dB value (default: 20.0).
    :return: Normalized Mel-spectrogram tensor in the range [-1, 1].
    """
    mel = torch.clamp(mel, min=min_db, max=max_db)
    return ((mel - min_db) / (max_db - min_db)) * 2.0 - 1.0


def denormalize_mel(mel_norm: torch.Tensor, min_db: float = -100.0, max_db: float = 20.0) -> torch.Tensor:
    """
    Reverts the [-1, 1] scaling back to the original decibel (dB) scale.

    :param mel_norm: Normalized Mel-spectrogram tensor in the range [-1, 1].
    :param min_db: Minimum expected dB value (default: -100.0).
    :param max_db: Maximum expected dB value (default: 20.0).
    :return: Denormalized Mel-spectrogram tensor in decibels (dB).
    """
    mel_01 = (mel_norm + 1.0) / 2.0
    return mel_01 * (max_db - min_db) + min_db


class RandomInpaintingMasker:
    """
    Generates contiguous binary masks for audio inpainting in the Mel-spectrogram domain.
    1 (True) indicates the missing region (hole to be generated).
    0 (False) indicates the known context.
    """

    def __init__(self, min_hole_ratio: float = 0.1, max_hole_ratio: float = 0.4):
        """
        :param min_hole_ratio: Minimum fraction of the sequence to mask.
        :param max_hole_ratio: Maximum fraction of the sequence to mask.
        """
        self.min_hole_ratio = min_hole_ratio
        self.max_hole_ratio = max_hole_ratio

    def __call__(self, time_frames: int) -> torch.Tensor:
        """
        Generates a 1D boolean mask for the time dimension using PyTorch RNG.

        :param time_frames: The length of the spectrogram in the time dimension.
        :return: Boolean tensor of shape [Time] where True is the hole.
        """
        mask = torch.zeros(time_frames, dtype=torch.bool)

        # Calculate random hole length based on ratio
        min_frames = max(1, int(time_frames * self.min_hole_ratio))
        max_frames = max(1, int(time_frames * self.max_hole_ratio))

        if min_frames >= max_frames:
            hole_length = min_frames
        else:
            hole_length = torch.randint(low=min_frames, high=max_frames + 1, size=(1,)).item()

        # Calculate random start index to position the hole
        max_start_idx = time_frames - hole_length
        if max_start_idx <= 0:
            start_idx = 0
        else:
            start_idx = torch.randint(low=0, high=max_start_idx + 1, size=(1,)).item()

        # Set the hole region to True
        mask[start_idx: start_idx + hole_length] = True

        return mask
