import torch
import torchaudio
import torchaudio.transforms as T


def load_wav(wav_path, sample_rate=16000) -> torch.Tensor:
    """Load a WAV file and resample to the target sample rate if needed.

    :param wav_path: Path to the input WAV file.
    :param sample_rate: Target sample rate (default: 16000).
    :return: A tensor containing the audio data, resampled to the target sample rate.
    """
    wav, sr = torchaudio.load(wav_path)
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
    amplitude_to_db = T.AmplitudeToDB()

    return torch.nn.Sequential(mel_spectrogram, amplitude_to_db)
