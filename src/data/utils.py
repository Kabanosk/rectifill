import random

import numpy as np
import torch
import torchaudio
import soundfile as sf
import torchaudio.transforms as T
from loguru import logger

_vocoder = None


def load_wav(wav_path, sample_rate=16000) -> torch.Tensor:
    """Load a WAV file and resample to the target sample rate if needed.

    :param wav_path: Path to the input WAV file.
    :param sample_rate: Target sample rate (default: 16000).
    :return: A tensor containing the audio data, resampled to the target sample rate.
    """
    wav_np, sr = sf.read(str(wav_path), dtype='float32')
    wav = torch.from_numpy(wav_np)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.t()

    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(wav)
    return wav


def save_wav(wav_path, wav, sample_rate=16000) -> None:
    """Save a tensor as a WAV file with the specified sample rate.

    :param wav_path: Path to save the output WAV file.
    :param wav: A tensor containing the audio data to save.
    :param sample_rate: Sample rate for the output WAV file (default: 16000).
    """
    sf.write(str(wav_path), wav.cpu().numpy().T, sample_rate)


def get_mel_transform(sample_rate: int = 16000, n_mels: int = 80) -> torch.nn.Module:
    """Creates a transform to convert audio waveforms into Log-Mel-Spectrograms
    compatible with the SpeechBrain HiFi-GAN vocoder trained on LibriTTS at 16kHz.

    Uses natural logarithm (ln) scaling with slaney norm and mel scale,
    matching the speechbrain/tts-hifigan-libritts-16kHz preprocessing exactly.

    :param sample_rate: Expected sample rate of the audio (default: 16000).
    :param n_mels: Number of mel filterbanks (default: 80).
    :return: A callable applying MelSpectrogram then log-scaling.
    """
    mel = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=n_mels,
        f_min=0.0,
        f_max=8000.0,
        norm="slaney",
        mel_scale="slaney",
        normalized=False,
    )
    return lambda wav: torch.log(torch.clamp(mel(wav), min=1e-10))


def prepare_mel_for_hifigan(mel_ln: torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Prepares a log-mel spectrogram for the SpeechBrain HiFi-GAN vocoder.

    Since get_mel_transform() already produces output in natural log (ln) scale
    matching the vocoder's preprocessing, this function only handles shape
    formatting to the expected [Batch, Mel_Bins, Time] layout used by SpeechBrain.

    :param mel_ln: Input log-mel spectrogram in ln scale. Expected shape is either
        [Mel_Bins, Time] or [Batch, Mel_Bins, Time].
    :return: A formatted mel spectrogram tensor of shape [Batch, Mel_Bins, Time]
        ready for HiFi-GAN.
    :raises ValueError: If the number of mel-bins is not exactly 80.
    """
    if isinstance(mel_ln, np.ndarray):
        mel_ln = torch.from_numpy(mel_ln)

    if mel_ln.dim() == 2:
        mel_ln = mel_ln.unsqueeze(0)

    if mel_ln.shape[1] != 80:
        raise ValueError(
            f"Dimensionality error! HiFi-GAN requires exactly 80 mel-bins. "
            f"Your tensor has shape: {mel_ln.shape}. Make sure n_mels=80."
        )

    return mel_ln


def mel_to_waveform(mel: torch.Tensor | np.ndarray) -> torch.Tensor:
    """
    Converts a log-mel spectrogram back to audio using the SpeechBrain
    HiFi-GAN vocoder trained on LibriTTS at 16kHz.

    :param mel: Log-mel spectrogram of shape [Mel_Bins, Time] or [Batch, Mel_Bins, Time].
    :return: A 1D tensor representing the audio waveform.
    """
    global _vocoder

    if _vocoder is None:
        logger.info("Loading pre-trained HiFi-GAN vocoder (speechbrain/tts-hifigan-libritts-16kHz)...")
        from speechbrain.inference.vocoders import HIFIGAN
        _vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz",
                                        savedir="pretrained_models/hifigan_libritts")

    mel = prepare_mel_for_hifigan(mel)

    with torch.no_grad():
        wav = _vocoder.decode_batch(mel)

    return wav.squeeze(0).squeeze(0).cpu()


def normalize_mel(mel: torch.Tensor, min_val: float = -11.51, max_val: float = 2.0) -> torch.Tensor:
    """
    Scales log-mel spectrogram values from the [min_val, max_val] range to [-1, 1].
    Ideal for diffusion models and Flow Matching.

    The default range [-11.51, 2.0] matches the natural log scale produced by
    get_mel_transform(), where -11.51 ≈ log(1e-5) and 2.0 is a safe upper bound.

    :param mel: Log-mel spectrogram tensor (ln scale).
    :param min_val: Minimum expected ln value (default: -11.51).
    :param max_val: Maximum expected ln value (default: 2.0).
    :return: Normalized mel spectrogram tensor in the range [-1, 1].
    """
    mel = torch.clamp(mel, min=min_val, max=max_val)
    return ((mel - min_val) / (max_val - min_val)) * 2.0 - 1.0


def denormalize_mel(mel_norm: torch.Tensor, min_val: float = -11.51, max_val: float = 2.0) -> torch.Tensor:
    """
    Reverts the [-1, 1] scaling back to the original natural logarithm (ln) scale.

    :param mel_norm: Normalized mel spectrogram tensor in the range [-1, 1].
    :param min_val: Minimum expected ln value (default: -11.51).
    :param max_val: Maximum expected ln value (default: 2.0).
    :return: Denormalized mel spectrogram tensor in ln scale.
    """
    mel_01 = (mel_norm + 1.0) / 2.0
    return mel_01 * (max_val - min_val) + min_val


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


class SemanticMasker:
    """
    Generates contiguous binary masks for audio inpainting based on semantic
    text units (T5 sub-tokens) rather than arbitrary mel-spectrogram frames.
    """

    def __init__(self, min_tokens: int = 2, max_tokens: int = 15):
        """
        :param min_tokens: Minimum number of text tokens to mask.
        :param max_tokens: Maximum number of text tokens to mask.
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def __call__(self, time_frames: int, durations: torch.Tensor) -> torch.Tensor:
        """
        Generates a 1D boolean mask for the time dimension.

        :param time_frames: The actual length of the spectrogram in the time dimension.
        :param durations: Tensor of shape [Seq_Len] containing mel-frame duration for each text token.
        :return: Boolean tensor of shape [Time] where True represents the hole.
        """
        mask = torch.zeros(time_frames, dtype=torch.bool)
        seq_len = durations.shape[0]

        if seq_len == 0:
            return mask

        high_limit = min(self.max_tokens + 1, seq_len + 1)
        if self.min_tokens >= high_limit:
            num_tokens_to_mask = seq_len
        else:
            num_tokens_to_mask = torch.randint(
                low=self.min_tokens,
                high=high_limit,
                size=(1,)
            ).item()

        max_start_idx = max(0, seq_len - num_tokens_to_mask)
        start_token_idx = torch.randint(low=0, high=max_start_idx + 1, size=(1,)).item()

        start_frame = durations[:start_token_idx].sum().item()
        mask_frames = durations[start_token_idx: start_token_idx + num_tokens_to_mask].sum().item()

        start_frame = min(start_frame, time_frames)
        end_frame = min(start_frame + mask_frames, time_frames)

        # Apply the mask
        if start_frame < end_frame:
            mask[start_frame:end_frame] = True

        if not mask.any() and time_frames > 0:
            fallback_len = min(time_frames // 4, 20)
            fallback_start = torch.randint(low=0, high=max(1, time_frames - fallback_len), size=(1,)).item()
            mask[fallback_start: fallback_start + fallback_len] = True

        return mask


class UniversalMasker:
    """
    Universal Masking Strategy applied at the semantic level.
    Ensures that masks always align perfectly with word/token boundaries
    using the durations tensor. Enables training for TTS, Voice Cloning, and Inpainting.
    """

    def __init__(
            self,
            p_tts: float = 0.15,
            p_continuation: float = 0.25,
            p_prefix: float = 0.20,
            p_inpainting: float = 0.40,
            min_tokens_inpaint: int = 2,
            max_tokens_inpaint: int = 15
    ):
        total = p_tts + p_continuation + p_prefix + p_inpainting
        self.p_tts = p_tts / total
        self.p_continuation = p_continuation / total
        self.p_prefix = p_prefix / total
        self.p_inpainting = p_inpainting / total

        self.min_tokens = min_tokens_inpaint
        self.max_tokens = max_tokens_inpaint

    def __call__(self, time_frames: int, durations: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(time_frames, dtype=torch.bool)
        seq_len = durations.shape[0]

        if seq_len < 3:
            mask[:] = True
            return mask

        rand_val = random.random()
        cumulative = self.p_tts

        # --- TTS MASK ---
        if rand_val < cumulative:
            mask[:] = True
            return mask

        cumulative += self.p_continuation

        # --- CONTINUATION MASK ---
        if rand_val < cumulative:
            # Leave at least 1 token as a prompt, mask the rest
            start_token = random.randint(1, seq_len - 1)
            start_frame = durations[:start_token].sum().item()
            mask[start_frame:] = True
            return mask

        cumulative += self.p_prefix

        # --- PREFIX MASK ---
        if rand_val < cumulative:
            end_token = random.randint(1, seq_len - 1)
            end_frame = durations[:end_token].sum().item()
            mask[:end_frame] = True
            return mask

        # --- INPAINTING MASK ---
        high_limit = min(self.max_tokens + 1, seq_len + 1)
        if self.min_tokens >= high_limit:
            num_tokens_to_mask = seq_len
        else:
            num_tokens_to_mask = random.randint(self.min_tokens, high_limit - 1)

        max_start_idx = max(0, seq_len - num_tokens_to_mask)
        start_token_idx = random.randint(0, max_start_idx)

        start_frame = durations[:start_token_idx].sum().item()
        mask_frames = durations[start_token_idx: start_token_idx + num_tokens_to_mask].sum().item()

        start_frame = min(start_frame, time_frames)
        end_frame = min(start_frame + mask_frames, time_frames)

        if start_frame < end_frame:
            mask[start_frame:end_frame] = True

        if not mask.any():
            mask[:] = True

        return mask
