import argparse
from pathlib import Path

import librosa
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
from loguru import logger
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from src.config.config import DataConfig, MelConfig, TrainConfig
from src.data.utils import get_mel_transform
from src.model import BaseModel, get_model


def euler_solver(
        model: BaseModel,
        x_context: torch.Tensor,
        mask_bool: torch.Tensor,
        text_emb: torch.Tensor,
        text_mask: torch.Tensor,
        device: torch.device,
        steps: int = 50
) -> torch.Tensor:
    """
    Solves the ODE using the Euler method for Rectified Flow Matching.

    :param model: The trained PyTorch model predicting the velocity field.
    :param x_context: The original clean mel-spectrogram used as context.
    :param mask_bool: A boolean mask indicating the missing regions (True for holes).
    :param text_emb: Text embeddings conditioning the generation.
    :param text_mask: Padding mask for the text embeddings.
    :param device: The computing device (CPU or CUDA).
    :param steps: The number of integration steps.

    :return: A reconstructed mel-spectrogram tensor of shape [Batch, Mel_Bins, Time].
    """
    x = torch.randn_like(x_context)
    dt = 1.0 / steps
    mask_float = mask_bool.to(torch.float32)

    for step in tqdm(range(steps), desc="Euler ODE Solving"):
        t_val = step * dt
        t_tensor = torch.full((x.shape[0],), t_val, device=device)

        x = torch.where(mask_bool.expand_as(x), x, x_context)

        with torch.no_grad():
            v_pred = model(xt=x, mask=mask_float, t=t_tensor, text_emb=text_emb, text_mask=text_mask)

        x_next = x + v_pred * dt
        x = torch.where(mask_bool.expand_as(x), x_next, x_context)

    return x


def mel_to_audio(mel_tensor: torch.Tensor, save_path: str, mel_config: MelConfig) -> None:
    """
    Converts a generated log-mel spectrogram back to an audio waveform.

    :param mel_tensor: The predicted log-mel spectrogram tensor.
    :param save_path: The file path where the generated .wav file will be saved.
    :param mel_config: Configuration object containing audio processing parameters.
    """
    mel_db = mel_tensor.squeeze().cpu().numpy()
    mel_power = 10.0 ** (mel_db / 10.0)

    logger.info(f"Applying Griffin-Lim algorithm to reconstruct {save_path}...")
    y = librosa.feature.inverse.mel_to_audio(
        M=mel_power,
        sr=mel_config.sample_rate,
        n_fft=mel_config.n_fft,
        hop_length=mel_config.hop_length
    )

    sf.write(save_path, y, mel_config.sample_rate)


def plot_spectrograms(
        mel_orig: torch.Tensor,
        mel_corrupt: torch.Tensor,
        mel_recon: torch.Tensor,
        save_path: str
) -> None:
    """
    Plots the original, corrupted, and reconstructed mel-spectrograms side-by-side.

    :param mel_orig: The original clean mel-spectrogram.
    :param mel_corrupt: The masked (corrupted) mel-spectrogram.
    :param mel_recon: The reconstructed mel-spectrogram.
    :param save_path: Path to save the output PNG image.
    """
    logger.info("Generating spectrogram plots...")

    orig_np = mel_orig.squeeze().cpu().numpy()
    corr_np = mel_corrupt.squeeze().cpu().numpy()
    recon_np = mel_recon.squeeze().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    im0 = axes[0].imshow(orig_np, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title('Original Mel-Spectrogram')
    axes[0].set_ylabel('Mel Bins')

    im2 = axes[1].imshow(recon_np, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title('Reconstructed Mel-Spectrogram (Output)')
    axes[1].set_xlabel('Time Frames')
    axes[1].set_ylabel('Mel Bins')

    im1 = axes[2].imshow(corr_np, aspect='auto', origin='lower', cmap='magma')
    axes[2].set_title('Corrupted Mel-Spectrogram (Input to Model)')
    axes[2].set_ylabel('Mel Bins')


    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.success(f"Spectrogram plot saved successfully: {save_path}")


def parse_args() -> argparse.Namespace:
    """ Parses command line arguments """
    parser = argparse.ArgumentParser(description="Audio Inpainting Custom Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model .pt checkpoint file.")
    parser.add_argument("--input_wav", type=str, required=True, help="Path to the user's input WAV file.")
    parser.add_argument("--text", type=str, required=True, help="The transcript/text of the spoken audio.")
    parser.add_argument("--hole_start", type=float, default=2.0, help="Start time of the hole in seconds.")
    parser.add_argument("--hole_duration", type=float, default=1.0, help="Duration of the hole in seconds.")
    parser.add_argument("--steps", type=int, default=50, help="Number of ODE solver steps (NFE).")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for the generated audio files.")
    args = parser.parse_args()
    return args


def main() -> None:
    """
    Main function to execute the custom user-driven audio generation (inference) script.
    """
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = Path(args.input_wav).stem

    train_config = TrainConfig()
    data_config = DataConfig()
    mel_config = data_config.mel_params
    text_config = data_config.text_params

    logger.info("Loading DiT model, T5 Text Encoder, and Mel-Transform...")
    model = get_model(train_config.model_name, train_config.model_params).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tokenizer = T5Tokenizer.from_pretrained(text_config.model_name)
    text_encoder = T5EncoderModel.from_pretrained(text_config.model_name).to(device)
    text_encoder.eval()

    mel_transform = get_mel_transform(sample_rate=mel_config.sample_rate, n_mels=mel_config.n_mels).to(device)

    logger.info(f"Processing audio file: {args.input_wav}")
    wav, sr = torchaudio.load(args.input_wav)

    if sr != mel_config.sample_rate:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=mel_config.sample_rate)(wav)

    wav = wav.to(device)
    with torch.no_grad():
        mel_original = mel_transform(wav).unsqueeze(0)

    mel_original = mel_original.squeeze(1)
    time_frames = mel_original.shape[-1]

    start_frame = int((args.hole_start * mel_config.sample_rate) / mel_config.hop_length)
    duration_frames = int((args.hole_duration * mel_config.sample_rate) / mel_config.hop_length)
    end_frame = min(start_frame + duration_frames, time_frames)

    mask_bool = torch.zeros((1, 1, time_frames), dtype=torch.bool, device=device)
    if start_frame < time_frames:
        mask_bool[..., start_frame:end_frame] = True
    else:
        logger.warning("Hole start is beyond the audio length! No mask applied.")

    logger.info(f"Extracting T5 embeddings for text: '{args.text}'")
    text_inputs = tokenizer(args.text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_outputs = text_encoder(**text_inputs)
        text_emb = text_outputs.last_hidden_state

    text_mask = torch.zeros((1, text_emb.shape[1]), dtype=torch.bool, device=device)

    mel_corrupted = torch.where(mask_bool.expand_as(mel_original), torch.tensor(-100.0, device=device), mel_original)

    logger.info(f"Starting generation (Inpainting {args.hole_duration}s hole)...")
    mel_generated = euler_solver(
        model=model,
        x_context=mel_original,
        mask_bool=mask_bool,
        text_emb=text_emb,
        text_mask=text_mask,
        device=device,
        steps=args.steps
    )

    plot_path = str(out_dir / f"{file_prefix}_spectrograms.png")
    plot_spectrograms(mel_original, mel_corrupted, mel_generated, plot_path)

    mel_to_audio(mel_corrupted, str(out_dir / f"{file_prefix}_corrupted.wav"), mel_config)
    mel_to_audio(mel_generated, str(out_dir / f"{file_prefix}_reconstructed.wav"), mel_config)
    mel_to_audio(mel_original, str(out_dir / f"{file_prefix}_original.wav"), mel_config)


if __name__ == "__main__":
    main()