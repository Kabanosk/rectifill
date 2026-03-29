from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

from src.config.config import DataConfig, TrainConfig
from src.data.dataset import get_dataloader
from src.data.utils import (denormalize_mel, mel_to_waveform, normalize_mel,
                            save_wav)
from src.model import get_model
from src.utils.rfm import sample_euler


def visualize_and_listen(checkpoint_path: str):
    """
    Evaluates a single batch from the validation set, plots the spectrograms,
    and reconstructs audio using Griffin-Lim algorithm.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = DataConfig(batch_size=1, shuffle=True, data_path="data/processed/test-clean")
    train_config = TrainConfig()

    val_loader = get_dataloader(data_config)
    batch = next(iter(val_loader))

    model = get_model(train_config.model_name, train_config.model_params).to(device)

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'ema_model_state_dict' in ckpt and ckpt['ema_model_state_dict']:
        model.load_state_dict(ckpt['ema_model_state_dict'])
        logger.info("Loaded EMA weights.")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info("Loaded standard model weights.")

    model.eval()

    mel_raw = batch['mel'].squeeze(1).to(device)  # [1, 128, Time]
    mel_norm = normalize_mel(mel_raw)
    mask_bool = batch['inpainting_mask'].to(device)  # [1, 1, Time]
    text_emb = batch['embedding'].to(device)
    text_mask = batch['text_padding_mask'].to(device)
    mel_pad_mask = batch['mel_padding_mask'].to(device)

    logger.info("Running ODE solver (Euler)...")
    with torch.no_grad():
        generated_mel_norm = sample_euler(
            model=model,
            x1_context=mel_norm,
            mask_bool=mask_bool,
            text_emb=text_emb,
            text_mask=text_mask,
            mel_pad_mask=mel_pad_mask,
            num_steps=50,
            cfg_scale=train_config.cfg_scale
        )

    generated_mel_db = denormalize_mel(generated_mel_norm)
    masked_mel_db = mel_raw.clone()
    masked_mel_db = torch.where(mask_bool.expand_as(masked_mel_db), torch.tensor(-100.0, device=device), masked_mel_db)
    original_np = mel_raw[0].cpu().numpy()
    masked_np = masked_mel_db[0].cpu().numpy()
    generated_np = generated_mel_db[0].cpu().numpy()

    # --- PLOTTING ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)

    vmax = np.max(original_np)
    vmin = np.min(original_np)

    axes[0].imshow(original_np, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='magma')
    axes[0].set_title('Original Mel-Spectrogram')

    axes[1].imshow(masked_np, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='magma')
    axes[1].set_title('Masked Input (Hole to inpaint)')

    axes[2].imshow(generated_np, aspect='auto', origin='lower', vmin=vmin, vmax=vmax, cmap='magma')
    axes[2].set_title('Inpainted Mel-Spectrogram (RFM Output)')

    for ax in axes:
        ax.set_ylabel('Mel Bins')
    axes[-1].set_xlabel('Time Frames')

    plot_path = output_dir / "inpainting_result.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.success(f"Plot saved to {plot_path}")

    # --- AUDIO GENERATION (Griffin-Lim) ---
    logger.info("Converting Mel-spectrograms back to audio using Griffin-Lim...")

    sr = data_config.mel_params.sample_rate

    # Process original
    wav_orig = mel_to_waveform(original_np, sr=sr)
    save_wav(str(output_dir / "original.wav"), torch.tensor(wav_orig).unsqueeze(0), sample_rate=sr)

    # Process masked (with silence in the hole)
    wav_masked = mel_to_waveform(masked_np, sr=sr)
    save_wav(str(output_dir / "masked.wav"), torch.tensor(wav_masked).unsqueeze(0), sample_rate=sr)

    # Process generated
    wav_gen = mel_to_waveform(generated_np, sr=sr)
    save_wav(str(output_dir / "inpainted.wav"), torch.tensor(wav_gen).unsqueeze(0), sample_rate=sr)

    logger.success(f"Audio files saved to {output_dir}/ directory!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str)
    args = parser.parse_args()

    visualize_and_listen(args.model)