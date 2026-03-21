import csv
from pathlib import Path

import torch
import torchaudio
from loguru import logger
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from src.config.config import DataConfig, MelConfig, TextConfig
from src.data.utils import get_mel_transform, load_wav


def load_transcripts(source_dir: Path) -> dict[str, str]:
    """
    Scans the directory for .trans.txt files and creates an ID -> Text map.

    :param source_dir: Path to the directory containing .trans.txt files.
    :return: Dictionary mapping file IDs to their transcriptions.
    """
    transcripts = {}
    logger.info(f"Searching for transcripts in: {source_dir}...")
    trans_files = list(source_dir.rglob("*.trans.txt"))

    for trans_file in trans_files:
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(" ", 1)
                if len(parts) == 2:
                    file_id, text = parts
                    transcripts[file_id] = text

    logger.info(f"Found {len(transcripts)} transcript entries.")
    return transcripts


def process_audio_and_text(source_dir: Path, output_dir: Path, config: DataConfig) -> None:
    """
    Processes audio files (resampling), generates log-mel spectrograms and T5 embeddings,
    and saves .pt + .txt pairs, all driven by the provided configuration.

    :param source_dir: Path to the directory containing the raw audio files and transcripts.
    :param output_dir: Path to the directory where processed .pt and .txt files will be saved.
    :param config: DataConfig object containing all processing parameters.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transcripts = load_transcripts(source_dir)

    # Initialization of models based on config
    logger.info(f"Loading text model ({config.text_params.model_name}) on {device}...")
    tokenizer = T5Tokenizer.from_pretrained(config.text_params.model_name)
    text_encoder = T5EncoderModel.from_pretrained(config.text_params.model_name).to(device)
    text_encoder.eval()

    logger.info("Initializing Mel-Spectrogram transform from config...")
    mel_transform = get_mel_transform(
        sample_rate=config.mel_params.sample_rate,
        n_mels=config.mel_params.n_mels
    ).to(device)

    flac_files = list(source_dir.rglob("*.flac"))
    logger.info(f"Found {len(flac_files)} audio files. Processing...")

    metadata = []

    for flac_path in tqdm(flac_files, desc="Converting & Embedding"):
        file_id = flac_path.stem

        if file_id not in transcripts:
            continue

        text_content = transcripts[file_id]

        try:
            # --- Audio Processing ---
            waveform = load_wav(flac_path, config.mel_params.sample_rate).to(device)

            with torch.no_grad():
                mel_spec = mel_transform(waveform).cpu()

            output_mel_path = output_dir / f"{file_id}_mel.pt"
            torch.save(mel_spec, output_mel_path)

            # --- Text Processing ---
            inputs = tokenizer(text_content, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = text_encoder(**inputs)
                embedding = outputs.last_hidden_state.squeeze(0).cpu()

            output_emb_path = output_dir / f"{file_id}_emb.pt"
            torch.save(embedding, output_emb_path)

            # --- Text Saving ---
            output_txt_path = output_dir / f"{file_id}.txt"
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            # Add to metadata
            metadata.append([f"{file_id}_mel.pt", f"{file_id}_emb.pt", text_content])

        except Exception as e:
            logger.error(f"Error processing file {flac_path}: {e}")

    # --- Saving METADATA.CSV ---
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        writer.writerow(["mel_file", "embedding_file", "transcription"])
        writer.writerows(metadata)

    logger.success(f"Done! Data saved in: {output_dir}")
    logger.info("Created .pt pairs and metadata.csv")


def download_and_prepare(config: DataConfig, dataset_type: str = "dev-clean") -> None:
    """
    Main function orchestrating download and processing based on config.

    :param config: DataConfig object specifying paths and rules.
    :param dataset_type: The specific LibriSpeech split to download and process.
    """
    root_path = Path(config.data_path)
    raw_path = root_path / "raw"
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path = root_path / "processed" / dataset_type

    logger.info(f"--- Starting work on dataset: {dataset_type} ---")

    logger.info(f"Downloading to {raw_path} (if not exists)...")
    try:
        _ = torchaudio.datasets.LIBRISPEECH(
            root=str(raw_path),
            url=dataset_type,
            download=True
        )
    except Exception as e:
        logger.error(f"Download error: {e}")
        return

    extracted_path = raw_path / "LibriSpeech" / dataset_type

    process_audio_and_text(extracted_path, processed_path, config)


if __name__ == "__main__":
    main_config = DataConfig(data_path="data", mel_params=MelConfig(), text_params=TextConfig())

    logger.info("Starting processing: VALIDATION SET")
    download_and_prepare(main_config, dataset_type="dev-clean")

    logger.info("Starting processing: TRAINING SET")
    download_and_prepare(main_config, dataset_type="train-clean-100")

    logger.info("Starting processing: TEST SET")
    download_and_prepare(main_config, dataset_type="test-clean")