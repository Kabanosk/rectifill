import csv
from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm
from loguru import logger


def load_transcripts(source_dir: Path) -> dict[str, str]:
    """
    Scans the directory for .trans.txt files and creates an ID -> Text map.
    """
    transcripts = {}
    logger.info(f"Searching for transcripts in: {source_dir}...")
    # LibriSpeech has a folder structure, searching recursively for all .trans.txt files
    trans_files = list(source_dir.rglob("*.trans.txt"))

    for trans_file in trans_files:
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Line format: "FILE_ID TRANSCRIPTION_TEXT"
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    file_id, text = parts
                    transcripts[file_id] = text

    logger.info(f"Found {len(transcripts)} transcript entries.")
    return transcripts


def load_wav(wav_path: Path, target_sample_rate: int = 16000) -> torch.Tensor:
    """
    Loads an audio file and resamples it if necessary.
    (Based on the provided utils.py)
    """
    wav, sr = torchaudio.load(str(wav_path))
    if sr != target_sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sample_rate)
        wav = resampler(wav)
    return wav


def save_wav(wav_path: Path, wav: torch.Tensor, sample_rate: int = 16000) -> None:
    """
    Saves a tensor as a WAV file.
    """
    # Ensure the directory exists
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(wav_path), wav, sample_rate)


def process_audio_and_text(
        source_dir: Path,
        output_dir: Path,
        target_sample_rate: int = 16000
) -> None:
    """
    Processes audio files (resampling) and saves .wav + .txt pairs.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. First, load all texts into memory
    transcripts = load_transcripts(source_dir)

    # 2. Find all FLAC files
    flac_files = list(source_dir.rglob("*.flac"))
    logger.info(f"Found {len(flac_files)} audio files. Processing...")

    # Prepare list for metadata.csv
    metadata = []

    # Progress bar
    for flac_path in tqdm(flac_files, desc="Converting"):
        file_id = flac_path.stem  # e.g., '1272-128104-0000'

        # Check if we have text for this file
        if file_id not in transcripts:
            # Optional: uncomment to see missing files
            # logger.warning(f"No transcript for {file_id}, skipping.")
            continue

        text_content = transcripts[file_id]

        # --- Audio Processing ---
        # Using built-in helper functions (modeled after utils.py)
        try:
            waveform = load_wav(flac_path, target_sample_rate)

            output_wav_path = output_dir / f"{file_id}.wav"
            save_wav(output_wav_path, waveform, target_sample_rate)

            # --- Text Saving ---
            output_txt_path = output_dir / f"{file_id}.txt"
            with open(output_txt_path, "w", encoding="utf-8") as f:
                f.write(text_content)

            # Add to metadata (filename | text)
            metadata.append([f"{file_id}.wav", text_content])

        except Exception as e:
            logger.error(f"Error processing file {flac_path}: {e}")

    # --- Saving METADATA.CSV ---
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="|")
        writer.writerow(["file_name", "transcription"])
        writer.writerows(metadata)

    logger.success(f"Done! Data saved in: {output_dir}")
    logger.info("Created .wav + .txt pairs and metadata.csv")


def download_and_prepare(
        root_dir: str,
        dataset_type: str = "dev-clean"
) -> None:
    """
    Main function orchestrating download and processing.
    """
    root_path = Path(root_dir)
    raw_path = root_path / "raw"
    processed_path = root_path / "processed" / dataset_type

    logger.info(f"--- Starting work on dataset: {dataset_type} ---")

    # 1. Downloading (torchaudio checks if files exist)
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

    # Path where torchaudio extracted the files
    # Usually: raw/LibriSpeech/dev-clean
    extracted_path = raw_path / "LibriSpeech" / dataset_type

    # 2. Processing
    process_audio_and_text(extracted_path, processed_path)


if __name__ == "__main__":
    # Directory configuration
    DATA_ROOT = "data"

    # --- VALIDATION SET ---
    download_and_prepare(DATA_ROOT, dataset_type="dev-clean")

    # --- TRAINING SET ---
    download_and_prepare(DATA_ROOT, dataset_type="train-clean-100")

    # --- TEST SET ---
    download_and_prepare(DATA_ROOT, dataset_type="test-clean")