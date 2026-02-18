import csv
from pathlib import Path
from typing import Optional, Callable

import torch
import torchaudio
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from src.config.config import DataConfig


class LibriSpeechCollator:
    """
    Class responsible for batch preparation (Dynamic Padding).
    Adjusts the batch length to the LONGEST element in THIS SPECIFIC batch.
    """

    def __call__(self, batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

        # 1. Extract audio and texts
        audios = [item['audio'].squeeze(0) for item in batch] # squeeze(0) removes the channel dimension
        texts = [item['text'] for item in batch]
        ids = [item['id'] for item in batch]

        # 2. Audio padding
        audio_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)

        # 3. Collect lengths (for masking in the model)
        lengths = torch.tensor([audio.shape[0] for audio in audios], dtype=torch.long)

        return {
            "audio": audio_padded,  # Shape: [Batch, Time]
            "text": texts,          # List of strings
            "lengths": lengths,     # Original lengths
            "id": ids,
        }


class LibriSpeechDataset(Dataset):
    """
    PyTorch Dataset for loading the processed LibriSpeech data.
    Now supports transformations (augmentations).
    """

    def __init__(self, data_dir: str | Path, transform: Optional[Callable] = None, max_length: Optional[int] = None):
        """Initializes the dataset by loading metadata and preparing file paths.

        :param data_dir: Directory containing the processed audio files and metadata.csv.
        :param transform: Optional transformation function to apply to the audio data.
        :param max_length: Optional maximum length of audio in samples to prevent memory issues.
        """
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "metadata.csv"
        self.transform = transform
        self.max_length = max_length  # In samples (e.g., 16000 * seconds)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}. Please run processing first.")

        self.samples = []
        logger.info(f"Loading dataset from {self.data_dir}...")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            next(reader)  # Skip header
            for row in reader:
                if len(row) == 2:
                    self.samples.append(row)

        logger.info(f"Dataset loaded. Found {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, text = self.samples[idx]
        audio_path = self.data_dir / file_name

        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))

            # 1. Apply user transforms (e.g. Augmentation, Noise, Speed)
            if self.transform:
                waveform = self.transform(waveform)

            # 2. Safety cut (to prevent memory overflow with extremely long files)
            if self.max_length and waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]

            return {
                "audio": waveform,
                "text": text,
                "sample_rate": sample_rate,
                "id": file_name.replace(".wav", "")
            }
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            return None


def get_dataloader(config: DataConfig, transform: Optional[Callable] = None) -> DataLoader:
    """Creates a DataLoader for the LibriSpeech dataset based on the provided configuration.

    :param config: An instance of DataConfig containing dataset parameters.
    :param transform: Optional transformation function to apply to the audio data.
    :return: A DataLoader instance for the LibriSpeech dataset.
    """
    dataset = LibriSpeechDataset(
        config.data_path,
        transform=transform,
        max_length=config.max_audio_length
    )

    collator = LibriSpeechCollator()

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=collator,
        drop_last=config.drop_last,
        pin_memory=True
    )

    return loader