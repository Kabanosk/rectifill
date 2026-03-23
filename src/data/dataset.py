import csv
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.config.config import DataConfig
from src.data.utils import RandomInpaintingMasker


class LibriSpeechCollator:
    """
    Collator for dynamic padding of Mel-spectrograms, text embeddings, and inpainting masks.
    Ensures all tensors in a batch have the same dimensions by padding with zeros or False.
    """

    def __call__(self, batch: list[dict]) -> Optional[dict]:
        batch = [item for item in batch if item is not None]
        if not batch:
            return None

        # Extract features from the batch list
        mels = [item['mel'] for item in batch]  # List of [1, 128, T]
        inpainting_masks = [item['inpainting_mask'] for item in batch]  # List of [T]
        embeddings = [item['embedding'] for item in batch]  # List of [Seq_Len, 768]
        texts = [item['text'] for item in batch]
        ids = [item['id'] for item in batch]

        # Pad Mel-spectrograms and Inpainting Masks to the longest in the batch
        max_mel_len = max(mel.shape[-1] for mel in mels)

        padded_mels = []
        padded_inpainting_masks = []
        pad_attention_masks = []

        for mel, inp_mask in zip(mels, inpainting_masks):
            time_frames = mel.shape[-1]
            pad_amount = max_mel_len - time_frames

            # Pad Mel with zeros
            padded_mel = F.pad(mel, (0, pad_amount), value=0.0)
            padded_mels.append(padded_mel)

            # Pad Inpainting Mask with False (padding is NOT a hole to inpaint)
            padded_inp_mask = F.pad(inp_mask, (0, pad_amount), value=False)
            padded_inpainting_masks.append(padded_inp_mask)

            # Create Attention Padding Mask (False for real data, True for padding)
            att_mask = torch.zeros(time_frames, dtype=torch.bool)
            padded_att_mask = F.pad(att_mask, (0, pad_amount), value=True)
            pad_attention_masks.append(padded_att_mask)

        # Stack lists into actual batch tensors
        mels_tensor = torch.stack(padded_mels)  # [B, 1, 128, Max_Time]
        inpainting_masks_tensor = torch.stack(padded_inpainting_masks).unsqueeze(1)  # [B, 1, Max_Time]
        pad_attention_masks_tensor = torch.stack(pad_attention_masks)  # [B, Max_Time]

        # Pad Text Embeddings along the sequence length
        embeddings_tensor = pad_sequence(embeddings, batch_first=True, padding_value=0.0)  # [B, Max_Seq, 768]

        # Create text attention masks
        text_lengths = torch.tensor([emb.shape[0] for emb in embeddings], dtype=torch.long)
        max_text_len = embeddings_tensor.shape[1]
        text_masks_tensor = (
                torch.arange(max_text_len).expand(len(text_lengths), max_text_len) >= text_lengths.unsqueeze(1)
        )

        return {
            "mel": mels_tensor,  # [B, 1, 128, T]
            "inpainting_mask": inpainting_masks_tensor,  # [B, 1, T] - Hole
            "mel_padding_mask": pad_attention_masks_tensor,  # [B, T] - Ignore padding in Attention
            "embedding": embeddings_tensor,  # [B, Seq, 768] - Guidance
            "text_padding_mask": text_masks_tensor,  # [B, Seq]
            "text": texts,
            "id": ids,
        }


class LibriSpeechDataset(Dataset):
    """
    PyTorch Dataset for loading precomputed Log-Mel-Spectrograms and T5 embeddings.
    Dynamically generates masks for the inpainting task using RandomInpaintingMasker.
    """

    def __init__(self, data_dir: str | Path, max_mel_length: Optional[int] = None):
        """
        :param data_dir: Directory containing the processed .pt files and metadata.csv.
        :param max_mel_length: Optional maximum length of mel frames to prevent memory issues.
        """
        self.data_dir = Path(data_dir)
        self.metadata_path = self.data_dir / "metadata.csv"
        self.max_mel_length = max_mel_length
        self.mask_generator = RandomInpaintingMasker(min_hole_ratio=0.1, max_hole_ratio=0.4)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}. Run preparation script first.")

        self.samples = []
        logger.info(f"Loading dataset from {self.data_dir}...")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            next(reader)  # Skip header
            for row in reader:
                if len(row) == 3:
                    self.samples.append(row)

        logger.info(f"Dataset loaded. Found {len(self.samples)} valid samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[dict]:
        mel_file, emb_file, text = self.samples[idx]
        mel_path = self.data_dir / mel_file
        emb_path = self.data_dir / emb_file

        try:
            mel_spec = torch.load(mel_path, weights_only=True)
            embedding = torch.load(emb_path, weights_only=True)

            # Enforce max length if provided in config
            if self.max_mel_length and mel_spec.shape[-1] > self.max_mel_length:
                mel_spec = mel_spec[..., :self.max_mel_length]

            # Generate the inpainting mask dynamically for this specific audio
            time_frames = mel_spec.shape[-1]
            inpainting_mask = self.mask_generator(time_frames)

            return {
                "mel": mel_spec,
                "inpainting_mask": inpainting_mask,
                "embedding": embedding,
                "text": text,
                "id": mel_file.replace("_mel.pt", "")
            }
        except Exception as e:
            logger.error(f"Error loading tensors for {mel_file}: {e}")
            return None


def get_dataloader(config: DataConfig) -> DataLoader:
    """
    Creates a DataLoader for the LibriSpeech dataset based on the provided configuration.

    :param config: An instance of DataConfig containing dataset parameters.
    :return: A DataLoader instance ready for training.
    """
    dataset = LibriSpeechDataset(data_dir=config.data_path, max_mel_length=config.max_mel_length)

    collator = LibriSpeechCollator()

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        collate_fn=collator,
        drop_last=config.drop_last,
        pin_memory=config.pin_memory
    )

    return loader
