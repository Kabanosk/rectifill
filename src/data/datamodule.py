import lightning.pytorch as pl
from torch.utils.data import DataLoader
from typing import Optional

from src.config.config import DataConfig
from src.data.dataset import LibriSpeechDataset, LibriSpeechCollator


class LibriSpeechDataModule(pl.LightningDataModule):
    """
    LightningDataModule for LibriSpeech.
    Handles dataset initialization and DataLoader creation cleanly.
    """

    def __init__(self, train_config: DataConfig, val_config: DataConfig):
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config
        self.collator = LibriSpeechCollator()

        self.train_dataset: Optional[LibriSpeechDataset] = None
        self.val_dataset: Optional[LibriSpeechDataset] = None

    def setup(self, stage: Optional[str] = None):
        """Initializes datasets. This is called automatically by the Trainer."""
        if stage == "fit" or stage is None:
            self.train_dataset = LibriSpeechDataset(
                data_dir=self.train_config.data_path,
                max_mel_length=self.train_config.max_mel_length
            )
            self.val_dataset = LibriSpeechDataset(
                data_dir=self.val_config.data_path,
                max_mel_length=self.val_config.max_mel_length
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            shuffle=self.train_config.shuffle,
            num_workers=self.train_config.num_workers,
            collate_fn=self.collator,
            drop_last=self.train_config.drop_last,
            pin_memory=self.train_config.pin_memory
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_config.batch_size,
            shuffle=self.val_config.shuffle,
            num_workers=self.val_config.num_workers,
            collate_fn=self.collator,
            drop_last=self.val_config.drop_last,
            pin_memory=self.val_config.pin_memory
        )
