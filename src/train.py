import argparse
import dataclasses
import math
from pathlib import Path
from loguru import logger

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar

from src.config.config import DataConfig, TrainConfig
from src.data.datamodule import LibriSpeechDataModule
from src.model import get_model
from src.model.lit_rfm import LitRFM
from src.utils.callbacks import EMACallback


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="RectiFill Lightning Trainer")
    default_train = TrainConfig()
    default_data = DataConfig()

    # --- Data Paths Arguments ---
    parser.add_argument("--train_data", type=str, default="data/processed/train-clean-360",
                        help="Path to the training dataset directory.")
    parser.add_argument("--val_data", type=str, default="data/processed/dev-clean",
                        help="Path to the validation dataset directory.")

    # --- DataLoader Arguments ---
    parser.add_argument("--batch_size", type=int, default=default_data.batch_size,
                        help="Number of samples per batch.")
    parser.add_argument("--num_workers", type=int, default=default_data.num_workers,
                        help="Number of CPU threads for data loading.")
    parser.add_argument("--max_mel_length", type=int, default=default_data.max_mel_length,
                        help="Maximum sequence length for Mel-spectrograms.")

    # --- Training Arguments ---
    parser.add_argument("--model_name", type=str, default="rfm_dit", choices=["rfm_dit", "aligned_dit"],
                        help="Model architecture to use (e.g., 'rfm_dit', 'aligned_dit').")
    parser.add_argument("--checkpoint_path", type=str, default=default_train.checkpoint_path,
                        help="Checkpoint directory path.")
    parser.add_argument("--log_interval", type=int, default=default_train.log_interval,
                        help="Interval (in batches) for logging training progress.")
    parser.add_argument("--epochs", type=int, default=default_train.epochs,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=default_train.learning_rate,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--weight_decay", type=float, default=default_train.weight_decay,
                        help="Weight decay for optimizer (L2 regularization).")
    parser.add_argument("--gradient_clip_val", type=float, default=default_train.gradient_clip_val,
                        help="Max norm for gradient clipping.")
    parser.add_argument("--seed", type=int, default=default_train.seed,
                        help="Random seed for reproducibility.")

    # --- Lightning Specific Args ---
    parser.add_argument("--devices", type=int, default=default_train.devices, help="Number of GPUs to use")
    parser.add_argument("--strategy", type=str, default=default_train.strategy, help="Distributed training strategy (e.g., 'ddp', 'auto')")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to .ckpt to resume training")
    parser.add_argument("--wandb_id", type=str, default=None, help="WandB Run ID to resume")

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing PyTorch Lightning Training Pipeline...")

    train_config = TrainConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        checkpoint_path=args.checkpoint_path,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_val=args.gradient_clip_val,
        seed=args.seed,
        log_interval=args.log_interval,
        devices=args.devices,
        strategy=args.strategy,
    )

    pl.seed_everything(train_config.seed, workers=True)

    train_data_config = DataConfig(
        data_path=args.train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_mel_length=args.max_mel_length,
        shuffle=True,
        drop_last=True
    )
    val_data_config = DataConfig(
        data_path=args.val_data, batch_size=args.batch_size,
        num_workers=args.num_workers, max_mel_length=args.max_mel_length,
        shuffle=False, drop_last=False
    )

    datamodule = LibriSpeechDataModule(train_config=train_data_config, val_config=val_data_config)
    datamodule.setup()

    core_model = get_model(train_config.model_name, train_config.model_params)
    steps_per_epoch = math.ceil(len(datamodule.train_dataloader()) / train_config.accumulation_steps)

    lit_model = LitRFM(core_model=core_model, config=train_config, steps_per_epoch=steps_per_epoch)

    total_params = sum(p.numel() for p in core_model.parameters())
    logger.info(f"Model architecture [{train_config.model_name}] initialized. Total Params: {total_params:,}")

    checkpoint_dir = Path(train_config.checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{train_config.model_name}-{{epoch:02d}}-{{val_epoch_lsd}}",
            monitor="val/epoch_lsd",
            mode="min",
            save_top_k=3,
            save_last=True
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{train_config.model_name}-val-loss-{{epoch:02d}}-{{val_epoch_loss:.4f}}",
            monitor="val/epoch_loss",
            mode="min",
            save_top_k=2,
            save_last=False
        ),
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar(leave=True),
    ]

    if train_config.use_ema:
        callbacks.append(EMACallback(decay=train_config.ema_decay, update_every=train_config.ema_update_every))

    wandb_logger = None
    if train_config.wandb_params.use_wandb:
        config_dict = dataclasses.asdict(train_config) | dataclasses.asdict(train_data_config)
        wandb_logger = WandbLogger(
            project=train_config.wandb_params.project_name,
            id=args.wandb_id,
            config=config_dict,
            resume="allow" if args.wandb_id else None
        )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = "16-mixed" if accelerator == "gpu" else "32-true"

    trainer = pl.Trainer(
        max_epochs=train_config.epochs,
        accelerator=accelerator,
        devices=train_config.devices,
        strategy=train_config.strategy,
        precision=precision,
        accumulate_grad_batches=train_config.accumulation_steps,
        gradient_clip_val=train_config.gradient_clip_val,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=train_config.log_interval,
    )

    logger.info("Starting training loop! 🚀")
    trainer.fit(
        model=lit_model,
        datamodule=datamodule,
        ckpt_path=args.resume_from
    )


if __name__ == "__main__":
    main()