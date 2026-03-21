import argparse
import time

import torch
from loguru import logger
from tqdm import tqdm

from src.config.config import DataConfig, TrainConfig
from src.data.dataset import get_dataloader
from src.model import get_model


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments, pulling defaults from config classes.
    """
    parser = argparse.ArgumentParser(description="Audio Inpainting Training Pipeline")
    default_train = TrainConfig()
    default_data = DataConfig()

    # --- Data Paths Arguments ---
    parser.add_argument("--train_data", type=str, default="data/processed/train-clean-100",
                        help="Path to the training dataset directory.")
    parser.add_argument("--val_data", type=str, default="data/processed/dev-clean",
                        help="Path to the validation dataset directory.")

    # --- Dataloader Arguments ---
    parser.add_argument("--batch_size", type=int, default=default_data.batch_size,
                        help="Number of samples per batch.")
    parser.add_argument("--num_workers", type=int, default=default_data.num_workers,
                        help="Number of CPU threads for data loading.")
    parser.add_argument("--max_mel_length", type=int, default=default_data.max_mel_length,
                        help="Maximum sequence length for Mel-spectrograms.")

    # --- Training Arguments ---
    parser.add_argument("--model_name", type=str, default=default_train.model_name,
                        help="Model architecture to use (e.g., 'interpolate', 'rfm_dit').")
    parser.add_argument("--epochs", type=int, default=default_train.epochs,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=default_train.learning_rate,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--device", type=str, default=default_train.device,
                        help="Compute device.")
    parser.add_argument("--checkpoint_path", type=str, default=default_train.checkpoint_path,
                        help="Checkpoint directory path.")
    parser.add_argument("--log_interval", type=int, default=default_train.log_interval,
                        help="Interval (in batches) for logging training progress.")

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing configurations and DataLoaders...")

    train_data_config = DataConfig(
        data_path=args.train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_mel_length=args.max_mel_length,
        shuffle=True,
        drop_last=True
    )
    train_loader = get_dataloader(train_data_config)

    val_data_config = DataConfig(
        data_path=args.val_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_mel_length=args.max_mel_length,
        shuffle=False,
        drop_last=False
    )
    val_loader = get_dataloader(val_data_config)

    logger.info(f"Train batches per epoch: {len(train_loader)}")
    logger.info(f"Val batches per epoch: {len(val_loader)}")

    train_config = TrainConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        log_interval=args.log_interval
    )

    # 3. Initialize Model and Optimizer
    model = get_model(train_config.model_name).to(train_config.device)
    optimizer = model.configure_optimizers(learning_rate=train_config.learning_rate)
    logger.info(f"Loaded model [{train_config.model_name}] and moved to {train_config.device}")

    # ==========================================
    # MAIN TRAINING LOOP
    # ==========================================
    logger.info("Starting training loop...")

    for epoch in range(1, train_config.epochs + 1):
        # --- TRAIN PHASE ---
        model.train()
        train_start_time = time.time()
        train_loss_accumulated = 0.0

        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}:", leave=True)
        for batch_idx, batch in enumerate(train_pbar):
            mel = batch['mel'].to(train_config.device)
            mask = batch['inpainting_mask'].to(train_config.device)
            text_emb = batch['embedding'].to(train_config.device)
            if batch_idx % train_config.log_interval == 0:
                logger.info(f"mel={mel.shape} | mask={mask.shape} | embedding={text_emb.shape}")

            if optimizer is not None:
                optimizer.zero_grad()

            output = model(mel=mel, inpainting_mask=mask, embedding=text_emb)

            fake_loss = torch.tensor(0.001, requires_grad=True, device=train_config.device)
            if optimizer is not None:
                fake_loss.backward()
                optimizer.step()

            # Update progress bar
            train_loss_accumulated += fake_loss.item()
            train_pbar.set_postfix({"loss": f"{fake_loss.item():.4f}"})

        train_time = time.time() - train_start_time
        avg_train_loss = train_loss_accumulated / len(train_loader)
        logger.info(f"Epoch {epoch} Training Time: {train_time:.2f}s | Avg Train Loss: {avg_train_loss:.4f}")

        # --- VALIDATION PHASE ---
        model.eval()
        val_start_time = time.time()
        val_loss_accumulated = 0.0

        val_pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch}", leave=True)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                mel = batch['mel'].to(train_config.device)
                mask = batch['inpainting_mask'].to(train_config.device)
                text_emb = batch['embedding'].to(train_config.device)

                output = model(mel=mel, inpainting_mask=mask, embedding=text_emb)

                fake_val_loss = 0.001
                val_loss_accumulated += fake_val_loss

        val_time = time.time() - val_start_time
        avg_val_loss = val_loss_accumulated / len(val_loader) if len(val_loader) > 0 else 0

        logger.info(f"Epoch {epoch} Validation Time: {val_time:.2f}s | Avg Val Loss: {avg_val_loss:.4f}")

    logger.success("Training completely finished na essie! 🚀")


if __name__ == "__main__":
    main()