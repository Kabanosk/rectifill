import argparse
import dataclasses
import math
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from loguru import logger
from tqdm import tqdm

from src.config.config import DataConfig, TrainConfig
from src.data.dataset import get_dataloader
from src.model import get_model


def prepare_rfm_batch(
    x1: torch.Tensor, mask_bool: torch.Tensor, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples time t, generates noise x0, and computes the intermediate state xt
    and target velocity for Rectified Flow Matching.

    :param x1: Target clean mel-spectrogram of shape [Batch, Mel_Bins, Time]
    :param mask_bool: Boolean inpainting mask of shape [Batch, 1, Time]
    :param device: Computing device
    :return: Tuple containing (xt, target_velocity, t)
    """
    batch_size = x1.shape[0]

    # 1. Sample timestep t from U(0, 1)
    t = torch.rand((batch_size,), device=device)
    t_expanded = t.view(-1, 1, 1)  # [B, 1, 1] for broadcasting

    # 2. Sample Noise x0 from N(0, I)
    x0 = torch.randn_like(x1)

    # 3. Calculate intermediate state xt (Linear interpolation)
    xt_hole = t_expanded * x1 + (1.0 - t_expanded) * x0

    # 4. Inpainting condition: Keep original x1 where mask is False (context)
    # Only apply noise to the masked regions.
    xt = torch.where(mask_bool.expand_as(x1), xt_hole, x1)

    # 5. Calculate target velocity v
    target_v = x1 - x0

    return xt, target_v, t


def set_seed(seed: int = 42):
    """
    Ensures experiment reproducibility by setting the seed
    across all used libraries (Python, NumPy, PyTorch).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to: {seed}")


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments, pulling defaults from config classes.
    """
    parser = argparse.ArgumentParser(description="Audio Inpainting Training Pipeline (RFM)")
    default_train = TrainConfig()
    default_data = DataConfig()

    # --- Data Paths Arguments ---
    parser.add_argument("--train_data", type=str, default="data/processed/train-clean-360",
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
    parser.add_argument("--model_name", type=str, default="rfm_dit",
                        help="Model architecture to use (e.g., 'interpolate', 'rfm_dit').")
    parser.add_argument("--device", type=str, default=default_train.device,
                        help="Compute device.")
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

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing configurations and DataLoaders...")

    train_config = TrainConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_val=args.gradient_clip_val,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        log_interval=args.log_interval
    )
    set_seed(train_config.seed)

    checkpoint_dir = Path(args.checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        num_workers=0,
        max_mel_length=args.max_mel_length,
        shuffle=False,
        drop_last=False
    )
    val_loader = get_dataloader(val_data_config)

    if train_config.wandb_params.use_wandb:
        logger.info("Initializing Weights & Biases...")
        config = dataclasses.asdict(train_config) | dataclasses.asdict(train_data_config)
        wandb.init(
            project=train_config.wandb_params.project_name,
            config=config
        )

    logger.info(f"Train batches per epoch: {len(train_loader)}")
    logger.info(f"Val batches per epoch: {len(val_loader)}")

    model = get_model(train_config.model_name, train_config.model_params).to(train_config.device)
    optimizer = model.configure_optimizers(
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay
    )
    logger.info(f"Loaded model [{train_config.model_name}] and moved to {train_config.device}")

    scheduler = None
    global_step = 0
    if optimizer:
        total_steps = math.ceil(len(train_loader) / train_config.accumulation_steps) * train_config.epochs
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=train_config.warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - train_config.warmup_steps), eta_min=1e-6)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[train_config.warmup_steps]
        )
        logger.info("Using CosineAnnealingLR scheduler with linear warmup")

    # ==========================================
    # MAIN TRAINING LOOP
    # ==========================================
    logger.info("Starting training loop")
    history_train_loss = []
    history_val_loss = []
    plot_path = checkpoint_dir / "loss_curve.png"

    for epoch in range(1, train_config.epochs + 1):
        # --- TRAIN PHASE ---
        model.train()
        train_start_time = time.time()
        train_loss_accumulated = 0.0

        if optimizer:
            optimizer.zero_grad()

        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=True)
        for batch_idx, batch in enumerate(train_pbar):
            mel = batch['mel'].squeeze(1).to(train_config.device)
            mask_bool = batch['inpainting_mask'].to(train_config.device)
            mask_float = mask_bool.to(torch.float32)

            text_emb = batch['embedding'].to(train_config.device)
            text_mask = batch['text_padding_mask'].to(train_config.device)
            mel_pad_mask = batch['mel_padding_mask'].to(train_config.device)

            # --- RFM PREPARATION ---
            xt, target_v, t = prepare_rfm_batch(mel, mask_bool, train_config.device)

            # --- FORWARD PASS & LOSS ---
            v_pred = model(
                xt=xt,
                mask=mask_float,
                t=t,
                text_emb=text_emb,
                text_mask=text_mask,
                mel_pad_mask=mel_pad_mask
            )
            loss = F.mse_loss(v_pred, target_v, reduction='none')
            masked_loss = loss[mask_bool.expand_as(loss)].mean()

            is_accumulated = (batch_idx + 1) % train_config.accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(train_loader)

            if is_last_batch and not is_accumulated:
                effective_accum_steps = len(train_loader) % train_config.accumulation_steps
            else:
                effective_accum_steps = train_config.accumulation_steps

            normalized_loss = masked_loss / effective_accum_steps

            # --- BACKPROPAGATION ---
            if optimizer:
                normalized_loss.backward()

                is_accumulated = (batch_idx + 1) % train_config.accumulation_steps == 0
                is_last_batch = (batch_idx + 1) == len(train_loader)

                if is_accumulated or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.gradient_clip_val)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

            train_loss_accumulated += masked_loss.item()
            current_lr = scheduler.get_last_lr()[0] if scheduler else train_config.learning_rate
            train_pbar.set_postfix({"loss": f"{masked_loss.item():.4f}", "lr": f"{current_lr:.2e}"})

            if train_config.wandb_params.use_wandb and batch_idx % train_config.log_interval == 0:
                wandb.log(
                    {"train/batch_loss": masked_loss.item(), "train/learning_rate": current_lr},
                    step=global_step
                )

        train_time = time.time() - train_start_time
        avg_train_loss = train_loss_accumulated / len(train_loader)
        history_train_loss.append(avg_train_loss)

        logger.info(f"Epoch {epoch} Training Time: {train_time:.2f}s | Avg Train Loss: {avg_train_loss:.4f}")

        # --- VALIDATION PHASE ---
        model.eval()
        val_start_time = time.time()
        val_loss_accumulated = 0.0

        val_pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch}", leave=True)
        with torch.no_grad():
            torch.manual_seed(train_config.seed)
            for batch_idx, batch in enumerate(val_pbar):
                mel = batch['mel'].squeeze(1).to(train_config.device)
                mask_bool = batch['inpainting_mask'].to(train_config.device)
                mask_float = mask_bool.to(torch.float32)

                text_emb = batch['embedding'].to(train_config.device)
                text_mask = batch['text_padding_mask'].to(train_config.device)
                mel_pad_mask = batch['mel_padding_mask'].to(train_config.device)

                # --- RFM PREPARATION ---
                xt, target_v, t = prepare_rfm_batch(mel, mask_bool, train_config.device)

                # --- FORWARD PASS & LOSS ---
                v_pred = model(
                    xt=xt,
                    mask=mask_float,
                    t=t,
                    text_emb=text_emb,
                    text_mask=text_mask,
                    mel_pad_mask=mel_pad_mask
                )

                loss = F.mse_loss(v_pred, target_v, reduction='none')
                masked_loss = loss[mask_bool.expand_as(loss)].mean()

                val_loss_accumulated += masked_loss.item()
                val_pbar.set_postfix({"val_loss": f"{masked_loss.item():.4f}"})

        val_time = time.time() - val_start_time
        avg_val_loss = val_loss_accumulated / len(val_loader) if len(val_loader) > 0 else 0
        history_val_loss.append(avg_val_loss)

        if train_config.wandb_params.use_wandb:
            global_step += 1
            wandb.log(
                {"epoch": epoch, "train/epoch_loss": avg_train_loss, "val/epoch_loss": avg_val_loss},
                step=global_step
            )

        logger.info(f"Epoch {epoch} Validation Time: {val_time:.2f}s | Avg Val Loss: {avg_val_loss:.4f}")

        # --- SAVE CHECKPOINT ---
        if epoch % 5 == 0:
            ckpt_name = checkpoint_dir / f"{train_config.model_name}_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, ckpt_name)
            logger.info(f"Saved checkpoint: {ckpt_name}")

    logger.success("Training finished! 🎉🚀")
    if train_config.wandb_params.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
