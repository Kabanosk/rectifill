import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.utils.rfm import prepare_rfm_batch, sample_euler
from src.evaluation.metrics import calculate_lsd
from src.data.utils import denormalize_mel, normalize_mel
from src.config.config import TrainConfig


class LitRFM(pl.LightningModule):
    """
    PyTorch Lightning Module encapsulating the Rectified Flow Matching logic.
    """

    def __init__(self, core_model: torch.nn.Module, config: TrainConfig, steps_per_epoch: int):
        super().__init__()
        self.model = core_model
        self.config = config
        self.steps_per_epoch = steps_per_epoch
        self.save_hyperparameters(ignore=['core_model'])
        self.ema_model = None  # Populated by EMACallback

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        mel_raw = batch['mel'].squeeze(1)
        mel = normalize_mel(mel_raw)
        mask_bool = batch['inpainting_mask']
        mask_float = mask_bool.to(torch.float32)

        condition_kwargs = {
            "mel_pad_mask": batch.get('mel_padding_mask'),
            "text_mask": batch.get('text_padding_mask'),
        }
        if 'durations' in batch:
            condition_kwargs['durations'] = batch['durations']

        text_emb = batch['embedding']

        # CFG Dropout
        if self.config.cfg_prob > 0.0:
            drop_mask = torch.rand(text_emb.shape[0], 1, 1, device=self.device) < self.config.cfg_prob
            text_emb = torch.where(drop_mask, torch.zeros_like(text_emb), text_emb)

        condition_kwargs["text_emb"] = text_emb

        xt, target_v, t = prepare_rfm_batch(mel, mask_bool, self.device)

        v_pred = self.model(xt=xt, mask=mask_float, t=t, **condition_kwargs)

        loss = F.mse_loss(v_pred, target_v, reduction='mean')

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        eval_model = self.ema_model.ema_model if self.ema_model else self.model

        mel_raw = batch['mel'].squeeze(1)
        mel = normalize_mel(mel_raw)
        mask_bool = batch['inpainting_mask']
        mask_float = mask_bool.to(torch.float32)

        condition_kwargs = {
            "text_emb": batch['embedding'],
            "mel_pad_mask": batch.get('mel_padding_mask'),
            "text_mask": batch.get('text_padding_mask'),
        }
        if 'durations' in batch:
            condition_kwargs['durations'] = batch['durations']

        xt, target_v, t = prepare_rfm_batch(mel, mask_bool, self.device)

        v_pred = eval_model(xt=xt, mask=mask_float, t=t, **condition_kwargs)

        loss = F.mse_loss(v_pred, target_v, reduction='none')
        masked_loss = loss[mask_bool.expand_as(loss)].mean()
        self.log("val/epoch_loss", masked_loss, prog_bar=True, sync_dist=True, batch_size=len(batch))

        if batch_idx < self.config.validation_metrics_steps:
            generated_mel_norm = sample_euler(
                model=eval_model,
                x1_context=mel,
                mask_bool=mask_bool,
                num_steps=50,
                cfg_scale=self.config.cfg_scale,
                **condition_kwargs
            )
            generated_mel = denormalize_mel(generated_mel_norm)
            batch_lsd = calculate_lsd(generated_mel, mel_raw, mask_bool)
            self.log("val/epoch_lsd", batch_lsd, sync_dist=True, prog_bar=True, batch_size=len(batch))

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        total_steps = self.steps_per_epoch * self.trainer.max_epochs
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.config.warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=(total_steps - self.config.warmup_steps),
                                   eta_min=self.config.eta_min)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[self.config.warmup_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
