import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

from src.model.ema import ModelEMA


class EMACallback(Callback):
    """Updates the Exponential Moving Average of the model weights."""
    def __init__(self, decay: float = 0.9999, update_every: int = 1):
        super().__init__()
        self.decay = decay
        self.update_every = update_every

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Initializes the EMA model at the start of training.

        :param trainer: The current trainer.
        :param pl_module: The current pl module.
        """
        ema_model: ModelEMA = ModelEMA(pl_module.model, decay=self.decay)
        pl_module.ema_model = ema_model

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        """
        Updates the EMA model after each training batch.

        :param trainer: The current trainer.
        :param pl_module: The current pl module.
        :param outputs: The current outputs.
        :param batch: The current batch.
        :param batch_idx: The current batch index.
        """
        if trainer.global_step % self.update_every == 0:
            pl_module.ema_model.update(pl_module.model)

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict):
        """Saves the EMA weights into the Lightning checkpoint."""
        if getattr(pl_module, "ema_model", None) is not None:
            checkpoint["ema_model_state_dict"] = pl_module.ema_model.ema_model.state_dict()

    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict):
        """Extracts the EMA weights from the Lightning checkpoint."""
        if "ema_model_state_dict" in checkpoint:
            self._ema_state_dict_to_load = checkpoint["ema_model_state_dict"]
