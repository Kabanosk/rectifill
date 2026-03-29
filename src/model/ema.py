import copy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Exponential Moving Average of model parameters.

        :param model: PyTorch model
        :param decay: Exponential Moving Average decay factor
        """
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Smoothly updates the EMA model weights based on the active model.

        :param model: PyTorch model
        """
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)
