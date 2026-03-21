from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


class BaseModel(nn.Module, ABC):
    """
    Abstract Base Class for all audio inpainting models in this project.
    Forces the implementation of forward pass and optimizer configuration.
    """

    @abstractmethod
    def forward(self, mel: torch.Tensor, inpainting_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        pass

    @abstractmethod
    def configure_optimizers(self, **kwargs) -> Optional[Optimizer]:
        """
        Returns the optimizer tailored for this specific model,
        or None if the model has no learnable parameters.
        """
        pass
