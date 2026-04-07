from loguru import logger
import torch.nn as nn

from src.config.config import ModelConfig
from src.model.dit import DiTModel
from src.model.aligned_dit import AlignedDiTModel


def get_model(model_name: str, model_config: ModelConfig) -> nn.Module:
    """
    Factory function to instantiate the chosen model architecture.

    :param model_name: Name of the model (e.g., 'rfm_dit', 'aligned_dit').
    :param model_config: Model config.

    :return: An initialized PyTorch nn.Module.
    """
    logger.info(f"Instantiating model architecture: {model_name}")

    match model_name.lower():
        case "rfm_dit":
           return DiTModel(model_config)
        case "aligned_dit":
           return AlignedDiTModel(model_config)
        case _:
            raise ValueError(f"Unknown model name: {model_name}. Please check your config.")
