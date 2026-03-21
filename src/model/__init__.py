import torch.nn as nn
from loguru import logger

from src.model.base import BaseModel
from src.model.interpolate import InterpolationBaseline


def get_model(model_name: str) -> BaseModel:
    """
    Factory function to instantiate the chosen model architecture.

    :param model_name: Name of the model (e.g., 'interpolate', 'rfm_dit').
    :return: An initialized PyTorch nn.Module.
    """
    logger.info(f"Instantiating model architecture: {model_name}")

    match model_name.lower():
        case "interpolate":
            return InterpolationBaseline()

        case "rfm_dit":
           raise NotImplementedError("For now model is not implemented.")
        case _:
            raise ValueError(f"Unknown model name: {model_name}. Please check your config.")