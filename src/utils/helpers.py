import random

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int = 42):
    """
    Ensures experiment reproducibility by setting the seed
    across all used libraries (Python, NumPy, PyTorch).

    :param seed: Seed for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to: {seed}")
