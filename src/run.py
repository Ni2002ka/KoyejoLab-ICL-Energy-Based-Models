import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn
from typing import Any, Dict, Union


def set_seed(seed: int) -> torch.Generator:
    # Try to make this implementation as deterministic as possible.
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    # pl.utilities.seed_everything(seed=seed)
    return generator