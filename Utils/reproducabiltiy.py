import random
import numpy as np
import torch
from typing import Optional

def seed_everything(seed: int) -> None:
    """
    Seed all relevant random number generators for reproducibility.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior in cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """
    Initialize each DataLoader worker with a different but deterministic seed.
    
    Args:
        worker_id (int): The worker ID provided by DataLoader.
        base_seed (Optional[int]): Base seed to offset the worker seed. If None, use a default.
    """
    if base_seed is None:
        base_seed = 42  # or any default seed you prefer

    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
