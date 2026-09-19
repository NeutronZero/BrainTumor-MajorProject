"""Deterministic seeding — §28. Bit-exact on same HW/SW, not across HW."""

from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def worker_init_fn(worker_id: int, base_seed: int = 42):
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def make_generator(base_seed: int = 42):
    try:
        import torch

        return torch.Generator().manual_seed(base_seed)
    except ImportError:
        return None
