"""Dataset split assignment."""

import numpy as np
from typing import List, Tuple
from src.data.base import ClinicalSample


def assign_splits(
    samples: List[ClinicalSample],
    ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> List[ClinicalSample]:
    rng = np.random.RandomState(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n = len(samples)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    for i, j in enumerate(idx):
        if i < n_train:
            samples[j].split = "train"
        elif i < n_train + n_val:
            samples[j].split = "val"
        else:
            samples[j].split = "test"
    return samples
