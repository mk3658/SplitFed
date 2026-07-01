"""Orthogonality analysis between representation directions."""

import numpy as np
from typing import Dict, List
from src.representations.geometry import cosine_similarity


def pairwise_cosines(directions: Dict[str, np.ndarray]) -> Dict:
    names = list(directions.keys())
    result = {}
    for i, n1 in enumerate(names):
        for n2 in names[i:]:
            result[(n1, n2)] = cosine_similarity(directions[n1], directions[n2])
    return result


def orthogonality_score(v1: np.ndarray, v2: np.ndarray) -> float:
    """1 - |cos(v1, v2)| => 1 is perfectly orthogonal."""
    return 1.0 - abs(cosine_similarity(v1, v2))
