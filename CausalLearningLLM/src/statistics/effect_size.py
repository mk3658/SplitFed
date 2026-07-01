"""Effect size measures."""
from src.statistics.tests import cohens_d
import numpy as np


def cliffs_delta(a, b) -> float:
    a, b = list(a), list(b)
    n_greater = sum(x > y for x in a for y in b)
    n_less = sum(x < y for x in a for y in b)
    return (n_greater - n_less) / (len(a) * len(b))
