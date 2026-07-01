"""Synthetic linear SCM for ground-truth validation experiments."""

import numpy as np
from typing import Dict


class LinearSCM:
    """
    Linear SCM with known causal directions.

    X -> Z -> Y_hat
    S -> Z (sensitive, correlated gamma_s with Y)
    C -> Z (spurious, correlated gamma_c with Y in train, reversed in test)
    """

    def __init__(self, d: int = 64, seed: int = 42) -> None:
        self.d = d
        rng = np.random.RandomState(seed)

        self.clinical_direction = rng.randn(d)
        self.clinical_direction /= np.linalg.norm(self.clinical_direction)

        self.sensitive_direction = rng.randn(d)
        self.sensitive_direction /= np.linalg.norm(self.sensitive_direction)

        # Orthogonalise spurious wrt clinical
        raw = rng.randn(d)
        raw -= raw @ self.clinical_direction * self.clinical_direction
        self.spurious_direction = raw / (np.linalg.norm(raw) + 1e-12)

    def generate(
        self,
        n: int,
        gamma_s: float = 0.0,
        gamma_c: float = 0.8,
        noise_scale: float = 0.5,
        seed: int = 0,
    ) -> Dict:
        rng = np.random.RandomState(seed)
        Y = rng.randint(0, 2, n)
        S = (rng.random(n) < (gamma_s * Y + (1 - gamma_s) * 0.5)).astype(int)
        C = (rng.random(n) < (gamma_c * Y + (1 - gamma_c) * 0.5)).astype(int)

        Z = (
            np.outer(2 * Y - 1, self.clinical_direction) * 2.0
            + np.outer(2 * S - 1, self.sensitive_direction) * 1.0
            + np.outer(2 * C - 1, self.spurious_direction) * 1.5
            + noise_scale * rng.randn(n, self.d)
        )
        return {"Z": Z, "Y": Y, "S": S, "C": C}

    def get_true_directions(self) -> Dict[str, np.ndarray]:
        return {
            "clinical": self.clinical_direction,
            "sensitive": self.sensitive_direction,
            "spurious": self.spurious_direction,
        }
