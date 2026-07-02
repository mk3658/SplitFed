"""
Synthetic linear SCMs for ground-truth validation experiments.

Two SCM classes:

LinearSCM (original)
  X -> Z -> Y
  S -> Z via sensitive_direction (in same space as Z)
  Used for basic probe recovery tests.

LayerwiseSCM (new — fixes the SCM validation bug)
  Models the transformer computation Z_0 -> Z_l explicitly:

  Z_0 = input embeddings (lives in INPUT_SUBSPACE, first n_input_pcs dims)
        Z_0 correlates with S via v_s_input  (Path A: text->S correlation)

  Z_l = Z_0 + transformer_addition
      = Z_0 + gamma_s_layer * S * v_s_layer  +  gamma_c * Y * v_clinical  + noise
        where v_s_layer is ORTHOGONAL to INPUT_SUBSPACE (not in span(Z_0))

  Key property:
    residualise(Z_l, Z_0) removes INPUT_SUBSPACE component,
    leaving only v_s_layer signal  (Path B: transformer-built S encoding).

  This makes CRA's causal direction == v_s_layer after residualisation,
  while the correlational direction mixes v_s_input and v_s_layer.
"""

import numpy as np
from typing import Dict, Tuple


# ── original SCM (kept for backward compat) ──────────────────────────────────

class LinearSCM:
    """Simple linear SCM — all variables in one flat representation space."""

    def __init__(self, d: int = 64, seed: int = 42) -> None:
        self.d = d
        rng = np.random.RandomState(seed)

        self.clinical_direction = rng.randn(d)
        self.clinical_direction /= np.linalg.norm(self.clinical_direction)

        self.sensitive_direction = rng.randn(d)
        self.sensitive_direction /= np.linalg.norm(self.sensitive_direction)

        raw = rng.randn(d)
        raw -= raw @ self.clinical_direction * self.clinical_direction
        self.spurious_direction = raw / (np.linalg.norm(raw) + 1e-12)

    def generate(self, n: int, gamma_s: float = 0.0, gamma_c: float = 0.8,
                 noise_scale: float = 0.5, seed: int = 0) -> Dict:
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
            "clinical":  self.clinical_direction,
            "sensitive": self.sensitive_direction,
            "spurious":  self.spurious_direction,
        }


# ── layerwise SCM (fixes the causal direction recovery test) ─────────────────

class LayerwiseSCM:
    """
    SCM that explicitly models Z_0 (input embeddings) vs Z_l (layer l reps).

    Design
    ------
    d           : total representation dimension
    n_input_pcs : dimensionality of Z_0's span (the "input subspace")

    The input subspace is spanned by the first n_input_pcs standard basis
    vectors. Z_0 lives entirely in this subspace.

    v_s_input  : S direction IN the input subspace     (Path A target)
    v_s_layer  : S direction ORTHOGONAL to input subspace (Path B — CRA target)
    v_clinical : Y direction in input subspace (clinical content in text)

    After residualising Z_l on Z_0:
      - v_s_input component is removed  (blocked Path A)
      - v_s_layer component remains     (Path B, identified by CRA)

    This is the mathematically correct scenario for testing CRA's d-separation
    claim.  The SCM validation should now show causal direction recovery.
    """

    def __init__(self, d: int = 128, n_input_pcs: int = 20, seed: int = 42):
        self.d = d
        self.n_input_pcs = min(n_input_pcs, d // 2)
        rng = np.random.RandomState(seed)

        # Input subspace: first n_input_pcs standard basis vectors
        self.input_basis = np.eye(d)[:self.n_input_pcs]  # [k, d]

        # v_s_input: S direction at input level (IN span(Z_0))
        w = rng.randn(self.n_input_pcs)
        w /= np.linalg.norm(w)
        self.v_s_input = self.input_basis.T @ w            # [d], in input subspace

        # v_s_layer: S direction ORTHOGONAL to input subspace
        v_raw = rng.randn(d)
        for bv in self.input_basis:
            v_raw -= (v_raw @ bv) * bv                    # project out input dims
        self.v_s_layer = v_raw / (np.linalg.norm(v_raw) + 1e-12)   # [d]

        # v_clinical: clinical direction in input subspace (Y signal in text)
        w2 = rng.randn(self.n_input_pcs)
        w2 /= np.linalg.norm(w2)
        self.v_clinical = self.input_basis.T @ w2           # [d], in input subspace

        # Verify orthogonality
        assert abs(self.v_s_layer @ self.v_s_input) < 1e-8, "v_s_layer not orthog to v_s_input"
        assert abs(self.v_s_layer @ self.v_clinical) < 1e-8, "v_s_layer not orthog to v_clinical"

    def generate(
        self,
        n: int,
        gamma_s_input: float = 0.3,   # S->Z_0 strength (Path A)
        gamma_s_layer: float = 0.8,   # S->Z_l strength beyond Z_0 (Path B)
        gamma_c: float = 0.8,         # Y->Z_l strength (clinical)
        noise_scale: float = 0.2,
        seed: int = 0,
    ) -> Dict:
        """
        Generate (Z_0, Z_l, Y, S) tuples.

        Z_0 = gamma_s_input * S * v_s_input + clinical + noise  (in input subspace)
        Z_l = Z_0 + gamma_s_layer * S * v_s_layer + gamma_c * Y * v_clinical + noise
        """
        rng = np.random.RandomState(seed)
        Y = rng.randint(0, 2, n)
        S = rng.randint(0, 2, n)   # S independent of Y (pure spurious correlation)

        s_signal = (2 * S - 1).astype(float)   # {-1, +1}
        y_signal = (2 * Y - 1).astype(float)

        # Z_0: input embeddings — S encoded via v_s_input (Path A)
        Z_0 = (
            gamma_s_input * np.outer(s_signal, self.v_s_input)
            + gamma_c * np.outer(y_signal, self.v_clinical) * 0.5   # clinical in text
            + noise_scale * rng.randn(n, self.d)
        )
        # Project Z_0 to input subspace to enforce the "Z_0 lives in input subspace" property
        Z_0_proj = Z_0 @ self.input_basis.T @ self.input_basis

        # Z_l: transformer adds S in orthogonal direction (Path B)
        Z_l = (
            Z_0_proj
            + gamma_s_layer * np.outer(s_signal, self.v_s_layer)
            + gamma_c * np.outer(y_signal, self.v_clinical)          # stronger in Z_l
            + noise_scale * rng.randn(n, self.d)
        )

        return {"Z_0": Z_0_proj, "Z_l": Z_l, "Y": Y, "S": S}

    def get_true_directions(self) -> Dict[str, np.ndarray]:
        return {
            "v_s_input":  self.v_s_input,    # Path A direction (in Z_0)
            "v_s_layer":  self.v_s_layer,    # Path B direction (CRA target)
            "v_clinical": self.v_clinical,
        }

    def expected_causal_alignment(self) -> float:
        """Theoretical upper bound on cos(v_causal, v_s_layer) after perfect residualisation."""
        return 1.0   # v_s_layer is fully recoverable after blocking Path A
