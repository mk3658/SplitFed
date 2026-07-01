"""Statistical tests for CRA experiments."""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple


def mean_std(values: List[float]) -> Tuple[float, float]:
    a = np.array(values)
    return float(a.mean()), float(a.std())


def standard_error(values: List[float]) -> float:
    a = np.array(values)
    return float(a.std() / np.sqrt(len(a)))


def confidence_interval_95(values: List[float]) -> Tuple[float, float]:
    a = np.array(values, dtype=float)
    n = len(a)
    m = a.mean()
    se = a.std() / np.sqrt(n)
    t = stats.t.ppf(0.975, df=n - 1)
    return float(m - t * se), float(m + t * se)


def bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    rng = np.random.RandomState(seed)
    a = np.array(values, dtype=float)
    boot_means = [rng.choice(a, size=len(a), replace=True).mean() for _ in range(n_boot)]
    lo = (1 - ci) / 2
    return float(np.quantile(boot_means, lo)), float(np.quantile(boot_means, 1 - lo))


def paired_t_test(a: List[float], b: List[float]) -> Dict:
    t, p = stats.ttest_rel(a, b)
    d = cohens_d(a, b)
    return {"t_statistic": float(t), "p_value": float(p), "cohens_d": d}


def wilcoxon_signed_rank_test(a: List[float], b: List[float]) -> Dict:
    diff = np.array(a) - np.array(b)
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0}
    stat, p = stats.wilcoxon(diff)
    return {"statistic": float(stat), "p_value": float(p)}


def cohens_d(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2 + 1e-12)
    return float((a.mean() - b.mean()) / pooled_std)
