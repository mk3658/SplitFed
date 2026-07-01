"""Spurious probe training helper."""
from src.probing.probes import LogisticRegressionProbe


def train_spurious_probe(Z, c_labels, C=1.0, max_iter=500):
    probe = LogisticRegressionProbe(C=C, max_iter=max_iter)
    probe.fit(Z, c_labels)
    return probe
