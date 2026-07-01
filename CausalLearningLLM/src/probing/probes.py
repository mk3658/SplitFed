"""Probe models: logistic regression and SVM variants."""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)


class LogisticRegressionProbe:
    """Linear probe using logistic regression."""

    def __init__(self, C: float = 1.0, max_iter: int = 500, random_state: int = 42) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.clf: Optional[LogisticRegression] = None
        self.label_encoder = LabelEncoder()
        self.direction_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionProbe":
        y_enc = self.label_encoder.fit_transform(y)
        self.clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.clf.fit(X, y_enc)
        W = self.clf.coef_  # [n_classes, d]
        if W.shape[0] == 1:
            self.direction_ = W[0] / (np.linalg.norm(W[0]) + 1e-12)
        else:
            _, _, Vt = np.linalg.svd(W, full_matrices=False)
            self.direction_ = Vt[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        enc = self.clf.predict(X)
        return self.label_encoder.inverse_transform(enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict:
        y_enc = self.label_encoder.transform(y)
        preds_enc = self.clf.predict(X)
        proba = self.clf.predict_proba(X)
        n_classes = len(self.label_encoder.classes_)

        acc = accuracy_score(y_enc, preds_enc)
        bal_acc = balanced_accuracy_score(y_enc, preds_enc)
        f1 = f1_score(y_enc, preds_enc, average="macro", zero_division=0)
        ce = log_loss(y_enc, proba, labels=list(range(len(self.label_encoder.classes_))))

        try:
            if n_classes == 2:
                auroc = roc_auc_score(y_enc, proba[:, 1])
            else:
                auroc = roc_auc_score(y_enc, proba, multi_class="ovr", average="macro")
        except Exception:
            auroc = 0.5

        return {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(f1),
            "auroc": float(auroc),
            "cross_entropy": float(ce),
        }

    def get_direction(self) -> Optional[np.ndarray]:
        return self.direction_

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "LogisticRegressionProbe":
        with open(path, "rb") as f:
            return pickle.load(f)


class LinearSVMProbe:
    """Linear SVM probe."""

    def __init__(self, C: float = 1.0, max_iter: int = 2000) -> None:
        self.C = C
        self.max_iter = max_iter
        self.clf: Optional[LinearSVC] = None
        self.label_encoder = LabelEncoder()
        self.direction_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMProbe":
        y_enc = self.label_encoder.fit_transform(y)
        self.clf = LinearSVC(C=self.C, max_iter=self.max_iter, class_weight="balanced")
        self.clf.fit(X, y_enc)
        W = self.clf.coef_
        if W.shape[0] == 1:
            self.direction_ = W[0] / (np.linalg.norm(W[0]) + 1e-12)
        else:
            _, _, Vt = np.linalg.svd(W, full_matrices=False)
            self.direction_ = Vt[0]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(self.clf.predict(X))

    def get_direction(self) -> Optional[np.ndarray]:
        return self.direction_
