"""Answer scoring helpers."""

from typing import Dict, List
import numpy as np


def compute_accuracy(predictions: List, labels: List) -> float:
    if not labels:
        return 0.0
    return sum(p == l for p, l in zip(predictions, labels)) / len(labels)


def score_predictions(
    predicted_answers: List[str],
    target_answers: List[str],
    answer_probs: np.ndarray,
    answer_choices: List[str],
) -> Dict:
    labels = [answer_choices.index(t) if t in answer_choices else 0 for t in target_answers]
    pred_labels = [answer_choices.index(p) if p in answer_choices else 0 for p in predicted_answers]
    accuracy = compute_accuracy(pred_labels, labels)
    correct_probs = [answer_probs[i, labels[i]] for i in range(len(labels))]
    return {
        "accuracy": accuracy,
        "mean_correct_prob": float(np.mean(correct_probs)),
        "pred_labels": pred_labels,
        "true_labels": labels,
    }
