"""Synthetic clinical QA dataset with known ground-truth factor structure."""

import random
from typing import List, Optional

import numpy as np

from src.data.base import BaseDataset, ClinicalSample


# ---------------------------------------------------------------------------
# Template bank
# ---------------------------------------------------------------------------

CLINICAL_TEMPLATES = [
    (
        "A {age} {gender} patient presents with {symptom}. "
        "The {test} shows {finding}. "
        "Question: Is this consistent with {condition}?",
        "yes_no",
    ),
    (
        "Patient history: {age} {gender} with {chronic}. "
        "Current complaint: {symptom}. "
        "Based on the presentation, what is the most likely diagnosis?",
        "diagnosis",
    ),
    (
        "A {age} {gender} with {symptom} and {finding}. "
        "Labs show {lab}. "
        "Question: Does this patient require immediate intervention?",
        "yes_no",
    ),
    (
        "Chief complaint: {symptom}. "
        "Physical exam: {finding}. "
        "Patient is a {age} {gender}. "
        "Is {condition} a likely diagnosis?",
        "yes_no",
    ),
]

SYMPTOMS = [
    "chest pain",
    "shortness of breath",
    "fatigue",
    "headache",
    "abdominal pain",
    "joint pain",
    "dizziness",
    "palpitations",
    "peripheral oedema",
    "night sweats",
]

CONDITIONS = [
    "a cardiac event",
    "a respiratory infection",
    "a neurological disorder",
    "metabolic syndrome",
    "an autoimmune condition",
]

TESTS = ["ECG", "chest X-ray", "MRI", "blood work", "CT scan", "echocardiogram"]

FINDINGS_POSITIVE = [
    "abnormal results",
    "elevated inflammatory markers",
    "reduced ejection fraction",
    "new focal lesion",
]
FINDINGS_NEGATIVE = [
    "normal results",
    "no acute findings",
    "borderline findings within normal limits",
    "unremarkable study",
]

LABS_POSITIVE = ["elevated troponin", "raised CRP", "high D-dimer", "low haemoglobin"]
LABS_NEGATIVE = ["all values within normal range", "WBC normal", "electrolytes balanced"]

CHRONIC_CONDITIONS = [
    "hypertension",
    "type 2 diabetes",
    "COPD",
    "chronic kidney disease",
    "heart failure",
]

AGE_TOKENS = ["young", "middle-aged", "elderly"]
GENDER_TOKENS = ["male", "female", "non-binary"]


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------


class SyntheticClinicalDataset(BaseDataset):
    """Synthetic clinical QA dataset with known ground-truth causal structure.

    The dataset is generated from simple templates so that the true clinical
    direction (label Y), sensitive direction (S), and spurious direction (U)
    are explicitly controlled.  This enables quantitative validation of the
    CRA probes against the ground-truth factors.

    Parameters
    ----------
    config:
        Experiment configuration object.
    n_samples:
        Number of samples to generate.
    seed:
        Random seed for reproducibility.
    """

    def __init__(self, config, n_samples: int = 100, seed: int = 42) -> None:
        super().__init__(config)
        self.n_samples = n_samples
        self.seed = seed

        # Ground-truth direction vectors (set during / after load for validation)
        self.true_clinical_direction: Optional[np.ndarray] = None
        self.true_sensitive_direction: Optional[np.ndarray] = None
        self.true_spurious_direction: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> "SyntheticClinicalDataset":
        rng = np.random.RandomState(self.seed)
        random.seed(self.seed)

        for i in range(self.n_samples):
            # ---- Ground-truth clinical label --------------------------------
            label = int(rng.randint(0, 2))

            # ---- Sample template and fill -----------------------------------
            template, q_type = random.choice(CLINICAL_TEMPLATES)

            age = random.choice(AGE_TOKENS)
            gender = random.choice(GENDER_TOKENS)
            symptom = random.choice(SYMPTOMS)
            condition = random.choice(CONDITIONS)
            test = random.choice(TESTS)
            chronic = random.choice(CHRONIC_CONDITIONS)

            # Finding and lab correlate with clinical label
            finding = (
                random.choice(FINDINGS_POSITIVE)
                if label == 1
                else random.choice(FINDINGS_NEGATIVE)
            )
            lab = (
                random.choice(LABS_POSITIVE)
                if label == 1
                else random.choice(LABS_NEGATIVE)
            )

            input_text = template.format(
                age=age,
                gender=gender,
                symptom=symptom,
                condition=condition,
                test=test,
                finding=finding,
                chronic=chronic,
                lab=lab,
            )

            target_answer = "yes" if label == 1 else "no"
            if q_type == "diagnosis":
                target_answer = condition

            sample = ClinicalSample(
                sample_id=f"syn_{i:04d}",
                input_text=input_text,
                question=input_text,
                target_label=label,
                target_answer=target_answer,
                answer_choices=["yes", "no"],
                metadata={
                    "source": "synthetic",
                    "symptom": symptom,
                    "condition": condition,
                    "finding": finding,
                    "age_token": age,
                    "gender_token": gender,
                    "q_type": q_type,
                },
                split="train",
            )
            self.samples.append(sample)

        return self
