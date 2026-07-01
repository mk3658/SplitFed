"""Synthetic attribute injection and counterfactual generation.

Sensitive attributes (age, gender, hospital type) and spurious attributes
(note style, department) are injected into sample texts at controlled
correlation strengths, enabling downstream probing and CRA evaluation.
"""

from typing import Dict, List, Optional

import numpy as np

from src.data.base import ClinicalSample


# ---------------------------------------------------------------------------
# Attribute value banks
# ---------------------------------------------------------------------------

SENSITIVE_ATTR_VALUES: Dict[str, List[str]] = {
    "age_group": ["young", "middle", "older"],
    "gender": ["male", "female", "unknown"],
    "hospital_type": ["urban", "rural", "academic", "community"],
    "disease_group": ["cardiac", "respiratory", "neurological", "metabolic"],
}

SPURIOUS_ATTR_VALUES: Dict[str, List[str]] = {
    "note_style": ["formal", "informal", "structured"],
    "department": ["cardiology", "pulmonology", "neurology", "general"],
    "demographic_phrase": [
        "presented to clinic",
        "was seen at hospital",
        "admitted to ward",
    ],
    "source_template": ["template_A", "template_B", "template_C"],
    # hospital_type can act as either sensitive or spurious depending on context
    "hospital_type": ["urban", "rural", "academic", "community"],
}

# Phrase mappings used for text prepending / counterfactual replacement
AGE_PHRASES: Dict[str, str] = {
    "young": "a young",
    "middle": "a middle-aged",
    "older": "an elderly",
}
GENDER_PHRASES: Dict[str, str] = {
    "male": "male",
    "female": "female",
    "unknown": "",
}
HOSPITAL_PHRASES: Dict[str, str] = {
    "urban": "Urban Medical Center",
    "rural": "Rural Health Clinic",
    "academic": "Academic Medical Center",
    "community": "Community Hospital",
}
NOTE_STYLE_PHRASES: Dict[str, str] = {
    "formal": "Clinical note:",
    "informal": "Patient note:",
    "structured": "SOAP note:",
}


# ---------------------------------------------------------------------------
# Attribute injection
# ---------------------------------------------------------------------------


def inject_sensitive_attributes(
    samples: List[ClinicalSample],
    attributes: List[str],
    gamma: float = 0.0,
    mode: str = "independent",
    rng: Optional[np.random.RandomState] = None,
) -> List[ClinicalSample]:
    """Inject sensitive demographic attributes into sample texts.

    Parameters
    ----------
    samples:
        List of samples to modify *in place*.
    attributes:
        Names of sensitive attributes to inject (must be keys in
        ``SENSITIVE_ATTR_VALUES``).
    gamma:
        Correlation strength with the clinical label ``Y`` (0 = independent,
        1 = deterministic).  Used only when ``mode="correlated"``.
    mode:
        ``"independent"``  – attribute assigned uniformly at random.
        ``"correlated"``   – attribute correlated with label at strength γ.
    rng:
        Optional NumPy RandomState for reproducibility.

    Returns
    -------
    List[ClinicalSample]
        The same list with attributes injected (modified in place).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    for sample in samples:
        for attr in attributes:
            if attr not in SENSITIVE_ATTR_VALUES:
                continue
            vals = SENSITIVE_ATTR_VALUES[attr]
            n_vals = len(vals)

            if mode == "independent":
                val = vals[rng.randint(0, n_vals)]
            elif mode == "correlated":
                label = sample.target_label
                if rng.random() < gamma:
                    val = vals[label % n_vals]
                else:
                    val = vals[rng.randint(0, n_vals)]
            else:
                val = vals[rng.randint(0, n_vals)]

            sample.sensitive_attributes[attr] = val
            sample.input_text = _prepend_sensitive_attr(sample.input_text, attr, val)

    return samples


def inject_spurious_attributes(
    samples: List[ClinicalSample],
    attributes: List[str],
    gamma: float = 0.8,
    mode: str = "spurious_train",
    rng: Optional[np.random.RandomState] = None,
) -> List[ClinicalSample]:
    """Inject spurious contextual attributes into sample texts.

    Parameters
    ----------
    samples:
        List of samples to modify *in place*.
    attributes:
        Names of spurious attributes to inject.
    gamma:
        Spurious correlation strength.
    mode:
        ``"spurious_train"``        – correlated with Y (training regime).
        ``"spurious_test_reversed"``– reversed correlation (robustness test).
        ``"independent"``           – no correlation.

    Returns
    -------
    List[ClinicalSample]
        Modified in place.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    for sample in samples:
        for attr in attributes:
            if attr not in SPURIOUS_ATTR_VALUES:
                continue
            vals = SPURIOUS_ATTR_VALUES[attr]
            n_vals = len(vals)
            label = sample.target_label

            if mode == "spurious_train":
                if rng.random() < gamma:
                    val = vals[label % n_vals]
                else:
                    val = vals[rng.randint(0, n_vals)]
            elif mode == "spurious_test_reversed":
                reversed_label = (label + 1) % n_vals if n_vals == 2 else (label + 1) % n_vals
                if rng.random() < gamma:
                    val = vals[reversed_label % n_vals]
                else:
                    val = vals[rng.randint(0, n_vals)]
            elif mode == "independent":
                val = vals[rng.randint(0, n_vals)]
            else:
                val = vals[rng.randint(0, n_vals)]

            sample.spurious_attributes[attr] = val
            sample.input_text = _prepend_spurious_attr(sample.input_text, attr, val)

    return samples


# ---------------------------------------------------------------------------
# Counterfactual generation
# ---------------------------------------------------------------------------


def generate_counterfactuals(
    samples: List[ClinicalSample],
    cf_type: str = "sensitive",
    max_per_sample: int = 2,
) -> List[ClinicalSample]:
    """Attach counterfactual text variants to each sample.

    For each sample a small number of counterfactuals are generated by
    swapping one attribute value to an alternative.  The clinical label Y is
    unchanged — only the attribute phrase in the text is modified.

    Parameters
    ----------
    samples:
        List of ClinicalSample objects (modified in place).
    cf_type:
        ``"sensitive"`` or ``"spurious"``.
    max_per_sample:
        Maximum number of counterfactuals stored per sample.

    Returns
    -------
    List[ClinicalSample]
    """
    attr_map = (
        SENSITIVE_ATTR_VALUES if cf_type == "sensitive" else SPURIOUS_ATTR_VALUES
    )

    for sample in samples:
        attrs = (
            sample.sensitive_attributes
            if cf_type == "sensitive"
            else sample.spurious_attributes
        )
        cfs = []

        for attr, original_val in attrs.items():
            if attr not in attr_map:
                continue
            for new_val in attr_map[attr]:
                if new_val == original_val:
                    continue
                orig_phrase = _get_attr_phrase(attr, original_val, cf_type)
                new_phrase = _get_attr_phrase(attr, new_val, cf_type)
                cf_text = sample.input_text.replace(orig_phrase, new_phrase, 1)
                cfs.append(
                    {
                        "cf_type": cf_type,
                        "changed_attr": attr,
                        "original_val": original_val,
                        "new_val": new_val,
                        "input_text": cf_text,
                        "target_label": sample.target_label,
                    }
                )
                if len(cfs) >= max_per_sample:
                    break
            if len(cfs) >= max_per_sample:
                break

        sample.counterfactual_versions = cfs

    return samples


# ---------------------------------------------------------------------------
# Correlation statistics
# ---------------------------------------------------------------------------


def compute_empirical_correlation(
    attribute_vals: List[str],
    labels: List[int],
) -> tuple:
    """Compute Cramér's V between a categorical attribute and integer labels.

    Returns
    -------
    (cramers_v: float, p_value: float)
    """
    import pandas as pd
    from scipy.stats import chi2_contingency

    ct = pd.crosstab(pd.Series(attribute_vals), pd.Series(labels))
    chi2, p, dof, _ = chi2_contingency(ct)
    n = int(ct.values.sum())
    phi2 = chi2 / n
    r, k = ct.shape
    denom = min(k - 1, r - 1)
    cramers_v = float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0
    return cramers_v, float(p)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _prepend_sensitive_attr(text: str, attr: str, val: str) -> str:
    if attr == "age_group":
        phrase = AGE_PHRASES.get(val, "a")
        if phrase not in text[:60]:
            text = f"{phrase} patient case: {text}"
    elif attr == "gender":
        phrase = GENDER_PHRASES.get(val, "")
        if phrase and f"[{phrase}]" not in text[:60]:
            text = f"[{phrase}] {text}"
    elif attr == "hospital_type":
        phrase = HOSPITAL_PHRASES.get(val, "hospital")
        if phrase not in text[:100]:
            text = f"{phrase}: {text}"
    return text


def _prepend_spurious_attr(text: str, attr: str, val: str) -> str:
    if attr == "note_style":
        phrase = NOTE_STYLE_PHRASES.get(val, "Note:")
        if phrase not in text[:40]:
            text = f"{phrase} {text}"
    elif attr == "department":
        tag = f"[{val.title()} Dept]"
        if tag not in text[:80]:
            text = f"{tag} {text}"
    elif attr == "hospital_type":
        phrase = HOSPITAL_PHRASES.get(val, val)
        if phrase not in text[:100]:
            text = f"[{phrase}] {text}"
    elif attr == "source_template":
        tag = f"[{val}]"
        if tag not in text[:60]:
            text = f"{tag} {text}"
    return text


def _get_attr_phrase(attr: str, val: str, cf_type: str) -> str:
    """Return the canonical text phrase for an attribute value."""
    if cf_type == "sensitive":
        if attr == "age_group":
            return AGE_PHRASES.get(val, val)
        elif attr == "gender":
            return f"[{GENDER_PHRASES.get(val, val)}]"
        elif attr == "hospital_type":
            return HOSPITAL_PHRASES.get(val, val)
    else:
        if attr == "note_style":
            return NOTE_STYLE_PHRASES.get(val, val)
        elif attr == "department":
            return f"[{val.title()} Dept]"
    return val
