"""Text preprocessing utilities."""

import re
from typing import List
from src.data.base import ClinicalSample


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text: str, max_chars: int = 2000) -> str:
    return text[:max_chars] + "…" if len(text) > max_chars else text


def preprocess_samples(samples: List[ClinicalSample], max_length: int = 512) -> List[ClinicalSample]:
    for s in samples:
        s.input_text = clean_text(truncate_text(s.input_text, max_length * 4))
    return samples
