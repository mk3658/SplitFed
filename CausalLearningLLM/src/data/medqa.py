"""MedQA dataset loader (multiple-choice format)."""

from __future__ import annotations

import logging
from typing import Optional

from src.data.base import BaseDataset, ClinicalSample

logger = logging.getLogger(__name__)


class MedQADataset(BaseDataset):
    def __init__(self, config, max_samples: Optional[int] = None) -> None:
        super().__init__(config)
        self.max_samples = max_samples

    def load(self) -> "MedQADataset":
        logger.info("Loading MedQA …")
        try:
            from datasets import load_dataset
            ds = load_dataset("GBaker/MedQA-USMLE-4-options", trust_remote_code=True)
            split_key = "train" if "train" in ds else list(ds.keys())[0]
            data = ds[split_key]

            for i, item in enumerate(data):
                if self.max_samples and i >= self.max_samples:
                    break
                question = item.get("question", "")
                options = item.get("options", {})
                if isinstance(options, dict):
                    choices = list(options.values())
                    choice_keys = list(options.keys())
                else:
                    choices = list(options)
                    choice_keys = [chr(65 + j) for j in range(len(choices))]

                answer_key = str(item.get("answer_idx", item.get("answer", "A")))
                try:
                    label = choice_keys.index(answer_key) if answer_key in choice_keys else 0
                except Exception:
                    label = 0

                opts_str = "\n".join(f"({k}) {v}" for k, v in zip(choice_keys, choices))
                input_text = f"Question: {question}\n\nOptions:\n{opts_str}\n\nThe best answer is:"

                self.samples.append(ClinicalSample(
                    sample_id=f"medqa_{i:06d}",
                    input_text=input_text,
                    question=question,
                    target_label=label,
                    target_answer=answer_key,
                    answer_choices=choices,
                    metadata={"source": "medqa"},
                    split="train",
                ))

        except Exception as e:
            logger.warning("MedQA load failed (%s). Falling back to synthetic.", e)
            from src.data.synthetic_clinical import SyntheticClinicalDataset
            fallback = SyntheticClinicalDataset(self.config, n_samples=self.max_samples or 100)
            fallback.load()
            self.samples = fallback.samples

        logger.info("Loaded %d MedQA samples.", len(self.samples))
        return self
