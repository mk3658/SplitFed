"""PubMedQA dataset loader."""

from __future__ import annotations

import logging
from typing import Optional

from src.data.base import BaseDataset, ClinicalSample

logger = logging.getLogger(__name__)

LABEL_MAP = {"yes": 1, "no": 0, "maybe": 2}


class PubMedQADataset(BaseDataset):
    def __init__(self, config, split: str = "train", max_samples: Optional[int] = None) -> None:
        super().__init__(config)
        self.split_name = split
        self.max_samples = max_samples

    def load(self) -> "PubMedQADataset":
        logger.info("Loading PubMedQA …")
        try:
            from datasets import load_dataset
            ds = load_dataset("bigbio/pubmed_qa", name="pubmed_qa_labeled_fold0_source",
                              trust_remote_code=True)
            split_key = "train" if self.split_name in ("train", "all") else self.split_name
            if split_key not in ds:
                split_key = list(ds.keys())[0]
            data = ds[split_key]

            for i, item in enumerate(data):
                if self.max_samples and i >= self.max_samples:
                    break
                ctx = item.get("context", {})
                if isinstance(ctx, dict):
                    context = " ".join(ctx.get("contexts", []))
                else:
                    context = str(ctx)
                question = item.get("question", "")
                answer = str(item.get("final_decision", item.get("answer", "no"))).lower()
                if answer not in LABEL_MAP:
                    answer = "no"
                label = LABEL_MAP[answer]
                input_text = (
                    f"Context: {context[:400]}\nQuestion: {question}\n"
                    f"Answer (yes, no, or maybe):"
                )
                self.samples.append(ClinicalSample(
                    sample_id=f"pmqa_{i:06d}",
                    input_text=input_text,
                    question=question,
                    context=context[:400],
                    target_label=label,
                    target_answer=answer,
                    answer_choices=["yes", "no", "maybe"],
                    metadata={"source": "pubmedqa"},
                    split="train",
                ))

            logger.info("Loaded %d PubMedQA samples.", len(self.samples))
        except Exception as e:
            logger.warning("PubMedQA load failed (%s). Falling back to synthetic.", e)
            from src.data.synthetic_clinical import SyntheticClinicalDataset
            fallback = SyntheticClinicalDataset(self.config, n_samples=self.max_samples or 100)
            fallback.load()
            self.samples = fallback.samples

        return self
