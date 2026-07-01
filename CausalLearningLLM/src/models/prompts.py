"""Prompt templates for clinical QA datasets."""

from typing import List, Optional

PUBMEDQA_TEMPLATE = (
    "Context: {context}\n\nQuestion: {question}\n\nAnswer (yes, no, or maybe):"
)
MEDQA_TEMPLATE = "Question: {question}\n\nOptions:\n{options}\n\nThe best answer is:"
SIMPLE_QA_TEMPLATE = "{input_text}\n\nAnswer:"


def format_pubmedqa(context: str, question: str) -> str:
    return PUBMEDQA_TEMPLATE.format(context=context[:500], question=question)


def format_medqa(question: str, choices: List[str]) -> str:
    options = "\n".join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
    return MEDQA_TEMPLATE.format(question=question, options=options)


def format_sample(sample) -> str:
    """Format a ClinicalSample for model input."""
    if getattr(sample, "question", None) and getattr(sample, "context", None):
        return format_pubmedqa(sample.context, sample.question)
    return sample.input_text
