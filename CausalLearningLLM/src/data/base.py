"""Base data structures shared across all dataset loaders."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClinicalSample:
    """A single clinical QA sample with associated metadata.

    Attributes
    ----------
    sample_id:
        Unique string identifier.
    input_text:
        Full model input (may include prepended attribute phrases).
    target_label:
        Integer class label (e.g., 0 = no, 1 = yes).
    question:
        The original clinical question (before any augmentation).
    context:
        Supporting clinical context / passage.
    target_answer:
        String form of the answer (e.g., ``"yes"``, ``"no"``, ``"maybe"``).
    answer_choices:
        List of candidate answer strings.
    sensitive_attributes:
        Dict mapping attribute name → value (e.g. ``{"gender": "female"}``).
    spurious_attributes:
        Dict mapping spurious feature name → value.
    counterfactual_versions:
        Optional list of counterfactual dicts.  Each dict records which
        attribute was changed and what the resulting text looks like.
    metadata:
        Catch-all dict for source-specific extra fields.
    split:
        Dataset split assignment: ``"train"``, ``"val"``, or ``"test"``.
    """

    sample_id: str
    input_text: str
    target_label: int
    question: Optional[str] = None
    context: Optional[str] = None
    target_answer: Optional[str] = None
    answer_choices: Optional[List[str]] = None
    sensitive_attributes: Dict[str, str] = field(default_factory=dict)
    spurious_attributes: Dict[str, str] = field(default_factory=dict)
    counterfactual_versions: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    split: str = "train"


class BaseDataset:
    """Abstract base for all CRA dataset loaders.

    Subclasses must implement :meth:`load` which populates ``self.samples``.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self.samples: List[ClinicalSample] = []

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def load(self) -> "BaseDataset":
        raise NotImplementedError(
            f"{type(self).__name__} must implement load()."
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_split(self, split: str) -> List[ClinicalSample]:
        """Return all samples assigned to *split*."""
        return [s for s in self.samples if s.split == split]

    def get_labels(self, split: Optional[str] = None) -> List[int]:
        """Return all integer labels, optionally filtered by *split*."""
        samples = self.get_split(split) if split else self.samples
        return [s.target_label for s in samples]

    def label_set(self) -> List[int]:
        """Return the sorted set of unique integer labels."""
        return sorted(set(s.target_label for s in self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> ClinicalSample:
        return self.samples[idx]

    def __repr__(self) -> str:
        splits = {}
        for s in self.samples:
            splits[s.split] = splits.get(s.split, 0) + 1
        return (
            f"{type(self).__name__}("
            f"total={len(self.samples)}, "
            f"splits={splits})"
        )
