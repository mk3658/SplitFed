"""MIMIC-IV-Note placeholder dataset loader.

Requires institutional data use agreement.
See: https://physionet.org/content/mimic-iv-note/
"""

import logging
from src.data.base import BaseDataset

logger = logging.getLogger(__name__)


class MIMICPlaceholderDataset(BaseDataset):
    def load(self) -> "MIMICPlaceholderDataset":
        logger.warning(
            "MIMIC-IV-Note requires a signed DUA from PhysioNet. "
            "Download data to data/raw/mimic/ and implement a real loader. "
            "Falling back to synthetic data."
        )
        from src.data.synthetic_clinical import SyntheticClinicalDataset
        n = self.config.get("dataset", {}).get("max_samples", 100)
        fallback = SyntheticClinicalDataset(self.config, n_samples=n)
        fallback.load()
        self.samples = fallback.samples
        return self
