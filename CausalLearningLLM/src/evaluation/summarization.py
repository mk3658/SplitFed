"""Result summarization utilities."""

import json
import os
import pandas as pd
from typing import Dict


def summarize_results(results: Dict, output_dir: str) -> Dict:
    summary = {}

    for name, val in results.items():
        if isinstance(val, pd.DataFrame) and not val.empty:
            summary[name] = {
                "n_rows": len(val),
                "columns": list(val.columns),
            }
            numeric = val.select_dtypes("number")
            if not numeric.empty:
                summary[name]["means"] = numeric.mean().to_dict()

    path = os.path.join(output_dir, "metrics_summary.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
