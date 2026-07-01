"""LaTeX and CSV table generation."""

from __future__ import annotations

import os
import pandas as pd
from typing import Dict, List, Optional


def df_to_latex(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "",
    float_fmt: str = "{:.3f}",
) -> str:
    """Convert DataFrame to LaTeX tabular string."""
    col_fmt = "l" + "c" * (len(df.columns) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\hline",
        " & ".join(str(c).replace("_", " ").title() for c in df.columns) + r" \\",
        r"\hline",
    ]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save_table(
    df: pd.DataFrame,
    output_dir: str,
    name: str,
    caption: str = "",
    label: str = "",
) -> None:
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    csv_path = os.path.join(output_dir, "tables", f"{name}.csv")
    tex_path = os.path.join(output_dir, "tables", f"{name}.tex")
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(df_to_latex(df, caption=caption or name, label=label or name))


def generate_main_results_table(
    intervention_df: pd.DataFrame,
    output_dir: str,
) -> Optional[pd.DataFrame]:
    if intervention_df.empty:
        return None

    cols = [
        "direction_type", "pre_clinical_acc", "pre_task_loss",
        "pre_sensitive_leakage", "pre_spurious_score",
        "delta_leak", "delta_spur", "privacy_utility_score",
        "robustness_utility_score", "cra_score",
    ]
    available = [c for c in cols if c in intervention_df.columns]
    df = (
        intervention_df[available]
        .groupby("direction_type")
        .mean()
        .reset_index()
    )
    save_table(df, output_dir, "table1_main_results",
               caption="Main Results: CRA vs Baselines", label="tab:main_results")
    return df


def generate_layer_table(
    layer_info_df: pd.DataFrame,
    output_dir: str,
) -> Optional[pd.DataFrame]:
    if layer_info_df.empty:
        return None
    save_table(layer_info_df, output_dir, "table2_layer_analysis",
               caption="Layer-wise Representation Analysis", label="tab:layer_analysis")
    return layer_info_df


def generate_taxonomy_table(
    taxonomy_df: pd.DataFrame,
    output_dir: str,
) -> Optional[pd.DataFrame]:
    if taxonomy_df.empty:
        return None
    cols = ["layer", "direction_type", "clinical_score", "sensitive_score",
            "spurious_score", "cos_y_s", "taxonomy_label"]
    available = [c for c in cols if c in taxonomy_df.columns]
    df = taxonomy_df[available].drop_duplicates(subset=["layer", "direction_type"])
    save_table(df, output_dir, "table4_taxonomy",
               caption="Representation Taxonomy", label="tab:taxonomy")
    return df
