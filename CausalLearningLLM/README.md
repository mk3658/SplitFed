# CRA: Causal Representation Attribution for Clinical LLMs

**Paper:** *"Understanding Clinical Reasoning through Causal Representation Attribution in Large Language Models"*

---

## What is CRA?

Clinical LLMs encode multiple types of information in their hidden states across layers. Standard probing and attribution methods reveal what is **decodable** — but not what is **causally influential**.

CRA (Causal Representation Attribution) answers three questions simultaneously:

1. Which hidden representations **causally support clinical reasoning**?
2. Which representations **encode privacy-sensitive patient attributes**?
3. Which representations encode **spurious shortcuts** induced by dataset bias?

CRA combines:
- **Probing** — to identify what information is linearly decodable
- **Representation intervention** — to estimate causal influence via projection/removal/amplification
- **Counterfactual analysis** — using paired (x, x_cf) examples where sensitive/spurious attributes change but clinical content is preserved
- **Causal mediation analysis** — to decompose total effects into direct and indirect components
- **Representation taxonomy** — classifying layers into: *clinical*, *privacy-sensitive*, *spurious*, *mixed*, or *nuisance*

---

## Why this is not just a privacy/debiasing paper

- **Not just privacy**: CRA estimates *causal* effects on privacy leakage, not just decodability. It distinguishes between sensitive information that is necessary for clinical reasoning vs. information that leaks without clinical benefit.
- **Not just debiasing**: CRA identifies spurious shortcuts but also analyses mixed representations where removing bias harms clinical utility.
- **Not just attribution**: CRA goes beyond token-level attribution to representation-level *intervention*, providing causal (not correlational) measures.

---

## Research Question

> *Which hidden representations inside clinical LLMs causally contribute to clinical predictions, which representations are merely correlated or spurious, and which representations encode privacy-sensitive patient information?*

---

## Repository Structure

```
CausalLearningLLM/
├── configs/               # YAML experiment configs
│   ├── debug.yaml         # Fast debug run (50 samples, distilgpt2)
│   ├── pubmedqa.yaml      # PubMedQA experiment
│   ├── synthetic.yaml     # Synthetic dataset
│   ├── mimic_placeholder.yaml
│   └── ...
├── src/
│   ├── data/              # Dataset loaders + attribute injection + counterfactuals
│   ├── models/            # HuggingFace LLM wrapper + answer scoring + hooks
│   ├── representations/   # Hidden state extraction, pooling, geometry
│   ├── probing/           # Probe models (logistic, SVM) + INLP + directions
│   ├── causal/            # CRA framework, interventions, effect estimators, SCM
│   ├── mediation/         # Causal mediation analysis via activation patching
│   ├── privacy/           # Leakage metrics, conditional leakage, attribute inference
│   ├── spurious/          # Shortcut metrics, robustness evaluation
│   ├── theory/            # Theorem verification experiments
│   ├── evaluation/        # Experiment runner, metrics, summarization
│   ├── statistics/        # Bootstrap CI, paired t-test, effect sizes
│   ├── visualization/     # All 17 paper figures + LaTeX tables
│   └── utils/             # Config, logging, seed, device, I/O
├── scripts/               # CLI scripts for each pipeline stage
│   ├── run_all_debug.py   # End-to-end debug (START HERE)
│   ├── run_full_pipeline.py
│   ├── extract_representations.py
│   ├── train_probes.py
│   ├── verify_theorems.py
│   ├── generate_figures.py
│   └── generate_tables.py
├── tests/                 # 30 pytest tests
└── outputs/               # All generated artifacts (gitignored)
```

---

## Installation

```bash
# Create environment
conda env create -f environment.yml
conda activate cra_clinical

# Or with pip
pip install -r requirements.txt
```

**Note:** Requires Python 3.10+. The codebase runs on CPU and GPU (auto-detected). For float16, set `model.dtype: float16` in the config.

---

## Quick Debug Run

Runs the complete pipeline on 50 synthetic samples using `distilgpt2` (~2 min on CPU):

```bash
python scripts/run_all_debug.py --config configs/debug.yaml
```

This will:
1. Generate synthetic clinical QA data with known clinical/sensitive/spurious factors
2. Inject controlled sensitive attributes (age_group, gender) and spurious attributes (hospital_type)
3. Generate counterfactual pairs
4. Load `distilgpt2` and extract hidden representations from selected layers
5. Train logistic regression probes for clinical label Y, sensitive attribute S, spurious attribute C
6. Run CRA interventions across directions, layers, and λ values
7. Compute all metrics (Clin, Leak, Spur, CE_Y, PUS, RUS, CRA score)
8. Verify 6 theoretical claims empirically
9. Generate figures 1–3, 5–7, 15–16
10. Generate LaTeX tables 1–4

Outputs are saved to `outputs/debug/`.

---

## Full Pipeline

```bash
python scripts/run_full_pipeline.py --config configs/pubmedqa.yaml
```

---

## Step-by-step Usage

### 1. Extract Representations

```bash
python scripts/extract_representations.py --config configs/pubmedqa.yaml
```

Saves `.npy` arrays and `.json` metadata to `outputs/pubmedqa/representations/`.

### 2. Train Probes

```bash
python scripts/train_probes.py --config configs/pubmedqa.yaml
```

Trains probes q_y(Y|Z_l), q_s(S|Z_l), q_c(C|Z_l) for each layer. Saves probe objects and `probe_metrics.csv`.

### 3. Run CRA Interventions

CRA interventions are run as part of `run_full_pipeline.py` or `run_all_debug.py`. The core class is:

```python
from src.causal.cra import CRAFramework

cra = CRAFramework(layer_data, probe_results, config,
                   sensitive_attr="age_group", spurious_attr="hospital_type")
df = cra.run_full_intervention_study(output_dir="outputs/interventions")
```

### 4. Theorem Verification

```bash
python scripts/verify_theorems.py --config configs/debug.yaml
```

### 5. Generate Figures and Tables

```bash
python scripts/generate_figures.py --config configs/pubmedqa.yaml
python scripts/generate_tables.py --config configs/pubmedqa.yaml
```

---

## Metric Definitions

| Metric | Definition |
|--------|-----------|
| `Clin(Z_l)` | AUROC of clinical probe q_y(Y\|Z_l) |
| `Leak(Z_l)` | AUROC of sensitive probe q_s(S\|Z_l) |
| `Spur(Z_l)` | AUROC of spurious probe q_c(C\|Z_l) |
| `CE_Y` | Mean change in correct-class probability after intervention |
| `Δleak` | Leak(Z_l) − Leak(Z'_l) — leakage reduction after sensitive intervention |
| `Δspur` | Spur(Z_l) − Spur(Z'_l) — spuriousness reduction |
| `Δtask` | L_task(Z'_l) − L_task(Z_l) — utility loss |
| `PUS` | Privacy-Utility Score = Δleak − β·Δtask |
| `RUS` | Robustness-Utility Score = Δspur − η·Δtask |
| `CRA score` | α_y\|CE_Y\| + α_p·Δleak + α_c·Δspur − γ·Δtask |

---

## Representation Taxonomy

Each layer-direction combination is classified into:

| Category | Criteria |
|----------|---------|
| **Clinically useful** | Clin ≥ 0.6, Leak < 0.55, Spur < 0.55 |
| **Privacy-sensitive** | Leak ≥ 0.55 |
| **Spurious** | Spur ≥ 0.55, low counterfactual consistency |
| **Mixed clinical-sensitive** | Clin ≥ 0.6 AND Leak ≥ 0.55 |
| **Mixed clinical-spurious** | Clin ≥ 0.6 AND Spur ≥ 0.55 |
| **Highly entangled** | Clin, Leak, Spur all high; cos(v_y, v_s) high |
| **Nuisance** | All scores low |

---

## Theoretical Claims

CRA empirically verifies 6 claims:

1. **Leakage reduction**: Sensitive direction intervention reduces Leak(Z)
2. **Utility-orthogonality**: Utility loss after sensitive intervention correlates with |cos(v_y, v_s)|
3. **Spurious robustness**: Spurious direction removal improves counterfactual consistency
4. **Random projection**: Random directions have low expected alignment with sensitive subspace (≈ k/d)
5. **Tradeoff dominance**: CRA directions achieve better PUS/RUS than random/correlation baselines
6. **Conditional leakage**: Conditional Leak(Z;S|Y) reveals residual sensitive information beyond clinical label

---

## Limitations

1. Synthetic sensitive attributes are experimental controls, not real patient demographics
2. Probe-space interventions are approximations of full model-space interventions
3. Linear probes measure linear decodability only
4. Causal claims depend on the validity of the counterfactual construction
5. Mutual information estimates are proxies
6. Clinical utility proxied by QA accuracy ≠ real clinical safety
7. Framework is for research analysis, not clinical deployment

---

## Expected Outputs

After a full run, `outputs/{dataset}/` contains:

```
representations/          # layer_{l}_Z.npy, layer_{l}_meta.json
probes/                   # probe_metrics.csv, layer_{l}_{task}_probe.pkl
interventions/            # cra_intervention_results.csv
metrics/                  # layer_information.csv, direction_geometry.csv,
                          # representation_taxonomy.csv, privacy_utility_curve.csv
mediation/                # mediation_results.csv
statistics/               # theorem_claim{1-6}_*.csv
tables/                   # table{1-4}.tex, table{1-4}.csv
figures/pdf/              # fig{1-16}.pdf
figures/svg/              # fig{1-16}.svg
figures/png/              # fig{1-16}.png
```

---

## Running Tests

```bash
python3 -m pytest tests/ -v
```

30 tests covering: dataset loading, attribute injection, counterfactual generation, representation extraction, projection correctness, probe training, metrics, taxonomy, and visualization.

---

## Citation

```bibtex
@article{cra2025,
  title={Understanding Clinical Reasoning through Causal Representation Attribution in Large Language Models},
  author={...},
  year={2025}
}
```
