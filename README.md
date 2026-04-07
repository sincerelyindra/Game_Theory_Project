# Robust Quantile-Based Pricing: Reproduction, Rolling Backtest, and Adaptive CI Extension

This repository contains the code, data pointers, and result artifacts for a three-part study of robust quantile-based pricing under minimal information:

1. **Paper reproduction** of the core robust-pricing framework and synthetic experiments.
2. **Rolling historical backtest** on scanner-style retail data.
3. **Adaptive confidence-interval extension** for feasibility-aware calibration.

The repository is organized to mirror the report structure so that each script, dataset, and result bundle maps cleanly to a section of the write-up.

---

## Repository layout

```text
Game_Theory_Project/
├── README.md
├── .gitignore
├── requirements.txt
├── docs/
│   └── REPORT_ALIGNMENT.md
├── data/
│   └── raw/
│       ├── upctna.zip
│       ├── wtna.zip
│       └── README.md
├── notebooks/
├── results/
│   ├── reproduction/
│   │   ├── raw/
│   │   │   └── results_quick.zip
│   │   └── README.md
│   ├── backtest/
│   │   ├── raw/
│   │   │   └── tuna_backtest_results(1).zip
│   │   └── README.md
│   └── adaptive_ci/
│       ├── raw/
│       │   └── part_c_quick.zip
│       └── README.md
├── scripts/
│   ├── 01_reproduce_paper.py
│   ├── 02_rolling_backtest.py
│   └── 03_adaptive_ci_experiments.py
├── src/
│   └── robust_pricing/
│       ├── __init__.py
│       ├── core.py
│       ├── backtest_utils.py
│       ├── ci_utils.py
│       └── baselines.py
└── tests/
    ├── test_core.py
    ├── test_backtest.py
    └── test_ci_utils.py
```

---

## File mapping from your current repo

| Current file | Recommended location | Purpose |
|---|---|---|
| `robust_pricing_rewritten_best_effort.py` | `scripts/01_reproduce_paper.py` | Main reproduction script for the paper’s robust pricing framework |
| `tuna_colab_backtest.py` | `scripts/02_rolling_backtest.py` | Rolling historical backtest pipeline |
| `part_c_adaptive_ci_experiment.py` | `scripts/03_adaptive_ci_experiments.py` | Adaptive confidence-interval extension |
| `results_quick.zip` | `results/reproduction/raw/` | Output bundle for paper reproduction |
| `tuna_backtest_results(1).zip` | `results/backtest/raw/` | Output bundle for rolling backtest |
| `part_c_quick.zip` | `results/adaptive_ci/raw/` | Output bundle for adaptive CI extension |
| `upctna.zip` | `data/raw/` | Dataset archive |
| `wtna.zip` | `data/raw/` | Dataset archive |

---

## Recommended script responsibilities

### `scripts/01_reproduce_paper.py`
Use this script for:
- ambiguity-set construction
- interval tightening
- generalized Pareto interpolation
- feasibility experiments
- synthetic benchmark against parametric baselines

Expected outputs:
- feasibility tables
- synthetic benchmark summaries
- plots for reproduction experiments

### `scripts/02_rolling_backtest.py`
Use this script for:
- loading and cleaning scanner-style data
- building rolling train/test windows
- computing anchor prices and purchase-probability proxies
- evaluating ERM, linear, quadratic, logit, and robust methods

Expected outputs:
- backtest summary tables
- per-window performance metrics
- method comparison plots

### `scripts/03_adaptive_ci_experiments.py`
Use this script for:
- comparing fixed confidence levels (90%, 95%, 99%)
- running AdaptiveFeasible and AdaptiveValidated rules
- summarizing feasibility, width, and revenue-ratio trade-offs

Expected outputs:
- adaptive CI result tables
- mean ratio / feasibility / width plots

---

## Suggested modular refactor

If you want the repo to look more research-grade, move reusable logic out of the large scripts and into `src/robust_pricing/`:

- `core.py`: ambiguity set, envelope construction, robust value search
- `backtest_utils.py`: rolling windows, nearest-observed-price evaluation, data cleaning
- `ci_utils.py`: Wilson / Gaussian intervals, adaptive CI selection rules
- `baselines.py`: ERM, linear, quadratic, and logit benchmark models

This keeps `scripts/` thin and makes the project easier to test and extend.

---

## Reproducing the study

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place data and result archives

Put the following files into the locations shown below:

```text
data/raw/upctna.zip
 data/raw/wtna.zip
 results/reproduction/raw/results_quick.zip
 results/backtest/raw/tuna_backtest_results(1).zip
 results/adaptive_ci/raw/part_c_quick.zip
```

### 3. Run each component

```bash
python scripts/01_reproduce_paper.py
python scripts/02_rolling_backtest.py
python scripts/03_adaptive_ci_experiments.py
```

---


