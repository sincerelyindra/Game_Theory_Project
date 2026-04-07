# Report Alignment

This note maps the repository layout to the uploaded report **"Robust Quantile-Based Pricing with Minimal Information: A Reproduction Study, Historical Backtest, and Adaptive Interval Extension"**.

## Repository-to-report mapping

### 1. Reproduction study
- **Report sections:** Abstract, Sections 5.1, 6.1, 7.1, 7.2
- **Primary script:** `scripts/01_reproduce_paper.py`
- **Legacy implementation source:** `robust_pricing_rewritten_best_effort.py`
- **Expected raw outputs:** `results/reproduction/raw/results_quick.zip`

This track covers the best-effort robust solver, feasibility experiments, and the synthetic benchmark against parametric baselines.

### 2. Rolling historical backtest
- **Report sections:** Sections 5.3, 6.2, 7.3, Appendix B.1
- **Primary script:** `scripts/02_rolling_backtest.py`
- **Legacy implementation source:** `tuna_colab_backtest.py`
- **Expected raw outputs:** `results/backtest/raw/tuna_backtest_results(1).zip`
- **Expected raw data:** `data/raw/upctna.zip`, `data/raw/wtna.zip`

This track covers the Dominick's-style canned tuna backtest, rolling train/test windows, and evaluation against ERM and parametric baselines.

### 3. Adaptive confidence-interval extension
- **Report sections:** Sections 5.4, 6.3, 7.4, Appendix B.2
- **Primary script:** `scripts/03_adaptive_ci_experiments.py`
- **Legacy implementation source:** `part_c_adaptive_ci_experiment.py`
- **Expected raw outputs:** `results/adaptive_ci/raw/part_c_quick.zip`

This track covers fixed-CI baselines, AdaptiveFeasible, and AdaptiveValidated.

## Why this repo structure exists

The README describes a research-style layout that separates:
- reusable Python code in `src/robust_pricing/`,
- runnable experiment entrypoints in `scripts/`,
- raw datasets in `data/raw/`, and
- raw result bundles in `results/.../raw/`.

That layout makes the project easier to read, rerun, and grade while keeping a clean one-to-one mapping between the code and the write-up.

## Compatibility note

The original top-level scripts are intentionally preserved for backward compatibility. The new numbered scripts under `scripts/` act as stable entrypoints that reflect the report structure without breaking older references.
