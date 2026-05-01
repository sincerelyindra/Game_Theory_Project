# Robust Quantile-Based Pricing Submission Package

This directory contains the NeurIPS-2021-style submission package and experiment outputs for the robust quantile-pricing project.

## What is included

- `paper/main.tex`: NeurIPS-style LaTeX report source.
- `code/`: implementation scripts for the numerical robust-pricing solver, adaptive-CI experiment, and Dominick's tuna backtest.
- `results/`: CSV result tables extracted from the experiment bundles: feasibility reproduction, synthetic benchmark, sequential experimentation, bandit comparison, adaptive CI, and tuna backtest summaries.
- `requirements.txt`: minimal Python dependencies.

## What is intentionally excluded

The raw scanner-data archives `wtna.zip` and `upctna.zip` are not included here. This package is meant to store code and produced results, not redistribute the raw data.

The numerical solver is an implementation-oriented reproduction. It follows the generalized Pareto interpolation, interval tightening, envelope construction, and reduced robust price-search structure, but evaluates the reduced Nature problem on dense grids rather than implementing the original paper's exact appendix-level cubic-equation subroutines.
