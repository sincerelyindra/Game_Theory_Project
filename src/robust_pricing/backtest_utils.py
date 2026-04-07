from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _load_backtest_module():
    repo_root = Path(__file__).resolve().parents[2]
    legacy_path = repo_root / 'tuna_colab_backtest.py'
    spec = spec_from_file_location('robust_pricing_legacy_backtest', legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to import legacy backtest module from {legacy_path}')
    module = module_from_spec(spec)
    sys.modules['robust_pricing_legacy_backtest'] = module
    spec.loader.exec_module(module)
    return module


_legacy_backtest = _load_backtest_module()

rolling_windows = _legacy_backtest.rolling_windows
choose_anchor_prices = _legacy_backtest.choose_anchor_prices
choose_candidate_prices = _legacy_backtest.choose_candidate_prices
attach_proxy = _legacy_backtest.attach_proxy
summarize_anchor_quantiles = _legacy_backtest.summarize_anchor_quantiles
price_table = _legacy_backtest.price_table
nearest_test_price = _legacy_backtest.nearest_test_price
run_backtest = _legacy_backtest.run_backtest

__all__ = [
    'rolling_windows',
    'choose_anchor_prices',
    'choose_candidate_prices',
    'attach_proxy',
    'summarize_anchor_quantiles',
    'price_table',
    'nearest_test_price',
    'run_backtest',
]
