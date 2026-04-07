from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _load_legacy_module():
    repo_root = Path(__file__).resolve().parents[2]
    legacy_path = repo_root / 'robust_pricing_rewritten_best_effort.py'
    spec = spec_from_file_location('robust_pricing_legacy_core', legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to import legacy core from {legacy_path}')
    module = module_from_spec(spec)
    sys.modules['robust_pricing_legacy_core'] = module
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_module()

AmbiguitySet = _legacy.AmbiguitySet
MODELS = _legacy.MODELS
gamma_bar = _legacy.gamma_bar
gamma_bar_inv = _legacy.gamma_bar_inv
gbar = _legacy.gbar
wilson_interval = _legacy.wilson_interval
gaussian_ci = _legacy.gaussian_ci

__all__ = [
    'AmbiguitySet',
    'MODELS',
    'gamma_bar',
    'gamma_bar_inv',
    'gbar',
    'wilson_interval',
    'gaussian_ci',
]
