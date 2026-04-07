from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

from .core import gaussian_ci, wilson_interval


def _load_part_c_module():
    repo_root = Path(__file__).resolve().parents[2]
    legacy_path = repo_root / 'part_c_adaptive_ci_experiment.py'
    spec = spec_from_file_location('robust_pricing_legacy_part_c', legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Unable to import Part C module from {legacy_path}')
    module = module_from_spec(spec)
    sys.modules['robust_pricing_legacy_part_c'] = module
    spec.loader.exec_module(module)
    return module


_legacy_part_c = _load_part_c_module()

MethodResult = _legacy_part_c.MethodResult
build_intervals_for_level = _legacy_part_c.build_intervals_for_level
average_width = _legacy_part_c.average_width
choose_adaptive_feasible = _legacy_part_c.choose_adaptive_feasible
choose_adaptive_validated = _legacy_part_c.choose_adaptive_validated

__all__ = [
    'gaussian_ci',
    'wilson_interval',
    'MethodResult',
    'build_intervals_for_level',
    'average_width',
    'choose_adaptive_feasible',
    'choose_adaptive_validated',
]
