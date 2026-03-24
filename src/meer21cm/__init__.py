"""
Top-level package for ``meer21cm``.

This module exposes the main public classes while avoiding importing
heavy submodules during ``import meer21cm``. The actual classes are
loaded lazily on first access.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Specification",
    "CosmologyCalculator",
    "PowerSpectrum",
    "MockSimulation",
]


_LAZY_ATTRS = {
    "Specification": ("meer21cm.dataanalysis", "Specification"),
    "CosmologyCalculator": ("meer21cm.cosmology", "CosmologyCalculator"),
    "PowerSpectrum": ("meer21cm.power", "PowerSpectrum"),
    "MockSimulation": ("meer21cm.mock", "MockSimulation"),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import top-level attributes when accessed via ``meer21cm.<name>``.
    """
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'meer21cm' has no attribute {name!r}")


def __dir__() -> list[str]:
    """
    Ensure ``dir(meer21cm)`` shows lazily available attributes.
    """
    return sorted(set(globals().keys()) | set(_LAZY_ATTRS.keys()))
