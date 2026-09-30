"""Compatibility import for the legacy Ising converter.

The implementation now lives with the other coarse-graining entry points in
``deepmesod.data.coarse_graining``.
"""

from .coarse_graining import convert_legacy_ising_trajectory

__all__ = ["convert_legacy_ising_trajectory"]
