"""kindle: a continually self-training RL agent.

The public API is re-exported from the native pyo3 extension built via
maturin (``kindle._native``).
"""

from ._native import Agent, OBS_TOKEN_DIM  # type: ignore[attr-defined]

__all__ = ["Agent", "OBS_TOKEN_DIM"]
