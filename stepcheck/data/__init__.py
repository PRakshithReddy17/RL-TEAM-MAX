"""Data generation and preprocessing module."""

from .generator import (
    SyntheticConfig,
    Trail,
    generate_synthetic_data,
    create_splits,
)

__all__ = [
    "SyntheticConfig",
    "Trail",
    "generate_synthetic_data",
    "create_splits",
]