"""Structured compression experiments for Transformer linear layers."""

from .approximations import ApproximationResult
from .replacement import StructuredLinear

__all__ = ["ApproximationResult", "StructuredLinear"]
