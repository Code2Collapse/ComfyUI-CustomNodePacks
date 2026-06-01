"""Shared utilities for the c2c_ai spine (pure-Python, no torch deps)."""
from .qwen3_filter import strip_think, THINK_RE

__all__ = ["strip_think", "THINK_RE"]
