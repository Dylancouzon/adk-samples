"""
Evaluation templates and runners for the financial_advisor agent.

This package organizes reusable eval templates and simple runners so that
additional evals can be added incrementally without changing call sites.
"""

# Re-export templates for convenience
from . import templates  # noqa: F401

__all__ = [
    "templates",
]


